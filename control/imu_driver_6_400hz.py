"""
YIS IMU Driver - 6-Axis Mode (ACC + GYRO + EULER) @ 400Hz
Protocol: Yesense YIS v2.3
Baud: 460800 | Freq: 400Hz (code 0x0A) | Output: ACC+GYRO+EULER (0x00D0)
"""
import serial
import struct
import time
import threading
import math

# ================= 配置 =================
IMU_PORT = '/dev/ttyAMA0'
BAUD_RATE = 460800

# YIS 协议常量
HEADER = bytes([0x59, 0x53])

# 输出内容位掩码
BIT_ACC   = (1 << 7)   # 加速度
BIT_GYRO  = (1 << 6)   # 角速度
BIT_EULER = (1 << 4)   # 欧拉角

# 6轴输出: ACC + GYRO + EULER
OUTPUT_MASK_6AXIS = BIT_ACC | BIT_GYRO | BIT_EULER  # 0x00D0

# 数据ID定义
ID_ACC   = 0x10
ID_GYRO  = 0x20
ID_EULER = 0x40

# 数据缩放因子 (协议: int32 * 0.000001)
SCALE = 0.000001


class IMUDriver6Axis:
    def __init__(self, port, baudrate=460800):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.frame_count = 0
        self._last_fps_time = 0
        self._last_fps_count = 0
        self.fps = 0.0

        # 数据容器
        self.data = {
            'acc':   [0.0, 0.0, 0.0],   # m/s²
            'gyro':  [0.0, 0.0, 0.0],   # deg/s
            'euler': [0.0, 0.0, 0.0],   # deg (pitch, roll, yaw)
        }

    # ─────────── 协议工具函数 ───────────

    @staticmethod
    def _calc_checksum(data):
        """Fletcher 校验和 (CK1, CK2)"""
        ck1, ck2 = 0, 0
        for b in data:
            ck1 = (ck1 + b) & 0xFF
            ck2 = (ck2 + ck1) & 0xFF
        return ck1, ck2

    @staticmethod
    def _build_cmd(data_class, operator, data_bytes):
        """
        构建交互指令帧
        格式: Header(2) + DataClass(1) + OpLen byte1(1) + OpLen byte2(1) + Data(N) + CK1 + CK2
        """
        data_len = len(data_bytes)
        byte1 = ((data_len & 0x1F) << 3) | (operator & 0x07)
        byte2 = (data_len >> 5) & 0xFF
        payload = bytes([data_class, byte1, byte2]) + bytes(data_bytes)
        ck1, ck2 = IMUDriver6Axis._calc_checksum(payload)
        return HEADER + payload + bytes([ck1, ck2])

    def _send_cmd(self, data_class, operator, data_bytes):
        """发送交互指令"""
        cmd = self._build_cmd(data_class, operator, data_bytes)
        if self.ser and self.ser.is_open:
            self.ser.write(cmd)
            time.sleep(0.2)

    # ─────────── 配置函数 ───────────

    def _set_output(self, bitmask):
        """设置输出内容 (写入RAM)"""
        lo = bitmask & 0xFF
        hi = (bitmask >> 8) & 0xFF
        self._send_cmd(0x04, 0x01, [lo, hi])

    def _set_freq(self, code):
        """设置输出频率 (写入RAM)"""
        self._send_cmd(0x03, 0x01, [code])

    def _set_baud(self, code):
        """设置波特率 (写入RAM)"""
        self._send_cmd(0x02, 0x01, [code])

    def _set_mode_vru(self):
        """设置算法模式为 VRU (6轴, 无磁力计, yaw初始化为0)"""
        self._send_cmd(0x4D, 0x01, [0x02, 0x02])  # 子类型0x02(算法模式), 值0x02=VRU

    def _init_gyro_bias(self):
        """陀螺零偏初始化 (设备自采集, 需保持静止)"""
        self._send_cmd(0x4D, 0x01, [0x50, 0x01])  # 子类型0x50, 值0x01=设备自采集

    # ─────────── 打开/关闭 ───────────

    def open(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            print(f"[IMU-6Axis] 串口 {self.port} @ {self.baudrate} 打开成功")
            time.sleep(0.5)

            # 配置: VRU 模式 (6轴, 不使用磁力计, yaw从0开始)
            print("[IMU-6Axis] 设置算法模式: VRU (6轴)")
            self._set_mode_vru()
            time.sleep(0.3)

            # 陀螺零偏初始化 (设备自采集, 需静止约1秒)
            print("[IMU-6Axis] 陀螺零偏初始化...")
            self._init_gyro_bias()
            time.sleep(1.0)

            # 配置: 6轴输出 (ACC + GYRO + EULER)
            print("[IMU-6Axis] 设置输出内容: ACC + GYRO + EULER")
            self._set_output(OUTPUT_MASK_6AXIS)
            time.sleep(0.3)

            # 配置: 400Hz
            print("[IMU-6Axis] 设置输出频率: 400Hz (code 0x0A)")
            self._set_freq(0x0A)
            time.sleep(0.3)

            # 启动读取线程
            self.running = True
            self._last_fps_time = time.time()
            self._last_fps_count = 0
            self.thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.thread.start()
            print("[IMU-6Axis] 初始化完成, 400Hz 6轴模式运行中")
            return True
        except Exception as e:
            print(f"[IMU-6Axis] 打开失败: {e}")
            return False

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.ser and self.ser.is_open:
            # 恢复100Hz再关闭
            self._set_freq(0x08)
            time.sleep(0.1)
            self.ser.close()
            print("[IMU-6Axis] 串口已关闭")

    # ─────────── 数据读取与解析 ───────────

    def _reader_loop(self):
        """后台持续读取并解析 YIS 协议帧"""
        buf = b''
        while self.running:
            try:
                if self.ser.in_waiting:
                    buf += self.ser.read(self.ser.in_waiting)

                while len(buf) >= 7:  # 最小帧: Header(2)+TID(2)+LEN(1)+CK1(1)+CK2(1)
                    # 寻找帧头 0x59 0x53
                    idx = buf.find(HEADER)
                    if idx < 0:
                        buf = buf[-1:]  # 保留最后1字节防断帧头
                        break
                    if idx > 0:
                        buf = buf[idx:]
                        continue

                    # 检查长度字段是否可读
                    if len(buf) < 5:
                        break

                    # TID: buf[2:4], LEN: buf[4]
                    data_len = buf[4]
                    frame_len = 5 + data_len + 2  # Header(2)+TID(2)+LEN(1)+DATA(N)+CK(2)

                    if len(buf) < frame_len:
                        break

                    frame = buf[:frame_len]
                    buf = buf[frame_len:]

                    # 校验: 范围从 TID 到 DATA 末尾
                    ck_data = frame[2:5 + data_len]
                    ck1, ck2 = self._calc_checksum(ck_data)
                    if ck1 != frame[-2] or ck2 != frame[-1]:
                        continue  # 校验失败, 丢弃

                    # 解析有效数据域
                    self._parse_data(frame[5:5 + data_len])
                    self.frame_count += 1

                    # 计算实时FPS
                    now = time.time()
                    dt = now - self._last_fps_time
                    if dt >= 1.0:
                        self.fps = (self.frame_count - self._last_fps_count) / dt
                        self._last_fps_count = self.frame_count
                        self._last_fps_time = now

                time.sleep(0.0002)  # 400Hz需要更短的sleep
            except Exception as e:
                print(f"[IMU-6Axis] 读取错误: {e}")
                time.sleep(0.5)

    def _parse_data(self, payload):
        """解析 TLV 格式数据域"""
        offset = 0
        while offset + 2 <= len(payload):
            data_id = payload[offset]
            data_len = payload[offset + 1]
            offset += 2

            if offset + data_len > len(payload):
                break

            chunk = payload[offset:offset + data_len]
            offset += data_len

            if data_id == ID_ACC and data_len >= 12:
                ax, ay, az = struct.unpack('<iii', chunk[:12])
                with self.lock:
                    self.data['acc'] = [ax * SCALE, ay * SCALE, az * SCALE]

            elif data_id == ID_GYRO and data_len >= 12:
                gx, gy, gz = struct.unpack('<iii', chunk[:12])
                with self.lock:
                    self.data['gyro'] = [gx * SCALE, gy * SCALE, gz * SCALE]

            elif data_id == ID_EULER and data_len >= 12:
                pitch, roll, yaw = struct.unpack('<iii', chunk[:12])
                with self.lock:
                    self.data['euler'] = [pitch * SCALE, roll * SCALE, yaw * SCALE]

    # ─────────── 公开接口 ───────────

    def get_data(self):
        """线程安全地获取最新数据副本"""
        with self.lock:
            return {k: list(v) for k, v in self.data.items()}

    def get_acc(self):
        with self.lock:
            return list(self.data['acc'])

    def get_gyro(self):
        with self.lock:
            return list(self.data['gyro'])

    def get_euler(self):
        with self.lock:
            return list(self.data['euler'])


# ================= 主程序 =================
if __name__ == "__main__":
    imu = IMUDriver6Axis(IMU_PORT, BAUD_RATE)

    if imu.open():
        time.sleep(1)
        print("=" * 70)
        print("  6-Axis 400Hz IMU 实时数据 (ACC + GYRO + EULER)")
        print("  按 Ctrl+C 停止")
        print("=" * 70)
        header = (f"{'Time':>7} | {'Pitch':>9} {'Roll':>9} {'Yaw':>10} | "
                  f"{'AccX':>8} {'AccY':>8} {'AccZ':>8} | {'FPS':>6}")
        print(header)
        print("-" * len(header))

        try:
            start_t = time.time()
            while True:
                d = imu.get_data()
                e = d['euler']
                a = d['acc']
                t = time.time() - start_t

                print(f"{t:6.1f}s | {e[0]:9.3f} {e[1]:9.3f} {e[2]:10.3f} | "
                      f"{a[0]:8.4f} {a[1]:8.4f} {a[2]:8.4f} | {imu.fps:6.1f}",
                      end='\r')
                time.sleep(0.05)  # 20Hz 显示刷新

        except KeyboardInterrupt:
            print(f"\n\n总帧数: {imu.frame_count}")
        finally:
            imu.close()