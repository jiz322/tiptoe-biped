"""Minimal tfevents scalar reader (no tensorboard dependency).

Parses TFRecord framing + Event/Summary protobuf wire format by hand;
handles simple_value scalars as written by torch SummaryWriter.
"""
import struct
import sys
import glob
import os

def varint(buf, i):
    r = 0
    s = 0
    while True:
        b = buf[i]
        i += 1
        r |= (b & 0x7F) << s
        if not b & 0x80:
            return r, i
        s += 7

def fields(buf):
    """Yield (field_no, wire_type, value_bytes_or_int) over a proto message."""
    i = 0
    n = len(buf)
    while i < n:
        key, i = varint(buf, i)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(buf, i)
        elif wt == 1:
            v = buf[i:i + 8]; i += 8
        elif wt == 2:
            ln, i = varint(buf, i)
            v = buf[i:i + ln]; i += ln
        elif wt == 5:
            v = buf[i:i + 4]; i += 4
        else:
            return
        yield fno, wt, v

def read_events(path):
    """Yield (step, tag, value) from a tfevents file."""
    with open(path, "rb") as f:
        data = f.read()
    i = 0
    n = len(data)
    while i + 12 <= n:
        (ln,) = struct.unpack_from("<Q", data, i)
        i += 12  # len + len-crc
        if i + ln > n:
            break
        rec = data[i:i + ln]
        i += ln + 4  # data + data-crc
        step = 0
        summ = None
        for fno, wt, v in fields(rec):
            if fno == 2 and wt == 0:
                step = v
            elif fno == 5 and wt == 2:
                summ = v
        if summ is None:
            continue
        for fno, wt, v in fields(summ):
            if fno == 1 and wt == 2:  # Summary.Value
                tag = None
                val = None
                for f2, w2, v2 in fields(v):
                    if f2 == 1 and w2 == 2:
                        tag = v2.decode("utf-8", "replace")
                    elif f2 == 2 and w2 == 5:
                        (val,) = struct.unpack("<f", v2)
                if tag is not None and val is not None:
                    yield step, tag, val

RUNS = {
    "A_pgbias010": r"logs\rsl_rl\p9_replant\2026-08-18_16-06-53",
    "B_pgbias003": r"logs\rsl_rl\p9_replant\2026-08-18_20-19-08_B_pgbias003",
}

if __name__ != "__main__":
    RUNS = dict(RUNS)  # imported: skip the dump below
    import sys as _sys
    _sys.modules[__name__].__dict__.setdefault("_imported", True)

for name, d in (RUNS.items() if __name__ == "__main__" else []):
    print(f"\n===== {name} =====")
    series = {}
    for p in sorted(glob.glob(os.path.join(d, "events.out.tfevents.*"))):
        for step, tag, val in read_events(p):
            series.setdefault(tag, []).append((step, val))
    for tag in sorted(series):
        pts = series[tag]
        n = len(pts)
        idxs = sorted(set(int(k * (n - 1) / 11) for k in range(12)))
        line = ", ".join(f"{pts[j][0]}:{pts[j][1]:.4g}" for j in idxs)
        print(f"{tag} [n={n}] {line}")
