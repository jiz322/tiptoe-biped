Flight 2026-10-16: S11 (s11_11880) stand-up from d=0.6, heading off, grace 2.5, blind handover. 2 trials, 0 stand-ups.
Both fell to the RIGHT (roll -36 / -39 deg at 0.5 s) and both skipped R_knee belt teeth (user).
R_roll (106) is the NEW EL05 unit installed 10-15.

Source: pi:~/Documents/big_thigh/control/ (scp'd 10-16). Deploy code as run is byte-identical to the 10-14 as-run
copy (data/flight_20261014/asrun): balance_s11.py, s11_contract.py, config.py, motor_bridge.py, s11_11880.ts.pt (md5).

id  file                                    n    fell   note
F1  s11_11880_s11_log_20261016_110739.npz   150  0.51s  crawl after: R_knee -59.64 deg (expected +11.5) = -71 deg = ~6 teeth
F2  s11_11880_s11_log_20261016_111630.npz   133  0.45s  crawl after: R_knee -12.11 deg (expected +11.5) = -24 deg = ~2 teeth
Crawl reference (config CRAWL_JOINT_POS): L_knee -0.20 rad, R_knee +0.20 rad (L = -R, NOT L = R).
Both offsets are negative = the motor ran toward EXTENSION past the shank = slipped while carrying body weight.
ASSUMED: setzero_crawl / belt re-seat between F1 and F2 (else F2 could not have flown) -> F2's 2 teeth are new.

Question: hardware (R belt drive holds less) vs software (policy/deploy demanded more torque)?  Scripts: _load.py,
_opening.py (python), outputs _load.json / _opening.json. References: 10-11 A1..B3 (crawl-checked,
no skip), 10-14 T1..T10 (skip state never checked).

Pre-registered
  H-SW  today's R_knee load is above the no-skip envelope and above today's L_knee.
  H-HW  today's R_knee load is inside the envelope and/or at or below today's L_knee (which held).
Result: H-SW CONTRADICTED, H-HW SUPPORTED.
  - Commanded knee torque hits the 17 Nm motor clamp in all 17 trials; R clamp time F1 30 ms / F2 0 ms vs 10-11 10-40.
    Raw R peak F1 21.9 / F2 16.9 Nm vs 10-11 17.0-21.1. F2 is the LOWEST R load of all 17 trials and still skipped.
  - torque_fdb R peak (handover to fall) F1 10.7 / F2 10.1 Nm, BELOW every 10-11 and 10-14 R trial (11.4-12.2).
    Same instant, L_knee carried 12.7 / 12.3 Nm and held.
  - The opening is deterministic (blind handover): knee targets for ticks 0-2 match to 0.01 rad in all 17 trials.
    Through tick 4, F1's targets sit inside the no-skip spread. F2's R target is 0.09 rad more extended at tick 4
    (about +2 Nm motor), well under the clamp.
POST HOC signature (found after looking, not pre-registered): at 0.05-0.08 s the R_knee motor-side reading extends
at 10.7-11.6 rad/s while L moves at 6-8 rad/s, and R torque_fdb collapses (F2 3.9 vs L 12.3 Nm at 0.06 s). That
looks like the motor freewheeling over teeth. 10-11 no-skip: R 5.7-7.8 rad/s (except A2, the stall trial).
Also present on 10-14 T5 (S10b, 151126: 11.2 rad/s, 5.3 vs 12.1 Nm) and T6 (S12, 151500), the two "most violent"
roll -37 deg trials -> R slipping likely began 10-14, with the OLD R_roll motor.
The right fall is downstream of the slip (R leg left more flexed than commanded -> body drops right).
Tooth counts include fall-phase skips (F1 raw R cmd -26..+22 Nm after 0.42 s, while already rolling).
Caveats: torque_fdb is filtered ~25-35 ms (a lower bound). Motor-side channels cannot see a skip directly, so the
onset time comes from the signature only.

10-16 user inspection (disassembled; physical ground truth)
------------------------------------------------------------
R belt (HTD 5M PU, steel cord): teeth at the skip site no longer straight across the belt width; 3 of 7 steel cords
snapped. It skipped even with the 10-11 idler pressing on it. L belt looks relatively healthy.
User history: after 10-11 (stood), shims were added to centre the belts off the flanges, many more on the R knee than
the L. 10-14/15 flights: few skips. 10-16: screws tightened and R shims reduced before flying.
User hypothesis (their instinct): the excess R shims caused unmodeled leg deformation/bending -> the 10-14/15 failures
and the loose screws. Adding and then removing shims snapped cords first; skipping and tooth deformation followed.
Not testable from npz (motor-side channels see neither leg geometry nor cord state).
Open: are the 3 snapped cords at one EDGE (which side: flange/shim side?) or spread across the width?

10-16 later: NEW R belt fitted, 2 more trials (npz not yet copied here)
-----------------------------------------------------------------------
Crawl readouts after each (user paste; expected L -11.5 / R +11.5 deg):
  F3  L_knee -10.02  R_knee -48.37 deg  -> R off -59.9 deg = ~5 teeth   (L held)
  F4  L_knee -11.60  R_knee -61.36 deg  -> R off -72.9 deg = ~6 teeth   (L held)
User disassembly: the NEW R belt is broken the same way, at the 22T pulley next to the idler. The 22T PETG pulley
looks right. User suspects the PLA thigh plate has deformed (visible, amount unknown).
=> A fresh belt was destroyed in 2 trials on the R leg; the L belt survives the same commands. The R drive kills belts.
