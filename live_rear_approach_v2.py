"""
Live rear-approach threat scoring on a video, using PRETRAINED models.
====================================================================
Perception : YOLO-pose detects people AND ByteTrack keeps stable IDs (one model call)
Approach   : looming / time-to-contact + closing SPEED + ACCELERATION (speeding up)
Following  : distance trend + blind-spot bearing + persistence + dwell + follower memory
Aggression : wrist speed / arm-raise / jerk from pose keypoints
Motion     : per-person motion-energy line (last ~15 s), like the dashboard
Fusion     : weighted blend -> one threat score + CLEAR/MONITOR/VERIFY/ALERT bands

Special cue: a person who raises a hand to HEAD level is drawn in DARKER ORANGE
(box + skeleton + marker) so they stand out from everyone else, who keep their
normal state colour.

Alerts are SITUATIONAL (event + protective action), never a verdict about a person.

------------------------------------------------------------------
INSTALL:  pip install -r requirements.txt      (ultralytics pulls in PyTorch, ~2 GB)
RUN:      python live_rear_approach.py --source video.mp4 --out annotated.mp4
          python live_rear_approach.py --source 0                 # webcam window
CONFIG:   defaults are read from .env if python-dotenv is installed; CLI flags override.
LICENCE:  ultralytics YOLO is AGPL-3.0 (fine for prototyping); swap to RT-DETR to ship.
------------------------------------------------------------------
"""

import argparse
import os
from collections import deque
import numpy as np
import cv2
from ultralytics import YOLO

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# COCO-17 keypoints
NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 0, 5, 6, 7, 8, 9, 10
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]

STATE_COLOR = {"CLEAR": (90, 200, 90), "MONITOR": (70, 200, 240),
               "VERIFY": (40, 160, 250), "ALERT": (60, 60, 235)}   # BGR
STATE_ORDER = {"CLEAR": 0, "MONITOR": 1, "VERIFY": 2, "ALERT": 3}
BANNER = {"CLEAR": "Clear", "MONITOR": "Monitoring behind you",
          "VERIFY": "Someone approaching from behind - check",
          "ALERT": "Rear alert. Stay aware."}
DARK_ORANGE = (0, 90, 200)                                          # BGR deep orange

MONITOR_T = float(os.getenv("MONITOR_THRESHOLD", 0.34))
VERIFY_T = float(os.getenv("VERIFY_THRESHOLD", 0.52))
ALERT_T = float(os.getenv("ALERT_THRESHOLD", 0.72))


def norm(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0))


def ok_kp(kp_conf, i):
    return kp_conf is None or kp_conf[i] > 0.3


def hand_at_head(kp, kp_conf):
    """True if either wrist is raised to head (nose) level or above."""
    if kp is None:
        return False
    nose_y = kp[NOSE][1] if (ok_kp(kp_conf, NOSE) and kp[NOSE][1] > 0) else None
    if nose_y is None:
        return False
    for wr in (L_WR, R_WR):
        if ok_kp(kp_conf, wr) and kp[wr][1] > 0 and kp[wr][1] <= nose_y + 20:
            return True
    return False


class Track:
    def __init__(self, fps):
        self.h = deque(maxlen=30)
        self.cx = deque(maxlen=30)
        self.dhat = deque(maxlen=30)
        self.closing = deque(maxlen=20)
        self.speed = deque(maxlen=10)
        self.wr = deque(maxlen=8)
        self.energy = deque(maxlen=int(15 * fps))     # ~15 s motion-energy history
        self.kp_prev = None
        self.dwell = 0                                # frames spent in an elevated state
        self.score = 0.0
        self.state = "CLEAR"
        self.prev_state = "CLEAR"
        self.alert_frames = 0
        self.last_seen = 0


class Scorer:
    W_PROX, W_PERSIST, W_CENTRAL, W_TTC = 0.12, 0.20, 0.10, 0.16
    W_AGGR, W_SPEED, W_ACCEL, W_DWELL, W_DRIFT = 0.16, 0.12, 0.18, 0.08, 0.20

    def __init__(self, frame_w, frame_h, fps):
        self.W, self.H, self.fps = frame_w, frame_h, fps
        self.tracks = {}

    def get(self, tid):
        if tid not in self.tracks:
            self.tracks[tid] = Track(self.fps)
        return self.tracks[tid]

    def aggression(self, t, kp, kc, box_h):
        if kp is None:
            return 0.0
        raised = 0.0
        for wr, sh in ((L_WR, L_SH), (R_WR, R_SH)):
            if ok_kp(kc, wr) and ok_kp(kc, sh) and kp[sh][1] > 0 and kp[wr][1] > 0:
                if kp[wr][1] < kp[sh][1]:
                    raised = 1.0
        wrist = None
        for wr in (R_WR, L_WR):
            if ok_kp(kc, wr) and kp[wr][0] > 0:
                wrist = np.array(kp[wr], float); break
        speed = jerk = 0.0
        if wrist is not None:
            t.wr.append(wrist)
            if len(t.wr) >= 3:
                d = np.diff(np.array(t.wr), axis=0)
                v = np.linalg.norm(d, axis=1) / max(box_h, 1e-3)
                speed = norm(v[-1] * self.fps, 0.4, 3.0)
                jerk = norm(np.std(v) * self.fps, 0.3, 2.5)
        return float(np.clip(0.45 * raised + 0.35 * speed + 0.30 * jerk, 0.0, 1.0))

    def motion_energy(self, t, kp, box_h):
        if kp is not None and t.kp_prev is not None:
            valid = (kp[:, 0] > 0) & (t.kp_prev[:, 0] > 0)
            if valid.any():
                disp = np.linalg.norm(kp[valid] - t.kp_prev[valid], axis=1)
                t.energy.append(float(np.mean(disp)) / max(box_h, 1e-3))
            else:
                t.energy.append(0.0)
        t.kp_prev = kp.copy() if kp is not None else None

    def update(self, tid, bbox, kp, kc, frame_idx):
        t = self.get(tid)
        t.last_seen = frame_idx
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        cx = (x1 + x2) / 2.0
        dhat = 1000.0 / max(h, 1e-3)
        if t.dhat:
            t.closing.append(1.0 if dhat < t.dhat[-1] else 0.0)
        t.h.append(h); t.cx.append(cx); t.dhat.append(dhat)
        self.motion_energy(t, kp, h)

        proximity = norm(h / self.H, 0.15, 0.8)
        persistence = float(np.mean(t.closing)) if t.closing else 0.0
        centrality = 1.0 - norm(abs(cx - self.W / 2) / (self.W / 2), 0.1, 1.0)

        ttc_score = 0.0
        if len(t.h) >= 5:
            growth = (t.h[-1] - t.h[-5]) / 4.0
            if growth > 0.2:
                ttc = (t.h[-1] / growth) / self.fps
                ttc_score = 1.0 - norm(ttc, 0.8, 6.0)

        # closing SPEED and ACCELERATION (speeding up = lunge)
        closing_speed = 0.0
        if len(t.dhat) >= 3:
            closing_speed = (t.dhat[-3] - t.dhat[-1]) / 2.0
        t.speed.append(closing_speed)
        accel = 0.0
        if len(t.speed) >= 5:
            accel = np.mean(list(t.speed)[-2:]) - np.mean(list(t.speed)[-5:-3])
        speed_score = norm(closing_speed * self.fps, 0.5, 8.0)
        accel_score = norm(accel * self.fps, 0.3, 5.0)

        aggr = self.aggression(t, kp, kc, h)

        drift = 0.0
        if len(t.cx) >= 5 and centrality < 0.55 and persistence < 0.45:
            drift = norm(abs(t.cx[-1] - t.cx[-5]) / 4.0 / self.W, 0.01, 0.06)

        dwell_score = norm(t.dwell / self.fps, 2.0, 8.0)

        raw = (self.W_PROX * proximity + self.W_PERSIST * persistence
               + self.W_CENTRAL * centrality + self.W_TTC * ttc_score
               + self.W_AGGR * aggr + self.W_SPEED * speed_score
               + self.W_ACCEL * accel_score + self.W_DWELL * dwell_score
               - self.W_DRIFT * drift)
        raw = float(np.clip(raw, 0.0, 1.0))
        t.score = 0.8 * t.score + 0.2 * raw

        t.prev_state = t.state
        if t.score >= ALERT_T:
            t.alert_frames += 1
            t.state = "ALERT" if t.alert_frames >= int(0.5 * self.fps) else "VERIFY"
        elif t.score >= VERIFY_T:
            t.alert_frames = 0; t.state = "VERIFY"
        elif t.score >= MONITOR_T:
            t.alert_frames = 0; t.state = "MONITOR"
        else:
            t.alert_frames = 0; t.state = "CLEAR"
        t.dwell = t.dwell + 1 if t.state != "CLEAR" else 0

        return dict(score=t.score, state=t.state, prev=t.prev_state,
                    closing=closing_speed * self.fps, accel=accel * self.fps,
                    dwell=t.dwell / self.fps,
                    sub={                                  # the reasoning, exposed
                        "Approach":    max(proximity, ttc_score),
                        "Following":   max(persistence, centrality),
                        "Aggression":  aggr,
                        "Speeding up": accel_score,
                    })


def draw_pose(img, kp, kc, color):
    if kp is None:
        return
    for a, b in SKELETON:
        if not (ok_kp(kc, a) and ok_kp(kc, b)):
            continue
        pa, pb = kp[a], kp[b]
        if pa[0] > 0 and pb[0] > 0:
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 2)


def draw_energy_graph(img, series, x, y, w, h, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), 1)
    cv2.putText(img, "motion energy (15s)", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    if len(series) < 2:
        return
    vals = np.array(series, float)
    vals = vals / (vals.max() + 1e-6)
    n = len(vals)
    pts = [(x + int(i / n * w), y + h - int(v * h)) for i, v in enumerate(vals)]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, color, 1, cv2.LINE_AA)


def draw_score_bar(img, x, y, w, score, color, h=6):
    """Thin horizontal bar showing a person's 0..1 threat score."""
    w = max(w, 1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)                 # track
    cv2.rectangle(img, (x, y), (x + int(w * float(np.clip(score, 0, 1))), y + h), color, -1)   # fill


def draw_subscore_panel(img, sub, x, y, color, title="why (sub-scores)"):
    """Top-left panel: one bar per sub-score, explaining the blended score."""
    w, rowh = 200, 26
    cv2.rectangle(img, (x - 10, y - 26), (x + w + 10, y + len(sub) * rowh + 6), (25, 25, 25), -1)
    cv2.putText(img, title, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    for i, (name, val) in enumerate(sub.items()):
        yy = y + i * rowh
        cv2.putText(img, name, (x, yy + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
        bx = x + 95
        cv2.rectangle(img, (bx, yy + 2), (bx + 95, yy + 13), (60, 60, 60), -1)          # track
        cv2.rectangle(img, (bx, yy + 2), (bx + int(95 * val), yy + 13), color, -1)      # fill
        cv2.putText(img, f"{val:.2f}", (bx + 100, yy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 210, 210), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.getenv("SOURCE", "0"))
    ap.add_argument("--out", default=os.getenv("OUT", "annotated.mp4"))
    ap.add_argument("--weights", default=os.getenv("WEIGHTS", "yolov8n-pose.pt"))
    ap.add_argument("--show", action="store_true",
                    default=os.getenv("SHOW", "false").lower() == "true")
    args = ap.parse_args()

    model = YOLO(args.weights)
    source = int(args.source) if str(args.source).isdigit() else args.source
    is_webcam = isinstance(source, int)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    writer = None
    if not is_webcam:
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    scorer = Scorer(W, H, fps)
    log = []
    frame_idx = 0

    while True:
        okf, frame = cap.read()
        if not okf:
            break
        res = model.track(frame, persist=True, classes=[0],
                          tracker="bytetrack.yaml", verbose=False)[0]

        worst = ("CLEAR", 0.0)
        top_energy = None
        worst_sub, worst_tid = None, None
        boxes = res.boxes
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.int().cpu().tolist()
            kpts = res.keypoints.xy.cpu().numpy() if res.keypoints is not None else None
            kconf = (res.keypoints.conf.cpu().numpy()
                     if (res.keypoints is not None and res.keypoints.conf is not None) else None)

            for i, tid in enumerate(ids):
                bbox = xyxy[i]
                kp = kpts[i] if kpts is not None else None
                kc = kconf[i] if kconf is not None else None
                r = scorer.update(tid, bbox, kp, kc, frame_idx)

                # colour: state colour, OR darker orange if THIS person raises a hand to head level
                col = STATE_COLOR[r["state"]]
                raised = hand_at_head(kp, kc)
                if raised:
                    col = DARK_ORANGE

                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if raised else 2)
                draw_pose(frame, kp, kc, col)
                cv2.putText(frame, f"ID{tid} {r['score']:.2f} {r['state']}",
                            (x1, max(y1 - 26, 26)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
                cv2.putText(frame, f"close {r['closing']:+.1f}  accel {r['accel']:+.1f}"
                            + ("  HAND UP" if raised else ""),
                            (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
                # per-person score bar, just above the ID label
                draw_score_bar(frame, x1, max(y1 - 46, 2), x2 - x1, r["score"], col)

                if r["state"] != r["prev"]:
                    log.append((frame_idx / fps, tid, r["prev"], r["state"]))
                if STATE_ORDER[r["state"]] > STATE_ORDER[worst[0]]:
                    worst = (r["state"], r["score"])
                    top_energy = scorer.get(tid).energy
                    worst_sub, worst_tid = r["sub"], tid

        # motion-energy graph for the most urgent track (bottom-right, like the dashboard)
        if top_energy is not None:
            draw_energy_graph(frame, top_energy, W - 230, H - 150, 210, 90, STATE_COLOR[worst[0]])

        # "why" panel for the most urgent track only (top-left)
        if worst_sub is not None:
            draw_subscore_panel(frame, worst_sub, 20, 90, STATE_COLOR[worst[0]],
                                title=f"why ID{worst_tid} (sub-scores)")

        bc = STATE_COLOR[worst[0]]
        cv2.rectangle(frame, (0, H - 40), (W, H), bc, -1)
        cv2.putText(frame, BANNER[worst[0]], (12, H - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(frame, "on-device . no footage stored", (W - 250, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)
        if args.show or is_webcam:
            cv2.imshow("rear-approach", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("\nAlert transitions (time_s, track, from -> to):")
    for ts, tid, a, b in log:
        print(f"  {ts:6.1f}s  ID{tid}  {a} -> {b}")
    if writer is not None:
        print(f"\nWrote annotated video: {args.out}")


if __name__ == "__main__":
    main()
