"""
Live rear-approach threat scoring on a video, using PRETRAINED models.
====================================================================
Pipeline (five of the seven dashboard panels, all from pretrained weights + maths):

  Perception  : YOLO-pose detects people AND ByteTrack keeps stable IDs (one model call)
  Approach    : looming / time-to-contact from bounding-box growth        (geometry)
  Following   : distance trend + blind-spot bearing + persistence          (rule-based)
  Aggression  : wrist speed / arm-raise / jerk from the pose keypoints      (rule-based)
  Fusion      : weighted, calibrated-style blend -> one threat score + human-in-the-loop bands

Context-anomaly (Panel 6) and a trained fusion model (Panel 7) are left out on purpose:
they need training on your own data, which a quick demo does not have.

Alerts are SITUATIONAL (describe the event + a protective action), never a verdict
about a person -- matching the compliance design.

------------------------------------------------------------------
INSTALL (on your own machine, not a tiny sandbox -- pulls in PyTorch):
    pip install ultralytics opencv-python numpy

RUN:
    python live_rear_approach.py --source path/to/video.mp4 --out annotated.mp4
    python live_rear_approach.py --source 0                       # webcam, live window

NOTE ON LICENCE: ultralytics YOLO is AGPL-3.0 -- fine for this prototype/research,
but for a commercial product swap the model for RT-DETR (also in ultralytics) or buy
the Ultralytics Enterprise licence. See your feature/dataset/licence doc.
------------------------------------------------------------------
"""

import argparse
from collections import defaultdict, deque
import numpy as np
import cv2
from ultralytics import YOLO

# COCO-17 keypoint indices (what YOLO-pose returns)
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]

STATE_COLOR = {"CLEAR": (90, 200, 90), "MONITOR": (70, 200, 240),
               "VERIFY": (40, 160, 250), "ALERT": (60, 60, 235)}   # BGR
STATE_ORDER = {"CLEAR": 0, "MONITOR": 1, "VERIFY": 2, "ALERT": 3}
BANNER = {"CLEAR": "Clear", "MONITOR": "Monitoring behind you",
          "VERIFY": "Someone approaching from behind - check",
          "ALERT": "Rear alert. Stay aware."}


def norm(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0))


class Track:
    def __init__(self):
        self.h = deque(maxlen=30)        # box height history
        self.cx = deque(maxlen=30)       # box centre-x history
        self.dhat = deque(maxlen=30)     # distance proxy history
        self.closing = deque(maxlen=20)  # 1 if approaching this frame
        self.wr = deque(maxlen=8)        # wrist position history (for velocity/jerk)
        self.score = 0.0
        self.state = "CLEAR"
        self.alert_frames = 0


class Scorer:
    # weights are exposed so you can tune them later on real confirmed/dismissed logs
    W_PROX, W_PERSIST, W_CENTRAL, W_TTC, W_AGGR, W_DRIFT = 0.16, 0.26, 0.14, 0.20, 0.20, 0.22

    def __init__(self, frame_w, frame_h, fps):
        self.W, self.H, self.fps = frame_w, frame_h, fps
        self.tracks = defaultdict(Track)

    def aggression(self, t, kp, kp_conf, box_h):
        """Rough aggression proxy from pose: raised arm + fast, jerky wrist motion.
        Scale-invariant (normalised by box height). Returns 0..1."""
        if kp is None:
            return 0.0
        def ok(i):
            return kp_conf is None or kp_conf[i] > 0.3
        score = 0.0
        # arm raised above shoulder (image y grows downward -> wrist_y < shoulder_y)
        raised = 0.0
        for wr, sh in ((L_WR, L_SH), (R_WR, R_SH)):
            if ok(wr) and ok(sh) and kp[sh][1] > 0 and kp[wr][1] > 0:
                if kp[wr][1] < kp[sh][1]:
                    raised = 1.0
        # wrist speed + jerk from history (use the faster wrist)
        wrist = None
        for wr in (R_WR, L_WR):
            if ok(wr) and kp[wr][0] > 0:
                wrist = np.array(kp[wr], dtype=float); break
        speed = jerk = 0.0
        if wrist is not None:
            t.wr.append(wrist)
            if len(t.wr) >= 3:
                d = np.diff(np.array(t.wr), axis=0)
                v = np.linalg.norm(d, axis=1) / max(box_h, 1e-3)   # normalise by size
                speed = norm(v[-1] * self.fps, 0.4, 3.0)
                jerk = norm(np.std(v) * self.fps, 0.3, 2.5)
        return float(np.clip(0.45 * raised + 0.35 * speed + 0.30 * jerk, 0.0, 1.0))

    def update(self, tid, bbox, kp, kp_conf):
        t = self.tracks[tid]
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        cx = (x1 + x2) / 2.0
        dhat = 1000.0 / max(h, 1e-3)                 # distance proxy (bigger box = closer)
        if t.dhat:
            t.closing.append(1.0 if dhat < t.dhat[-1] else 0.0)
        t.h.append(h); t.cx.append(cx); t.dhat.append(dhat)

        proximity = norm(h / self.H, 0.15, 0.8)
        persistence = float(np.mean(t.closing)) if t.closing else 0.0
        centrality = 1.0 - norm(abs(cx - self.W / 2) / (self.W / 2), 0.1, 1.0)
        # time-to-contact from looming
        ttc_score = 0.0
        if len(t.h) >= 5:
            growth = (t.h[-1] - t.h[-5]) / 4.0
            if growth > 0.2:
                ttc = (t.h[-1] / growth) / self.fps
                ttc_score = 1.0 - norm(ttc, 0.8, 6.0)
        aggr = self.aggression(t, kp, kp_conf, h)
        # benign crossing: only credited when off to the side AND not sustaining approach
        drift = 0.0
        if len(t.cx) >= 5 and centrality < 0.55 and persistence < 0.45:
            drift = norm(abs(t.cx[-1] - t.cx[-5]) / 4.0 / self.W, 0.01, 0.06)

        raw = (self.W_PROX * proximity + self.W_PERSIST * persistence
               + self.W_CENTRAL * centrality + self.W_TTC * ttc_score
               + self.W_AGGR * aggr - self.W_DRIFT * drift)
        raw = float(np.clip(raw, 0.0, 1.0))
        t.score = 0.8 * t.score + 0.2 * raw          # temporal smoothing

        if t.score >= 0.72:
            t.alert_frames += 1
            t.state = "ALERT" if t.alert_frames >= int(0.5 * self.fps) else "VERIFY"
        elif t.score >= 0.52:
            t.alert_frames = 0; t.state = "VERIFY"
        elif t.score >= 0.34:
            t.alert_frames = 0; t.state = "MONITOR"
        else:
            t.alert_frames = 0; t.state = "CLEAR"
        return t.score, t.state


def draw_pose(img, kp, kp_conf, color):
    if kp is None:
        return
    for a, b in SKELETON:
        if kp_conf is not None and (kp_conf[a] < 0.3 or kp_conf[b] < 0.3):
            continue
        pa, pb = kp[a], kp[b]
        if pa[0] > 0 and pb[0] > 0:
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="video path or '0' for webcam")
    ap.add_argument("--out", default="annotated.mp4", help="output video (ignored for webcam)")
    ap.add_argument("--weights", default="yolov8n-pose.pt", help="pretrained pose weights")
    ap.add_argument("--show", action="store_true", help="show a live window")
    args = ap.parse_args()

    model = YOLO(args.weights)                       # downloads weights on first run
    source = int(args.source) if args.source.isdigit() else args.source
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
        ok, frame = cap.read()
        if not ok:
            break
        # ONE call: YOLO-pose detection + ByteTrack IDs + keypoints, persons only
        res = model.track(frame, persist=True, classes=[0],
                          tracker="bytetrack.yaml", verbose=False)[0]

        worst = ("CLEAR", 0.0)
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
                score, state = scorer.update(tid, bbox, kp, kc)

                col = STATE_COLOR[state]
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                draw_pose(frame, kp, kc, col)
                cv2.putText(frame, f"ID{tid} {score:.2f} {state}", (x1, max(y1 - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

                prev = scorer.tracks[tid].__dict__.get("_prev", "CLEAR")
                if state != prev:
                    log.append((frame_idx / fps, tid, prev, state))
                scorer.tracks[tid]._prev = state

                if STATE_ORDER[state] > STATE_ORDER[worst[0]]:
                    worst = (state, score)

        # bottom banner reflects the most urgent track
        bc = STATE_COLOR[worst[0]]
        cv2.rectangle(frame, (0, H - 40), (W, H), bc, -1)
        cv2.putText(frame, BANNER[worst[0]], (12, H - 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)
        if args.show or is_webcam:
            cv2.imshow("rear-approach", frame)
            if cv2.waitKey(1) & 0xFF == 27:      # Esc to quit
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
