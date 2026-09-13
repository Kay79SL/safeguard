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
DARK_ORANGE = (0, 90, 200)   # BGR - a deep orange, used when a hand is raised to the head


def norm(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0))


class Track:
    def __init__(self, fps=25):
        self.h = deque(maxlen=30)        # box height history
        self.cx = deque(maxlen=30)       # box centre-x history
        self.dhat = deque(maxlen=30)     # distance proxy history
        self.closing = deque(maxlen=20)  # 1 if approaching this frame
        self.wr = deque(maxlen=8)        # wrist position history (for velocity/jerk)
        self.speed = deque(maxlen=10)    # closing-speed history (for acceleration)
        self.closing_speed = 0.0         # last closing speed (distance units / frame)
        self.accel = 0.0                 # last closing acceleration (per frame)
        self.energy = deque(maxlen=int(15 * fps))   # motion energy, ~15 seconds
        self.kp_prev = None              # keypoints from the previous frame
        self.score = 0.0
        self.state = "CLEAR"
        self.alert_frames = 0


class Scorer:
    # weights are exposed so you can tune them later on real confirmed/dismissed logs
    W_PROX, W_PERSIST, W_CENTRAL, W_TTC, W_AGGR, W_DRIFT = 0.16, 0.26, 0.14, 0.20, 0.20, 0.22

    def __init__(self, frame_w, frame_h, fps):
        self.W, self.H, self.fps = frame_w, frame_h, fps
        self.tracks = defaultdict(lambda: Track(fps))

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

        # MOTION ENERGY: average joint movement since last frame, normalised by body size
        if kp is not None:
            if t.kp_prev is not None:
                seen = (kp[:, 0] > 0) & (t.kp_prev[:, 0] > 0)      # joint detected in both frames
                if kp_conf is not None:
                    seen &= kp_conf > 0.3
                if seen.any():
                    disp = np.linalg.norm(kp[seen] - t.kp_prev[seen], axis=1)   # movement per joint
                    t.energy.append(float(np.mean(disp)) / max(h, 1e-3))
            t.kp_prev = kp.copy()

        # closing SPEED: how fast the distance proxy is shrinking (positive = approaching)
        closing_speed = 0.0
        if len(t.dhat) >= 3:
            closing_speed = (t.dhat[-3] - t.dhat[-1]) / 2.0   # drop in distance over 2 frames
        t.speed.append(closing_speed)

        # closing ACCELERATION: is that speed getting bigger? (positive = speeding up / lunge)
        accel = 0.0
        if len(t.speed) >= 5:
            recent = np.mean(list(t.speed)[-2:])     # speed now
            earlier = np.mean(list(t.speed)[-5:-3])  # speed a moment ago
            accel = recent - earlier
        t.closing_speed, t.accel = closing_speed, accel

        speed_score = norm(closing_speed * self.fps, 0.5, 8.0)   # steady approach speed
        accel_score = norm(accel * self.fps, 0.3, 5.0)           # SPEEDING UP -> high
        # benign crossing: only credited when off to the side AND not sustaining approach
        drift = 0.0
        if len(t.cx) >= 5 and centrality < 0.55 and persistence < 0.45:
            drift = norm(abs(t.cx[-1] - t.cx[-5]) / 4.0 / self.W, 0.01, 0.06)

        raw = (self.W_PROX * proximity + self.W_PERSIST * persistence
               + self.W_CENTRAL * centrality + self.W_TTC * ttc_score
               + self.W_AGGR * aggr
               + 0.14 * speed_score          # NEW: fast approach
               + 0.20 * accel_score          # NEW: speeding up (the lunge)
               - self.W_DRIFT * drift)
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


def draw_pose(img, kp, kp_conf, color, thickness=2):
    if kp is None:
        return
    for a, b in SKELETON:
        if kp_conf is not None and (kp_conf[a] < 0.3 or kp_conf[b] < 0.3):
            continue
        pa, pb = kp[a], kp[b]
        if pa[0] > 0 and pb[0] > 0:
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, thickness)


def hand_at_head(kp, kp_conf, margin=20):
    """True if either wrist is raised to head (nose) level or above.
    margin is in pixels below the nose that still counts as head level."""
    if kp is None:
        return False
    def ok(i):
        return kp_conf is None or kp_conf[i] > 0.3
    nose_y = kp[0][1] if ok(0) and kp[0][1] > 0 else None
    if nose_y is None:
        return False
    for wr in (L_WR, R_WR):          # 9, 10
        if ok(wr) and kp[wr][1] > 0 and kp[wr][1] <= nose_y + margin:   # y grows downward
            return True
    return False


def draw_energy_graph(img, series, x, y, w, h, color, label, s=1.0):
    """Mini line graph of a track's motion energy, scaled 0..1 to its own peak."""
    if len(series) < 2:
        return
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)            # translucent dark panel
    cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), max(1, int(s)))
    vals = np.array(series)
    vals = vals / (vals.max() + 1e-6)                          # scale 0..1
    pts = [(x + int(i / (len(vals) - 1) * w), y + h - int(v * h)) for i, v in enumerate(vals)]
    cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, color, max(1, int(1.5 * s)), cv2.LINE_AA)
    cv2.putText(img, label, (x, y - int(8 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, color, max(1, int(s)), cv2.LINE_AA)


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
    s = H / 720                          # 1.0 at 720p, 3.0 at 4K
    th = max(2, int(2 * s))              # line thickness
    th_small = max(1, int(1 * s))        # thin text thickness
    bar = int(60 * s)                    # banner height

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
        focus_tid, focus_score = None, -1.0   # highest-scoring track -> shown in the energy graph
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
                raised = hand_at_head(kp, kc, margin=int(20 * s))
                if raised:
                    col = DARK_ORANGE
                x1, y1, x2, y2 = bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, th)
                draw_pose(frame, kp, kc, col, th)
                trk = scorer.tracks[tid]
                # two label lines above the box, or just inside it when the box touches the top
                ly = y1 - int(26 * s) if y1 > int(60 * s) else y1 + int(26 * s)
                cv2.putText(frame, f"ID{tid} {score:.2f} {state}", (x1 + th, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, col, th, cv2.LINE_AA)
                arrow = "UP" if trk.accel > 0 else "  "
                cv2.putText(frame, f"closing {trk.closing_speed * fps:+.1f}  accel {trk.accel * fps:+.1f} {arrow}",
                            (x1 + th, ly + int(18 * s)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, col, th_small, cv2.LINE_AA)
                if raised:   # third line: above the ID line, or below the closing line when inside the box
                    hy = ly - int(22 * s) if y1 > int(60 * s) else ly + int(38 * s)
                    cv2.putText(frame, "hand raised", (x1 + th, hy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, DARK_ORANGE, th_small, cv2.LINE_AA)

                prev = scorer.tracks[tid].__dict__.get("_prev", "CLEAR")
                if state != prev:
                    log.append((frame_idx / fps, tid, prev, state))
                scorer.tracks[tid]._prev = state

                if score > focus_score:
                    focus_tid, focus_score = tid, score
                if STATE_ORDER[state] > STATE_ORDER[worst[0]]:
                    worst = (state, score)

        # motion-energy mini graph (bottom-right, above the banner) for the highest-scoring track
        if focus_tid is not None:
            gw, gh, gm = int(260 * s), int(70 * s), int(16 * s)
            draw_energy_graph(frame, scorer.tracks[focus_tid].energy,
                              W - gw - gm, H - bar - gh - gm, gw, gh, (240, 240, 240),
                              f"ID{focus_tid} motion energy (15s)", s)

        # bottom banner reflects the most urgent track
        bc = STATE_COLOR[worst[0]]
        cv2.rectangle(frame, (0, H - bar), (W, H), bc, -1)
        cv2.putText(frame, BANNER[worst[0]], (int(18 * s), H - int(20 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.05 * s, (20, 20, 20), int(th * 1.5), cv2.LINE_AA)

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
