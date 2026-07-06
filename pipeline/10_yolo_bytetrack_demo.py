"""
10_yolo_bytetrack_demo.py — Phase 4 (Days 13-15): end-to-end qualitative demo.

Pipeline:
    raw set03 video
      -> YOLO26-M               (detect persons)
      -> ByteTrack              (stable per-pedestrian track IDs)
      -> per-track rolling 16-frame buffer of [x1, y1, x2, y2, ego_speed]
      -> normalize (train stats) -> BiLSTM -> sigmoid -> P(cross)
      -> color-coded overlay    -> annotated mp4

Design (locked, see THESIS_PLAN / Phase 4 plan):
  - Model    : 5D baseline (paper_and_artifacts/runs/bilstm_baseline) -- the locked main model.
  - Ego-speed: per-frame OBD speed, sourced from pie_annotations.pkl by default
               (set03 rows already carry vehicle_speed); --ego-source obd reads
               the *_obd.xml directly (full frame coverage) via the existing
               parse_obd() from 01_parse_annotations.py.
  - YOLO     : YOLO26-M is REQUIRED. No fallback -- if it won't load we raise so
               the environment gets upgraded rather than silently swapping model.

Frame handling:
  Frames are read with OpenCV so we can seek (--start-frame) and limit
  (--max-frames) to a short segment, and so absolute PIE frame numbers stay
  aligned with the ego-speed lookup. Each frame is fed to the tracker with
  persist=True (the streaming pattern), which keeps ByteTrack state across calls.

Stages (run independently for the day-by-day write-up):
  --stage detect : Day 13 -- YOLO26-M detections, save sample frames.
  --stage track  : Day 14 -- ByteTrack, save sample frames with IDs + ID stats.
  --stage demo   : Day 15 -- full pipeline, annotated mp4 (+ optional CSV dump).

Device is auto-selected: CUDA -> MPS (Apple Silicon) -> CPU. Override with --device.
"""

import argparse
import csv
from collections import deque
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
import torch

# Modules can't start with a digit -> importlib.
BiLSTM = import_module("03_bilstm_model").BiLSTMIntentPredictor

OBS_LEN = 16            # must match training (02_build_sequences.py / 04_train_bilstm.py)
THRESHOLD = 0.5         # must match evaluate() in 04_train_bilstm.py
PERSON_CLASS = 0        # COCO 'person'
EXPECTED_W, EXPECTED_H = 1920, 1080  # PIE native resolution (training coord scale)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ultralytics_device(device: torch.device):
    """Map a torch.device to the value Ultralytics expects."""
    return {"cuda": 0, "mps": "mps", "cpu": "cpu"}[device.type]


# ---------------------------------------------------------------------------
# Loading: model + normalization stats
# ---------------------------------------------------------------------------

def load_bilstm(weights_dir: Path, device: torch.device):
    """Load the 5D baseline checkpoint + train-only normalization stats."""
    ckpt_path = weights_dir / "best.pt"
    mean = np.load(weights_dir / "norm_mean.npy").astype(np.float32)  # (5,)
    std = np.load(weights_dir / "norm_std.npy").astype(np.float32)    # (5,)
    assert mean.shape == (5,) and std.shape == (5,), \
        f"expected (5,) norm stats, got mean{mean.shape} std{std.shape}"

    model = BiLSTM().to(device)
    # weights_only=False: checkpoint stores val_metrics (numpy scalars) alongside
    # the state_dict, which torch>=2.6 refuses under the default weights_only=True.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])           # checkpoint dict from 04_train_bilstm
    model.eval()
    print(f"[model] loaded {ckpt_path} (best epoch {ckpt.get('epoch', '?')}) on {device}")
    return model, mean, std


def load_yolo():
    """Load YOLO26-M. REQUIRED -- raise (no fallback) if unavailable."""
    import ultralytics
    from ultralytics import YOLO
    print(f"[yolo] ultralytics {ultralytics.__version__}")
    try:
        model = YOLO("yolo26m.pt")   # auto-downloads weights on first use
    except Exception as e:
        raise RuntimeError(
            "YOLO26-M ('yolo26m.pt') could not be loaded and there is no "
            "fallback by design. Upgrade ultralytics: `pip install -U ultralytics`.\n"
            f"Underlying error: {e}"
        ) from e
    print("[yolo] loaded yolo26m.pt")
    return model


# ---------------------------------------------------------------------------
# Ego-speed: frame -> OBD speed
# ---------------------------------------------------------------------------

def build_speed_map(args) -> dict[int, float]:
    """Return {frame_index: ego_speed} for the target set/video.

    Default source is pie_annotations.pkl (vehicle_speed is constant across
    pedestrians within a frame -> take the first). --ego-source obd reads the
    *_obd.xml directly, which covers every frame (not just annotated ones).
    """
    if args.ego_source == "obd":
        parse_obd = import_module("01_parse_annotations").parse_obd
        smap = parse_obd(args.obd_xml)
        print(f"[speed] {len(smap)} frames from OBD xml {args.obd_xml}")
        return smap

    import pandas as pd
    df = pd.read_pickle(args.annotations_pkl)
    sub = df[(df["set_id"] == args.set_id) & (df["video_id"] == args.video_id)]
    if sub.empty:
        raise ValueError(
            f"No rows in {args.annotations_pkl} for {args.set_id}/{args.video_id}. "
            f"Check --set-id/--video-id, or use --ego-source obd."
        )
    smap = (sub.groupby("frame")["vehicle_speed"].first().to_dict())
    smap = {int(k): float(v) for k, v in smap.items()}
    print(f"[speed] {len(smap)} annotated frames from {args.annotations_pkl} "
          f"for {args.set_id}/{args.video_id} "
          f"(range {min(smap)}..{max(smap)})")
    return smap


def speed_for(frame_idx: int, smap: dict[int, float], last: float) -> float:
    """Look up ego speed; carry the last known value across gaps."""
    v = smap.get(frame_idx)
    if v is None or not np.isfinite(v):
        return last
    return v


# ---------------------------------------------------------------------------
# Frame reader (seek + limit) and video meta
# ---------------------------------------------------------------------------

def video_meta(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, w, h, n


def frame_reader(path: str, start: int, maxn):
    """Yield (absolute_frame_index, BGR frame) from `start`, up to `maxn` frames."""
    cap = cv2.VideoCapture(path)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = 0
    while maxn is None or i < maxn:
        ok, frame = cap.read()
        if not ok:
            break
        yield start + i, frame
        i += 1
    cap.release()


# ---------------------------------------------------------------------------
# Inference on one 16-frame window
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_window(model, window: np.ndarray, mean, std, device) -> float:
    """window: (OBS_LEN, 5) raw [x1,y1,x2,y2,speed] -> P(cross)."""
    x = (window.astype(np.float32) - mean) / std          # match normalize()
    xt = torch.from_numpy(x).unsqueeze(0).to(device)       # (1, T, 5)
    logit = model(xt).squeeze()                            # scalar
    return float(torch.sigmoid(logit).item())


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_box(img, xyxy, label, prob=None):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    if prob is None:
        color = (200, 200, 200)        # grey: warming up (<16 frames)
    else:
        # green (safe) -> red (crossing); BGR
        color = (0, int(255 * (1 - prob)), int(255 * prob))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def check_resolution(w, h):
    if (w, h) != (EXPECTED_W, EXPECTED_H):
        print(f"[warn] video is {w}x{h}, training coords are {EXPECTED_W}x{EXPECTED_H}. "
              f"Bbox scale may not match the model's training distribution.")


# ---------------------------------------------------------------------------
# Stage 1: detect
# ---------------------------------------------------------------------------

def stage_detect(args, device):
    yolo = load_yolo()
    dev = ultralytics_device(device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fps, w, h, n = video_meta(args.video)
    check_resolution(w, h)
    print(f"[detect] {args.video} {w}x{h} @ {fps:.1f}fps, {n} frames total")

    saved, seen = 0, 0
    for fidx, frame in frame_reader(args.video, args.start_frame, args.max_frames):
        r = yolo.predict(frame, classes=[PERSON_CLASS], conf=args.conf,
                         device=dev, verbose=False)[0]
        seen += 1
        if seen % args.sample_every == 1 and saved < args.n_samples:
            img = frame.copy()
            for b in r.boxes:
                draw_box(img, b.xyxy[0].tolist(), f"p {float(b.conf[0]):.2f}")
            cv2.imwrite(str(out_dir / f"detect_{args.video_id}_f{fidx:05d}.png"), img)
            saved += 1
    print(f"[detect] processed {seen} frames, saved {saved} samples -> {out_dir}")


# ---------------------------------------------------------------------------
# Stage 2: track
# ---------------------------------------------------------------------------

def stage_track(args, device):
    yolo = load_yolo()
    dev = ultralytics_device(device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fps, w, h, n = video_meta(args.video)
    check_resolution(w, h)

    track_frames: dict[int, int] = {}
    saved, seen = 0, 0
    for fidx, frame in frame_reader(args.video, args.start_frame, args.max_frames):
        r = yolo.track(frame, classes=[PERSON_CLASS], conf=args.conf,
                       tracker="bytetrack.yaml", persist=True,
                       device=dev, verbose=False)[0]
        seen += 1
        ids = r.boxes.id
        if ids is not None:
            for tid in ids.int().tolist():
                track_frames[tid] = track_frames.get(tid, 0) + 1
        if seen % args.sample_every == 1 and saved < args.n_samples:
            img = frame.copy()
            if ids is not None:
                for b, tid in zip(r.boxes, ids.int().tolist()):
                    draw_box(img, b.xyxy[0].tolist(), f"ID{tid}")
            cv2.imwrite(str(out_dir / f"track_{args.video_id}_f{fidx:05d}.png"), img)
            saved += 1
    persistent = sum(1 for c in track_frames.values() if c >= OBS_LEN)
    print(f"[track] {seen} frames | {len(track_frames)} unique IDs | "
          f"{persistent} IDs persisted >= {OBS_LEN} frames")
    print(f"[track] saved {saved} samples -> {out_dir}")


# ---------------------------------------------------------------------------
# Stage 3: full demo
# ---------------------------------------------------------------------------

def stage_demo(args, device):
    yolo = load_yolo()
    dev = ultralytics_device(device)
    model, mean, std = load_bilstm(Path(args.weights_dir), device)
    smap = build_speed_map(args)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    fps, w, h, n = video_meta(args.video)
    check_resolution(w, h)
    print(f"[demo] {args.video} {w}x{h} @ {fps:.1f}fps | segment "
          f"start={args.start_frame} max={args.max_frames}")

    out_mp4 = out_dir / f"demo_{args.video_id}.mp4"
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    buffers: dict[int, deque] = {}
    last_frame: dict[int, int] = {}   # reset window on temporal gaps (keep consecutive)
    last_prob: dict[int, float] = {}  # cache prediction between updates

    csv_rows = []
    last_speed = 0.0
    sample_saved, seen, n_pred = 0, 0, 0

    for fidx, frame in frame_reader(args.video, args.start_frame, args.max_frames):
        seen += 1
        ego = speed_for(fidx, smap, last_speed)
        last_speed = ego
        img = frame.copy()

        r = yolo.track(frame, classes=[PERSON_CLASS], conf=args.conf,
                       tracker="bytetrack.yaml", persist=True,
                       device=dev, verbose=False)[0]
        ids = r.boxes.id

        if ids is not None:
            for b, tid in zip(r.boxes, ids.int().tolist()):
                xyxy = b.xyxy[0].tolist()
                feat = [xyxy[0], xyxy[1], xyxy[2], xyxy[3], ego]

                if tid in last_frame and fidx - last_frame[tid] > 1:
                    buffers.pop(tid, None)          # non-consecutive -> restart window
                last_frame[tid] = fidx

                buf = buffers.setdefault(tid, deque(maxlen=OBS_LEN))
                buf.append(feat)

                prob = None
                if len(buf) == OBS_LEN:
                    prob = predict_window(model, np.asarray(buf), mean, std, device)
                    last_prob[tid] = prob
                    n_pred += 1
                else:
                    prob = last_prob.get(tid)

                label = f"ID{tid} {prob:.2f}" if prob is not None else f"ID{tid} ..."
                draw_box(img, xyxy, label, prob)

                if args.dump_csv and prob is not None:
                    csv_rows.append([fidx, tid, *[f"{v:.1f}" for v in xyxy],
                                     f"{ego:.3f}", f"{prob:.4f}", int(prob >= THRESHOLD)])

        cv2.putText(img, f"frame {fidx} | ego {ego:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(img)

        if seen % args.sample_every == 1 and sample_saved < args.n_samples:
            cv2.imwrite(str(out_dir / f"demo_{args.video_id}_f{fidx:05d}.png"), img)
            sample_saved += 1

    writer.release()
    print(f"[demo] processed {seen} frames, {n_pred} window predictions made")
    print(f"[demo] wrote {out_mp4} ({sample_saved} sample frames)")

    if args.dump_csv:
        csv_path = out_dir / f"demo_{args.video_id}_predictions.csv"
        with open(csv_path, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["frame", "track_id", "x1", "y1", "x2", "y2",
                           "ego_speed", "prob_cross", "pred"])
            wcsv.writerows(csv_rows)
        print(f"[demo] wrote {len(csv_rows)} prediction rows -> {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Phase 4: YOLO26 + ByteTrack + BiLSTM demo")
    ap.add_argument("--stage", choices=["detect", "track", "demo"], default="demo")
    ap.add_argument("--video", required=True, help="path to set03 .mp4 clip")
    ap.add_argument("--set-id", default="set03")
    ap.add_argument("--video-id", default="video_0016",
                    help="PIE video id used for ego-speed lookup + output names")
    ap.add_argument("--weights-dir", default="paper_and_artifacts/runs/bilstm_baseline",
                    help="dir with best.pt + norm_mean.npy + norm_std.npy")
    # segment selection
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None)
    # ego-speed source
    ap.add_argument("--ego-source", choices=["pkl", "obd"], default="pkl")
    ap.add_argument("--annotations-pkl", default="pie_annotations.pkl")
    ap.add_argument("--obd-xml", default=None,
                    help="path to <video>_obd.xml (required if --ego-source obd)")
    # yolo / output / device
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--out-dir", default="pipeline/demo_out")
    ap.add_argument("--sample-every", type=int, default=60, help="save a sample frame every N")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--dump-csv", action="store_true",
                    help="write per-frame per-track predictions to CSV (demo stage)")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "mps", "cpu"])
    args = ap.parse_args()

    if args.ego_source == "obd" and not args.obd_xml:
        ap.error("--ego-source obd requires --obd-xml")

    device = pick_device(args.device)
    print(f"[init] device: {device} | stage: {args.stage}")

    if args.stage == "detect":
        stage_detect(args, device)
    elif args.stage == "track":
        stage_track(args, device)
    else:
        stage_demo(args, device)


if __name__ == "__main__":
    main()
