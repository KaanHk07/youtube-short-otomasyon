"""Yatay videodan dikey (9:16) kadraj: yuz / hareket takipli, titremesiz sanal kamera.

1. Ara videodan ~6 fps ornek alinir.
2. Her ornekte OpenCV YuNet ile yuzler bulunur (model yoksa Haar cascade).
   - Birden cok yuz kadraja sigiyorsa ikisini birden ortalar,
   - sigmiyorsa bir onceki hedefe yakin / en buyuk yuzu takip eder (konusan kisi).
   - Yuz yoksa hareket agirlik merkezi, o da yoksa bir onceki konum / merkez.
3. Sahne kesimleri (histogram farki) yolu boler; her parca icinde olu bolge +
   Gauss yumusatma ile kamera sakin hareket eder, kesimde aninda gecer.
Goruntu asla esnetilmez: yalnizca kirpilip orantili olceklenir.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"
DETECT_WIDTH = 640
SCENE_CUT = 0.45       # histogram Bhattacharyya mesafesi esigi
DEAD_ZONE = 0.07       # kadraj genisliginin orani: bu kadar kaymayi takip etme
SMOOTH_SEC = 0.6       # Gauss yumusatma sigmasi (saniye)


@dataclass
class CropPlan:
    mode: str                  # "scale" | "center" | "smart"
    crop_w: int
    crop_h: int
    xs: np.ndarray | None      # kare basina kirpma sol-ust x (smart)
    y: int = 0
    faces_found: int = 0


def crop_size(src_w: int, src_h: int, out_w: int, out_h: int) -> tuple[int, int]:
    """Kaynaktan kesilecek en buyuk hedef-oranli pencere (cift sayilar)."""
    target = out_w / out_h
    if src_w / src_h > target:
        cw, ch = int(round(src_h * target)), src_h
    else:
        cw, ch = src_w, int(round(src_w / target))
    return min(src_w, cw - cw % 2), min(src_h, ch - ch % 2)


def needs_horizontal_crop(src_w: int, src_h: int, out_w: int, out_h: int) -> bool:
    return (src_w / src_h) > (out_w / out_h) * 1.03


class _FaceDetector:
    def __init__(self) -> None:
        import cv2
        self.cv2 = cv2
        self.yunet = None
        self.haar = None
        if MODEL_PATH.exists():
            try:
                self.yunet = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (320, 320), 0.6, 0.3, 50)
            except Exception as exc:
                log.warning("YuNet yuklenemedi, Haar kullanilacak: %s", exc)
        if self.yunet is None:
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, img: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """(x, y, w, h, skor) listesi; img koordinatlarinda."""
        h, w = img.shape[:2]
        if self.yunet is not None:
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(img)
            if faces is None:
                return []
            return [(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[14])) for f in faces]
        gray = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
        found = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(max(24, w // 40),) * 2)
        return [(float(x), float(y), float(fw), float(fh), 0.7) for x, y, fw, fh in found]


def _choose_target(faces, prev_x: float | None, crop_w_s: float) -> float:
    """Yuz listesinden kadraj merkezi (kucultulmus koordinatta)."""
    big = max(f[2] * f[3] for f in faces)
    faces = [f for f in faces if f[2] * f[3] >= 0.25 * big]  # arka plandaki kucuk yuzleri at
    left = min(f[0] for f in faces)
    right = max(f[0] + f[2] for f in faces)
    if right - left <= crop_w_s * 0.9:
        return (left + right) / 2           # hepsi sigiyor: grubu ortala
    best = max(faces, key=lambda f: f[2] * f[3] * f[4])
    if prev_x is not None:
        near = min(faces, key=lambda f: abs(f[0] + f[2] / 2 - prev_x))
        if near[2] * near[3] * 1.6 >= best[2] * best[3]:
            best = near                      # mevcut kisiye sadik kal (gereksiz gecis yok)
    return best[0] + best[2] / 2


def analyze_path(video: Path, fps: float, src_w: int, src_h: int,
                 out_w: int, out_h: int, sample_fps: float = 6.0) -> CropPlan:
    import cv2

    cw, ch = crop_size(src_w, src_h, out_w, out_h)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"video acilamadi: {video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(fps / sample_fps)))
    scale = DETECT_WIDTH / src_w if src_w > DETECT_WIDTH else 1.0
    crop_w_s = cw * scale
    det = _FaceDetector()

    samples: list[tuple[int, float | None, bool]] = []   # (kare, merkez_x | None, kesim_mi)
    prev_hist = None
    prev_gray = None
    prev_x: float | None = None
    faces_found = 0
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else frame
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            cut = prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > SCENE_CUT
            prev_hist = hist
            if cut:
                prev_x = None

            gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (9, 9), 0)
            faces = det.detect(small)
            x: float | None = None
            if faces:
                faces_found += 1
                x = _choose_target(faces, prev_x, crop_w_s)
            elif prev_gray is not None and not cut:
                x = _motion_center(gray, prev_gray)
            prev_gray = gray
            if x is not None:
                prev_x = x
            samples.append((idx, None if x is None else x / scale, cut))
        idx += 1
    cap.release()
    n_frames = max(n_frames, idx)
    if n_frames == 0:
        raise RuntimeError("videoda kare bulunamadi")

    centers = _build_path(samples, n_frames, src_w, cw, fps)
    xs = np.clip(np.round(centers - cw / 2), 0, src_w - cw).astype(int)
    xs -= xs % 2
    log.info("Akilli crop: %d/%d ornekte yuz bulundu", faces_found, len(samples))
    return CropPlan("smart", cw, ch, xs, (src_h - ch) // 2, faces_found)


def _motion_center(gray: np.ndarray, prev: np.ndarray) -> float | None:
    import cv2
    diff = cv2.absdiff(gray, prev)
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] < 0.01 * mask.size:   # anlamli hareket yok
        return None
    return m["m10"] / m["m00"]


def _build_path(samples, n_frames: int, src_w: int, cw: int, fps: float) -> np.ndarray:
    """Ornek hedeflerinden kare basina yumusak kamera merkezi."""
    center = src_w / 2
    if not samples:
        return np.full(n_frames, center)
    # sahne parcalarina bol
    scenes: list[list[tuple[int, float | None]]] = [[]]
    for f, x, cut in samples:
        if cut and scenes[-1]:
            scenes.append([])
        scenes[-1].append((f, x))

    out = np.full(n_frames, center, dtype=float)
    lo_half, hi_half = cw / 2, src_w - cw / 2
    for si, scene in enumerate(scenes):
        f0 = scene[0][0] if si > 0 else 0
        f1 = scenes[si + 1][0][0] if si + 1 < len(scenes) else n_frames
        frames = np.array([f for f, _ in scene], dtype=float)
        vals = np.array([np.nan if x is None else x for _, x in scene], dtype=float)
        if np.all(np.isnan(vals)):
            vals[:] = center
        else:  # bosluklari komsu degerlerle doldur
            good = ~np.isnan(vals)
            vals = np.interp(frames, frames[good], vals[good])
        vals = np.clip(vals, lo_half, hi_half)
        vals = _median(vals, 5)
        vals = _dead_zone(vals, DEAD_ZONE * cw)
        sigma = max(1.0, SMOOTH_SEC * fps / max(1.0, (frames[1] - frames[0]) if len(frames) > 1 else 1.0))
        vals = _gauss(vals, sigma)
        span = np.arange(f0, f1, dtype=float)
        out[f0:f1] = np.interp(span, frames, vals)
    return np.clip(out, lo_half, hi_half)


def _median(v: np.ndarray, k: int) -> np.ndarray:
    if len(v) < k:
        return v
    pad = k // 2
    p = np.pad(v, pad, mode="edge")
    return np.array([np.median(p[i:i + k]) for i in range(len(v))])


def _dead_zone(v: np.ndarray, dz: float) -> np.ndarray:
    """Hedef kamera merkezinden dz'den az kaydiysa kamerayi oynatma."""
    out = np.empty_like(v)
    cam = v[0]
    for i, x in enumerate(v):
        if abs(x - cam) > dz:
            cam = x - np.sign(x - cam) * dz
        out[i] = cam
    return out


def _gauss(v: np.ndarray, sigma: float) -> np.ndarray:
    if len(v) < 3:
        return v
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(v, r, mode="edge"), k, mode="valid")
