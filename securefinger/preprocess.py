"""Fingerprint preprocessing: grayscale -> ROI crop -> CLAHE -> 96x96 -> zero-mean/unit-variance."""
import cv2
import numpy as np

from . import IMG_SIZE


def _roi_crop(img: np.ndarray, margin: float = 0.10) -> np.ndarray:
    """Crop to the ridge region (bounding box of dark pixels) with a small margin."""
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:  # nothing detected: keep the full image
        return img
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    h, w = img.shape
    mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
    return img[max(0, y0 - my):min(h, y1 + my + 1), max(0, x0 - mx):min(w, x1 + mx + 1)]


def preprocess_array(img: np.ndarray) -> np.ndarray:
    """Return a float32 array of shape (1, IMG_SIZE, IMG_SIZE)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.uint8)
    img = _roi_crop(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    return x[None, :, :]


def load_and_preprocess(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return preprocess_array(img)


def decode_and_preprocess(data: bytes) -> np.ndarray:
    """Preprocess an uploaded image given as raw bytes (BMP/PNG/JPG)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return preprocess_array(img)
