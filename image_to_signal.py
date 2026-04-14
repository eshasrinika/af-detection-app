import cv2
import numpy as np
from scipy.signal import medfilt, butter, filtfilt, find_peaks
import torch
import torch.nn as nn


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class FusionAttentionNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True,
            bidirectional=True
        )

        self.attn = nn.Linear(128, 1)

        self.fc = nn.Sequential(
            nn.Linear(128 + 19, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x_ecg, x_feat):
        x = self.cnn(x_ecg)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)

        w = torch.softmax(self.attn(x), dim=1)  # 🔥 attention
        x_weighted = (x * w).sum(dim=1)

        fused = torch.cat([x_weighted, x_feat], dim=1)
        out = self.fc(fused)

        return out, w  # 🔥 return attention


# Load model
model = FusionAttentionNet()
model.load_state_dict(torch.load("af_model.pth", map_location='cpu'))
model.eval()


# ─────────────────────────────────────────────
# IMAGE LOADING
# ─────────────────────────────────────────────
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


# ─────────────────────────────────────────────
# GRID REMOVAL
# ─────────────────────────────────────────────
def remove_grid(gray):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    v_lines = cv2.morphologyEx(blackhat, cv2.MORPH_OPEN, v_kernel)
    blackhat = cv2.subtract(blackhat, v_lines)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    h_lines = cv2.morphologyEx(blackhat, cv2.MORPH_OPEN, h_kernel)
    blackhat = cv2.subtract(blackhat, h_lines)

    return blackhat


# ─────────────────────────────────────────────
# THRESHOLD
# ─────────────────────────────────────────────
def extract_mask(img):
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


# ─────────────────────────────────────────────
# SIGNAL EXTRACTION
# ─────────────────────────────────────────────
def extract_signal(mask):
    h, w = mask.shape
    signal = np.zeros(w)

    for col in range(w):
        rows = np.where(mask[:, col] > 0)[0]

        if len(rows) > 0:
            signal[col] = float(np.min(rows))  # preserve peaks
        else:
            signal[col] = np.nan

    signal = h - signal

    # Handle NaNs safely
    valid = ~np.isnan(signal)
    if valid.sum() < 10:
        raise ValueError("Signal extraction failed")

    nan_signal = signal.copy()
    nan_indices = np.where(~valid)[0]

    if len(nan_indices) > 0:
        gaps = np.split(nan_indices, np.where(np.diff(nan_indices) > 1)[0] + 1)

        for gap in gaps:
            left = gap[0] - 1
            right = gap[-1] + 1

            if left >= 0 and right < len(signal):
                nan_signal[gap] = np.linspace(signal[left], signal[right], len(gap))
            elif left >= 0:
                nan_signal[gap] = signal[left]
            elif right < len(signal):
                nan_signal[gap] = signal[right]

    return nan_signal


# ─────────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────────
def process_signal(signal, target_len=3000):
    signal = medfilt(signal, 3)

    fs_equiv = 180
    cutoff_hz = 0.5
    b, a = butter(1, cutoff_hz / (fs_equiv / 2), btype='high')
    signal = filtfilt(b, a, signal)

    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)

    return np.interp(
        np.linspace(0, len(signal), target_len),
        np.arange(len(signal)),
        signal
    )


# ─────────────────────────────────────────────
# RR FEATURES
# ─────────────────────────────────────────────
def extract_rr_features(signal):
    features = np.zeros(19, dtype=np.float32)

    # 🔥 FIXED peak detection
    peaks, _ = find_peaks(signal, distance=120, prominence=1.3)

    if len(peaks) < 3:
        return features

    rr = np.diff(peaks).astype(np.float32)

    # 🔥 filter unrealistic intervals
    rr = rr[(rr > 50) & (rr < 400)]

    if len(rr) < 2:
        return features

    mean_rr = float(np.mean(rr))
    std_rr = float(np.std(rr))
    cv_rr = std_rr / (mean_rr + 1e-6)
    rmssd = float(np.sqrt(np.mean(np.diff(rr)**2)))
    irregularity = float(np.std(np.diff(rr)) / (mean_rr + 1e-6))

    features[:8] = [
        mean_rr,
        std_rr,
        cv_rr,
        rmssd,
        0,
        0,
        float(np.max(rr) - np.min(rr)),
        irregularity
    ]

    # 🔥 NSR stabilizer
    if features[1] < 15:
        features *= 0.7

    return features


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run(image_path):
    img, gray = load_image(image_path)

    cleaned = remove_grid(gray)
    mask = extract_mask(cleaned)

    raw = extract_signal(mask)
    final_signal = process_signal(raw)

    rr_features = extract_rr_features(final_signal)

    x_ecg = torch.tensor(final_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x_feat = torch.tensor(rr_features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output, attn = model(x_ecg, x_feat)
        probs = torch.softmax(output, dim=1)

    pred = torch.argmax(output, dim=1).item()

    return pred, probs.numpy(), final_signal, attn.squeeze().numpy()