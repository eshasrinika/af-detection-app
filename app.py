import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# MODEL DEFINITION
# -------------------------------
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
        x = x.permute(0,2,1)

        x,_ = self.lstm(x)

        w = torch.softmax(self.attn(x), dim=1)
        x = (x * w).sum(dim=1)

        fused = torch.cat([x, x_feat], dim=1)

        return self.fc(fused)   # 🔥 ONLY output (matches saved model)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model = FusionAttentionNet()
    model.load_state_dict(torch.load("af_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------------
# ATTENTION EXTRACT FUNCTION
# -------------------------------
def get_attention(model, x_ecg):
    with torch.no_grad():
        x = model.cnn(x_ecg)
        x = x.permute(0,2,1)

        x,_ = model.lstm(x)

        w = torch.softmax(model.attn(x), dim=1)

    return w

# -------------------------------
# UI
# -------------------------------
st.title("🫀 Atrial Fibrillation Detection")

st.write("Upload ECG signal (3000 samples)")

uploaded_file = st.file_uploader("Upload .npy file", type=["npy"])

if uploaded_file is not None:

    ecg = np.load(uploaded_file)
    ecg = ecg.squeeze()

    if ecg.shape[0] != 3000:
        st.error(f"Invalid ECG length: {ecg.shape[0]}")
    else:
        st.success("ECG Loaded")

        st.subheader("Prediction Result")

        feat = np.zeros(19)

        x_ecg = torch.tensor(ecg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x_feat = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

        # -------------------------------
        # PREDICTION
        # -------------------------------
        with torch.no_grad():
            out = model(x_ecg, x_feat)
            probs = torch.softmax(out, dim=1)

        af_prob = probs[0,1].item()

        # -------------------------------
        # ATTENTION (SAFE EXTRACTION)
        # -------------------------------
        attn = get_attention(model, x_ecg)
        attention = attn[0].cpu().numpy().squeeze()

        attention = (attention - attention.min()) / (attention.max() - attention.min())

        # -------------------------------
        # DISPLAY RESULT
        # -------------------------------
        if af_prob > 0.45:
            st.error(f"⚠️ AF Detected ({af_prob*100:.2f}%)")
        else:
            st.success(f"✅ Normal (NSR) ({(1-af_prob)*100:.2f}%)")

        st.write(f"AF Probability: {af_prob*100:.2f}%")
        st.write(f"NSR Probability: {(1-af_prob)*100:.2f}%")

        # -------------------------------
        # ECG PLOT
        # -------------------------------
        fig, ax = plt.subplots()
        ax.plot(ecg)
        ax.set_title("ECG Signal")
        st.pyplot(fig)

        # -------------------------------
        # ATTENTION PLOT
        # -------------------------------
        fig2, ax2 = plt.subplots()

        ax2.plot(ecg, label="ECG")

        attention_scaled = attention * np.max(np.abs(ecg))
        ax2.plot(attention_scaled, color='red', label="Attention")

        ax2.set_title("ECG with Attention")
        ax2.legend()

        st.pyplot(fig2)