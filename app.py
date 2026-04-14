import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from image_to_signal import run

st.set_page_config(page_title="AF Detection", layout="centered")

# ─────────────────────────────
# CLEAN ADAPTIVE STYLING (LIGHT + DARK FRIENDLY)
# ─────────────────────────────
st.markdown("""
<style>

/* Headings */
h1, h2, h3 {
    font-weight: 700;
}

/* Improve general text visibility */
.stMarkdown, p {
    font-size: 16px;
}

/* Captions clearer */
.stCaption {
    opacity: 0.9 !important;
}

/* Buttons */
button {
    border-radius: 8px !important;
}

/* Radio spacing */
.stRadio > div {
    gap: 20px;
}

/* Alert boxes */
div[data-testid="stAlert"] {
    border-radius: 10px;
    padding: 12px;
}

/* Plot background transparent (auto adapts) */
.plot-container {
    background: transparent !important;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────
# TITLE
# ─────────────────────────────
st.title("🫀 Atrial Fibrillation Detection")
st.caption("Upload ECG image or use sample data")

# ─────────────────────────────
# INPUT
# ─────────────────────────────
option = st.radio(
    "Choose input:",
    ["None", "Sample NSR", "Sample AF"],
    horizontal=True
)

file_path = None

if option == "Sample NSR":
    file_path = "nsr.png"
elif option == "Sample AF":
    file_path = "af2.png"

uploaded_file = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_path = "temp.png"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

# ─────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────
if file_path is not None:

    with st.spinner("Processing ECG..."):
        pred, probs, signal, attn = run(file_path)

    st.success("Processing complete ✅")

    # ─────────────────────────
    # PREDICTION
    # ─────────────────────────
    st.subheader("📊 Prediction")

    if pred == 1:
        st.error("⚠️ Atrial Fibrillation Detected")
    else:
        st.success("✅ Normal Sinus Rhythm")

    st.markdown(f"**AF Confidence:** {probs[0][1]*100:.2f}%")
    st.markdown(f"**NSR Confidence:** {probs[0][0]*100:.2f}%")

    # ─────────────────────────
    # ECG SIGNAL
    # ─────────────────────────
    st.subheader("📈 ECG Signal")

    fig, ax = plt.subplots()
    ax.plot(signal, linewidth=1.5)
    ax.set_title("Extracted ECG Signal")

    # adaptive colors
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)

    st.pyplot(fig)

    # ─────────────────────────
    # ATTENTION VISUALIZATION
    # ─────────────────────────
    st.subheader("🧠 Attention Visualization")

    fig2, ax2 = plt.subplots()

    # Smooth attention
    window = 100
    attn_smooth = np.convolve(attn, np.ones(window)/window, mode='same')
    attn_smooth = np.convolve(attn_smooth, np.ones(50)/50, mode='same')

    # Normalize
    attn_smooth = (attn_smooth - attn_smooth.min()) / (attn_smooth.max() - attn_smooth.min() + 1e-6)

    # Softer peaks
    attn_smooth = np.sqrt(attn_smooth)

    # Scale
    signal_range = max(signal) - min(signal)
    attn_scaled = attn_smooth * signal_range * 0.5

    # Plot signal
    ax2.plot(signal, linewidth=1.5, label="ECG Signal")

    # Highlight regions
    threshold = 0.75
    important = np.convolve(
        (attn_smooth > threshold).astype(float),
        np.ones(30)/30,
        mode='same'
    ) > 0.5

    ax2.fill_between(
        range(len(signal)),
        signal,
        signal + attn_scaled,
        where=important,
        color = 'red',
        alpha=0.3,
        label="High Attention"
    )

    # Light attention curve
    ax2.plot(attn_scaled, alpha=0.4, linewidth=1)

    ax2.set_title("Model Focus Regions")
    ax2.legend()

    ax2.set_facecolor("none")
    fig2.patch.set_alpha(0)

    st.pyplot(fig2)

    st.caption("Highlighted regions indicate where the model focuses most")