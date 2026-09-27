import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

GREEN = "#3aa17e"; DARK = "#1f3b34"; GREY = "#5b6b66"; BG = "#f4f7f6"
PRE = "#2f6f8f"
fig, ax = plt.subplots(figsize=(10, 6.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")

def box(x, y, w, h, text, fc="white", ec=GREY, tc="#15201d", fs=8.4, tag=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)
    if tag:
        ax.text(x + w - 0.07, y + h - 0.13, tag, ha="right", va="top", fontsize=6.2,
                color=PRE, style="italic")

def arrow(x1, y1, x2, y2, color=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.4, color=color))

# headers
ax.text(2.55, 6.15, "Pipeline A   sign \u2192 speech   (Deaf / Mute user)",
        ha="center", fontsize=10, fontweight="bold", color=GREEN)
ax.text(7.45, 6.15, "Pipeline B   speech \u2192 sign + captions   (Hearing user)",
        ha="center", fontsize=10, fontweight="bold", color=GREEN)
ax.plot([5, 5], [0.2, 5.85], color="#cfd8d5", lw=1, ls="--")

# client/server band labels
ax.text(0.12, 5.45, "CLIENT (browser)", rotation=90, va="center", fontsize=7, color=PRE)
ax.text(0.12, 2.0, "SERVER (FastAPI)", rotation=90, va="center", fontsize=7, color=PRE)

# Pipeline A (left)
ya = 5.15
box(0.7, ya, 3.7, 0.55, "Webcam capture  (getUserMedia)", fc="#eaf3ef")
arrow(2.55, ya, 2.55, ya-0.28)
box(0.7, ya-0.95, 3.7, 0.6, "MediaPipe Hands \u2014 21 landmarks (WASM)", fc="#eaf3ef", tag="pre-trained")
arrow(2.55, ya-0.95, 2.55, ya-1.23)
box(0.7, ya-1.95, 3.7, 0.6, "ONNX gesture classifier\n42 coords \u2192 ASL letter", fc="#dff0e8", ec=GREEN, tag="CUSTOM (prototype)")
arrow(2.55, ya-1.95, 2.55, ya-2.23)
box(0.7, ya-2.95, 3.7, 0.6, "Llama 3:8b (Ollama) \u2014 sequence \u2192 sentence", fc="#e7eef7", tag="pre-trained")
arrow(2.55, ya-2.95, 2.55, ya-3.23)
box(0.7, ya-3.85, 3.7, 0.55, "pyttsx3 TTS \u2192 voice to caller", fc="white")

# Pipeline B (right)
yb = 5.15
box(5.6, yb, 3.7, 0.55, "Microphone capture  (16 kHz mono)", fc="#eaf3ef")
arrow(7.45, yb, 7.45, yb-0.28)
box(5.6, yb-0.95, 3.7, 0.6, "OpenAI Whisper (base) \u2014 speech \u2192 text", fc="#e7eef7", tag="pre-trained")
arrow(7.45, yb-0.95, 7.45, yb-1.23)
box(5.6, yb-1.95, 3.7, 0.6, "XLM-RoBERTa sentiment \u2014 5-class tone", fc="#e7eef7", tag="pre-trained")
arrow(7.45, yb-1.95, 7.45, yb-2.23)
box(5.6, yb-2.95, 3.7, 0.6, "Sign-avatar renderer  (text \u2192 ASL avatar)", fc="white")
arrow(7.45, yb-2.95, 7.45, yb-3.23)
box(5.6, yb-3.85, 3.7, 0.55, "Captions + emoji sentiment overlay", fc="white")

ax.text(5, 0.06, "Privacy boundary: only 42 landmark coordinates leave the browser; "
                 "all model inference runs on-device.",
        ha="center", fontsize=7.2, color=GREY, style="italic")
plt.tight_layout()
plt.savefig("/home/claude/signbridge_prototype/artifacts/fig_architecture.png", dpi=160,
            bbox_inches="tight", facecolor="white")
print("architecture diagram saved")
