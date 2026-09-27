"""Writes tests/e2e/fakemic.wav: three seconds of two loud tone bursts. It is not real speech (the automated
integration test runs with SIGNBRIDGE_FAKE_MODELS=1, so Whisper is stubbed and never actually listens to it);
it only needs to be loud enough to cross the browser's voice-activity threshold when played into Chromium's
--use-file-for-fake-audio-capture, so the test exercises the real VAD and WAV encoder end to end.
Run once from the repository root: python tests/e2e/gen_fake_mic.py"""
import os
import numpy as np
import soundfile as sf

sr = 48000
t = np.arange(int(sr * 3)) / sr
sig = np.zeros_like(t)
for start, dur in [(0.3, 0.8), (1.5, 0.9)]:
    m = (t >= start) & (t < start + dur)
    sig[m] = 0.35 * np.sin(2 * np.pi * 180 * t[m]) * np.sign(np.sin(2 * np.pi * 3 * t[m]))
sig += 0.005 * np.random.RandomState(0).randn(len(t))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fakemic.wav")
sf.write(out, sig.astype(np.float32), sr, subtype="PCM_16")
print("wrote", out)
