"""
audio_manager.py - Generates and plays synthesized sci-fi / high-tech sound effects
using numpy wave synthesis. No external audio files needed.
"""
import numpy as np
import threading
import wave
import io
import os
import struct
import math


def _generate_wave(duration: float, sample_rate: int = 44100) -> bytes:
    """Build a PCM byte buffer."""
    raise NotImplementedError


class AudioManager:
    """
    Plays synthesized audio tones for tracking state changes.
    Uses only stdlib + numpy — no additional audio libs required on Windows
    because we write a temp .wav and use winsound.
    """

    def __init__(self, enabled: bool = True, volume: float = 0.8):
        self.enabled = enabled
        self.volume = max(0.0, min(1.0, volume))
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play_detected(self):
        """Two rising tones — object found."""
        self._play_async(self._synth_detected)

    def play_lost(self):
        """Descending, warbling tone — object lost."""
        self._play_async(self._synth_lost)

    def play_redetecting(self):
        """Short pulsing beep — retrying detection."""
        self._play_async(self._synth_redetecting)

    def play_recovered(self):
        """Bright ascending arpeggio — tracking recovered."""
        self._play_async(self._synth_recovered)

    def play_error(self):
        """Low buzz — max retries exceeded."""
        self._play_async(self._synth_error)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _play_async(self, fn):
        if not self.enabled:
            return
        t = threading.Thread(target=fn, daemon=True)
        t.start()

    @staticmethod
    def _to_pcm(samples: np.ndarray) -> bytes:
        """Convert float32 [-1,1] array to 16-bit PCM bytes."""
        clipped = np.clip(samples, -1.0, 1.0)
        ints = (clipped * 32767).astype(np.int16)
        return ints.tobytes()

    def _play_pcm(self, pcm: bytes, sample_rate: int = 44100):
        """Write PCM to an in-memory WAV and play via winsound (Windows only)."""
        try:
            import winsound
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm)
            buf.seek(0)
            # winsound only supports file paths on some builds; write to temp
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(buf.read())
                tmp_path = tmp.name
            winsound.PlaySound(tmp_path, winsound.SND_FILENAME | winsound.SND_NODEFAULT)
            os.unlink(tmp_path)
        except Exception:
            pass  # Silently fail if audio not available

    # ------------------------------------------------------------------
    # Sound synthesis recipes
    # ------------------------------------------------------------------

    SR = 44100

    def _make_tone(self, freq: float, duration: float, shape: str = "sine",
                   attack: float = 0.02, release: float = 0.05) -> np.ndarray:
        n = int(self.SR * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        if shape == "sine":
            wave_data = np.sin(2 * np.pi * freq * t)
        elif shape == "sawtooth":
            wave_data = 2 * (t * freq - np.floor(t * freq + 0.5))
        elif shape == "square":
            wave_data = np.sign(np.sin(2 * np.pi * freq * t))
        else:
            wave_data = np.sin(2 * np.pi * freq * t)

        # Envelope
        env = np.ones(n)
        atk = int(attack * self.SR)
        rel = int(release * self.SR)
        if atk > 0:
            env[:atk] = np.linspace(0, 1, atk)
        if rel > 0 and rel < n:
            env[-rel:] = np.linspace(1, 0, rel)
        return wave_data * env * self.volume

    def _concat(self, *arrays) -> np.ndarray:
        return np.concatenate(arrays)

    def _synth_detected(self):
        a = self._make_tone(440, 0.08, "sine", attack=0.01, release=0.03)
        b = self._make_tone(660, 0.12, "sine", attack=0.01, release=0.05)
        gap = np.zeros(int(self.SR * 0.04))
        pcm = self._to_pcm(self._concat(a, gap, b))
        self._play_pcm(pcm)

    def _synth_lost(self):
        # Descending warble
        n = int(self.SR * 0.35)
        t = np.linspace(0, 0.35, n)
        freq = 600 - 300 * t / 0.35  # 600 → 300 Hz
        phase = 2 * np.pi * np.cumsum(freq) / self.SR
        wave_data = np.sin(phase) * self.volume
        # Vibrato
        vibrato = 1 + 0.05 * np.sin(2 * np.pi * 8 * t)
        wave_data *= vibrato
        env = np.ones(n)
        env[-int(0.1 * self.SR):] = np.linspace(1, 0, int(0.1 * self.SR))
        self._play_pcm(self._to_pcm(wave_data * env))

    def _synth_redetecting(self):
        a = self._make_tone(880, 0.05, "sine", attack=0.005, release=0.02)
        gap = np.zeros(int(self.SR * 0.06))
        pcm = self._to_pcm(self._concat(a, gap, a))
        self._play_pcm(pcm)

    def _synth_recovered(self):
        notes = [330, 440, 550, 660]
        parts = []
        for i, f in enumerate(notes):
            parts.append(self._make_tone(f, 0.07, "sine", attack=0.01, release=0.02))
            parts.append(np.zeros(int(self.SR * 0.02)))
        pcm = self._to_pcm(self._concat(*parts))
        self._play_pcm(pcm)

    def _synth_error(self):
        a = self._make_tone(150, 0.2, "sawtooth", attack=0.01, release=0.08)
        gap = np.zeros(int(self.SR * 0.05))
        pcm = self._to_pcm(self._concat(a, gap, a))
        self._play_pcm(pcm)
