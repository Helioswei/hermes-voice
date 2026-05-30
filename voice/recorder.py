import logging
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import torch
from silero_vad import load_silero_vad

logger = logging.getLogger("hermes-voice")


class AudioRecorder:
    """VAD-based audio recorder using sounddevice + silero-vad."""

    def __init__(self, samplerate=16000, channels=1, silence_timeout=1.5,
                 max_record_sec=15):
        self.samplerate = samplerate
        self.channels = channels
        self.silence_timeout = silence_timeout
        self.max_record_sec = max_record_sec

        logger.info("加载 Silero VAD 模型 …")
        self._vad = load_silero_vad(onnx=True)
        logger.info("Silero VAD 加载完毕")

        self._buffer = []
        self._speaking = False
        self._speech_end_time = None
        self._utterance_ready = threading.Event()
        self._utterance_result = None
        self._lock = threading.Lock()

        self._stream = None
        self._running = False

        # Wake-word hook — set via set_wake_hook()
        self._wake_hook = None
        self._wake_triggered = threading.Event()

        # Ring buffer for trailing audio after wake-word detection
        self._ring_buffer = deque(maxlen=int(samplerate * 5))  # 5 seconds
        self._ring_lock = threading.Lock()

    def start(self):
        if self._stream is not None:
            return
        self._running = True
        self._reset_state()

        logger.info("打开音频流 …")
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=512,
        )
        self._stream.start()
        logger.info("音频流已启动")

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_wake_hook(self, hook):
        """Set a callback(chunk) -> bool called on every audio chunk.

        If the hook returns True, ``wait_for_wake_word()`` unblocks.
        Pass ``None`` to disable.
        """
        self._wake_hook = hook
        self._wake_triggered.clear()

    def wait_for_wake_word(self, timeout=None):
        """Block until the wake hook signals a detection or *timeout* expires.

        Returns True if wake word was detected, False on timeout.
        """
        if not self._running:
            self.start()
        self._wake_triggered.clear()
        return self._wake_triggered.wait(timeout=timeout)

    def drain_ring_buffer(self):
        """Atomically drain and return the trailing-audio ring buffer as float32."""
        with self._ring_lock:
            if self._ring_buffer:
                audio = np.array(list(self._ring_buffer), dtype=np.float32)
                self._ring_buffer.clear()
                return audio
            return None

    # ------------------------------------------------------------------
    def read_utterance(self, idle_timeout=None):
        if not self._running:
            self.start()

        self._utterance_ready.clear()
        self._utterance_result = None
        self._reset_state()

        timeout_at = None
        if idle_timeout is not None:
            timeout_at = time.monotonic() + idle_timeout

        while self._running:
            remaining = _remaining_or_none(timeout_at)
            if remaining is not None and remaining <= 0:
                self._reset_state()
                return None

            if self._utterance_ready.wait(timeout=min(remaining or 0.1, 0.1)):
                with self._lock:
                    audio = self._utterance_result
                    self._utterance_result = None
                self._utterance_ready.clear()
                if audio is not None:
                    return audio

    def close(self):
        self.stop()

    def _reset_state(self):
        with self._lock:
            self._buffer = []
            self._speaking = False
            self._speech_end_time = None
        self._vad.reset_states()

    def _audio_callback(self, indata, frames, time_info, status):
        if not self._running:
            return

        if status:
            logger.warning("音频流状态异常: %s", status)

        # sounddevice delivers float32 in [-1, 1] on macOS
        mono = indata[:, 0].copy()

        # ── Wake-word hook (KWS) ─────────────────────
        if self._wake_hook is not None:
            try:
                int16_chunk = (mono * 32767).astype("int16")
                if self._wake_hook(int16_chunk):
                    self._wake_triggered.set()
            except Exception as exc:
                logger.error("唤醒检测失败: %s", exc)

        # ── Ring buffer (always active when running) ──
        with self._ring_lock:
            self._ring_buffer.extend(mono.tolist())

        try:
            tensor = torch.from_numpy(mono)
            prob = self._vad.audio_forward(tensor, self.samplerate).item()
        except Exception as exc:
            logger.error("VAD 失败: %s", exc)
            return

        now = time.monotonic()
        with self._lock:
            if prob > 0.5:
                if not self._speaking:
                    logger.debug("VAD 检测到语音 (prob=%.3f)", prob)
                    self._speaking = True
                    self._buffer = []
                    self._speech_end_time = None
                self._speech_end_time = None
                self._buffer.append(mono)
            elif self._speaking:
                self._buffer.append(mono)
                if self._speech_end_time is None:
                    self._speech_end_time = now
                elif now - self._speech_end_time >= self.silence_timeout:
                    logger.debug("语音结束 (%.1fs 静音)", self.silence_timeout)
                    audio = np.concatenate(self._buffer)
                    self._utterance_result = audio
                    self._speaking = False
                    self._buffer = []
                    self._speech_end_time = None
                    self._utterance_ready.set()
                    self._vad.reset_states()


def _remaining_or_none(timeout_at):
    if timeout_at is None:
        return None
    return timeout_at - time.monotonic()
