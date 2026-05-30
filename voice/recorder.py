import logging
import threading
import time

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

        logger.info("loading silero-vad model …")
        self._vad = load_silero_vad(onnx=True)
        logger.info("silero-vad loaded")

        self._buffer = []
        self._speaking = False
        self._speech_end_time = None
        self._utterance_ready = threading.Event()
        self._utterance_result = None
        self._lock = threading.Lock()

        self._stream = None
        self._running = False

    def start(self):
        if self._stream is not None:
            return
        self._running = True
        self._reset_state()

        logger.info("opening audio stream …")
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=512,
        )
        self._stream.start()
        logger.info("audio stream started, listening for wake word …")

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

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
            logger.warning("audio stream status: %s", status)

        # sounddevice delivers float32 in [-1, 1] on macOS
        mono = indata[:, 0].copy()

        try:
            tensor = torch.from_numpy(mono)
            prob = self._vad.audio_forward(tensor, self.samplerate).item()
        except Exception as exc:
            logger.error("VAD failed: %s", exc)
            return

        now = time.monotonic()
        with self._lock:
            if prob > 0.5:
                if not self._speaking:
                    logger.debug("speech started (VAD prob=%.3f)", prob)
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
                    logger.debug("speech ended (%.1fs silence)", self.silence_timeout)
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
