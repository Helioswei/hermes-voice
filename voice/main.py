import logging
import os
import signal
import subprocess
import sys
import time
from logging.handlers import RotatingFileHandler

import yaml

from .recorder import AudioRecorder
from .wake_word import WakeWordDetector
from .stt import process_transcription
from .hermes_client import HermesClient
from .tts import TTSEngine

logger = logging.getLogger("hermes-voice")


def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    api_key = os.environ.get("HERMES_API_KEY") or cfg.get("hermes_api_key", "")
    cfg["hermes_api_key"] = api_key
    return cfg


def play_beep():
    try:
        subprocess.run(
            ["afplay", "/System/Library/Sounds/Blow.aiff"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def setup_logging():
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "hermes-voice.log")

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5_242_880, backupCount=3  # 5MB × 3
    )
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


def main():
    setup_logging()
    config = load_config()

    logger.info("initialising components …")

    recorder = AudioRecorder(
        samplerate=config.get("samplerate", 16000),
        silence_timeout=config.get("silence_timeout", 1.5),
        max_record_sec=config.get("max_record_sec", 15),
    )
    wake = WakeWordDetector(
        model_size=config.get("model_size", "tiny"),
        model_source=config.get("model_source", "huggingface"),
        model_path=config.get("model_path") or None,
    )
    hermes = HermesClient(
        base_url=config.get("hermes_url", "http://localhost:8642"),
        api_key=config["hermes_api_key"],
    )
    tts = TTSEngine()

    # ── 消息日志 ＋ TTS 期间关麦防回声 ──────────────
    def _speak_and_recover(recorder, tts, text):
        """关 → TTS → 开（读一次静默缓冲耗尽残留回声）。"""
        recorder.stop()          # 关麦
        try:
            tts.speak(text)      # 阻塞直到读完
        finally:
            recorder.start()     # 开麦（同步启动流）
            time.sleep(0.3)      # 等流稳定 + 排空残留缓冲区

    session_timeout = config.get("session_timeout", 30)

    def shutdown(sig, frame):
        logger.info("shutting down …")
        tts.stop()
        recorder.stop()
        hermes.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    state = "LISTENING"
    logger.info("[STATE] ready — say \"小九\" to wake me up")

    while True:
        try:
            if state == "LISTENING":
                audio = recorder.read_utterance(idle_timeout=None)

                if audio is None or len(audio) == 0:
                    continue

                text = wake.transcribe(audio)
                if not text:
                    continue

                logger.info("[MIC] %s", text)

                if wake.contains_wake_word(text):
                    play_beep()
                    command = process_transcription(text, state)
                    logger.info("[MIC] (command) %s", command)

                    reply = hermes.send(command or "你好")
                    logger.info("[API] → %s", command or "你好")
                    logger.info("[API] ← %s", reply)
                    logger.info("[TTS] %s", reply)
                    _speak_and_recover(recorder, tts, reply)
                    state = "AWAKE"
                    logger.info("[STATE] entered AWAKE (%.0fs follow-up window)",
                                session_timeout)
                else:
                    logger.info("[STATE] no wake word in text, staying in LISTENING")

            elif state == "AWAKE":
                audio = recorder.read_utterance(
                    idle_timeout=session_timeout
                )

                if audio is None:
                    logger.info("[STATE] follow-up window expired → LISTENING")
                    hermes.clear_context()
                    state = "LISTENING"
                    continue

                text = wake.transcribe(audio)
                if not text:
                    continue

                logger.info("[MIC] %s", text)
                command = process_transcription(text, state)

                if command:
                    reply = hermes.send(command)
                    logger.info("[API] → %s", command)
                    logger.info("[API] ← %s", reply)
                    logger.info("[TTS] %s", reply)
                    _speak_and_recover(recorder, tts, reply)

        except ConnectionError:
            logger.warning("Hermes API unreachable, falling back to LISTENING")
            _speak_and_recover(recorder, tts, "请先启动 Hermes 服务")
            state = "LISTENING"

        except KeyboardInterrupt:
            break

        except Exception:
            logger.exception("unexpected error, recovering …")
            state = "LISTENING"

    recorder.stop()
    hermes.close()


if __name__ == "__main__":
    main()
