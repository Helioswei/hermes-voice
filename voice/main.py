import logging
import os
import signal
import subprocess
import sys
import time
import warnings
from logging.handlers import RotatingFileHandler

import yaml

warnings.filterwarnings("ignore", message=".*pkg_resources.*")
import zhconv

from .av_recorder import AVRecorder
from .stt_engine import create_stt_engine
from .wake_word_engine import WakeWordEngine
from .hermes_client import HermesClient
from .tts import TTSEngine

logger = logging.getLogger("hermes-voice")

# ANSI color codes
_COLORS = {
    logging.DEBUG: "\033[2m",       # dim
    logging.INFO: "\033[0m",        # default
    logging.WARNING: "\033[33m",    # yellow
    logging.ERROR: "\033[31m",      # red
    logging.CRITICAL: "\033[35m",   # magenta
}
_RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Formatter with ANSI colors, filename, and line number."""

    def format(self, record):
        color = _COLORS.get(record.levelno, "")
        # Override levelname to show filename:lineno
        record.colored_levelname = (
            f"{color}[{record.filename}:{record.lineno}]{_RESET}"
        )
        # Build format manually for clean output
        ts = self.formatTime(record, "%H:%M:%S")
        return (
            f"{color}{ts}{_RESET} "
            f"{record.colored_levelname} "
            f"{record.getMessage()}"
        )


def load_config(path="config.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)

    api_key = os.environ.get("HERMES_API_KEY") or cfg.get("hermes_api_key", "")
    cfg["hermes_api_key"] = api_key
    return cfg


def _to_simplified(text):
    """Convert traditional Chinese to simplified."""
    return zhconv.convert(text, "zh-cn")


def _strip_wake_word(text, keywords):
    """Strip wake word prefix from transcribed text."""
    for kw in sorted(keywords, key=len, reverse=True):
        if text.startswith(kw):
            return text[len(kw):].lstrip("，, ").strip()
    return text


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

    # File handler: plain format with filename:lineno
    file_fmt = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5_242_880, backupCount=3
    )
    file_handler.setFormatter(file_fmt)

    # Console handler: colored + filename:lineno
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


def main():
    setup_logging()
    config = load_config()

    logger.info("初始化组件 …")

    recorder = AVRecorder(
        samplerate=config.get("samplerate", 16000),
        silence_timeout=config.get("silence_timeout", 1.5),
        max_record_sec=config.get("max_record_sec", 15),
    )
    stt = create_stt_engine(config)
    wake_words = config.get("kws_keywords", ["小九"])
    kws = WakeWordEngine(
        keywords=wake_words,
        threshold=config.get("kws_threshold", 0.25),
    )
    hermes = HermesClient(
        base_url=config.get("hermes_url", "http://localhost:8642"),
        api_key=config["hermes_api_key"],
    )
    tts = TTSEngine()

    def _speak_and_recover(recorder, tts, text):
        """Speak TTS reply with barge-in support.

        Returns audio bytes (numpy float32) if user interrupted TTS,
        None if TTS completed normally or text was empty.
        """
        if not text or not text.strip():
            return None

        recorder.set_tts_active(True, config.get("bargein_threshold", 0.65))

        def check_barge_in():
            return recorder.check_interrupt(
                min_duration=config.get("bargein_duration", 0.3)
            )

        completed = tts.speak(text, interrupt_check=check_barge_in)
        recorder.set_tts_active(False)

        if not completed:
            logger.info("TTS 被用户打断")
            # 等待 VAD 完成当前语句的缓冲（最多 3 秒）
            audio = recorder.read_utterance(idle_timeout=3.0)
            return audio

        return None

    def _process_interruption(recorder, stt, hermes, tts, interrupted_audio):
        """Handle audio captured during TTS barge-in.

        If the user spoke during TTS playback and we have their audio,
        transcribe it, send to Hermes, and read the reply.
        """
        if interrupted_audio is None:
            return
        # 最多等 3 秒让用户把话说完
        audio = recorder.read_utterance(idle_timeout=3.0)
        if audio is None:
            return

        text = _to_simplified(stt.transcribe(audio))
        if not text or not text.strip():
            return

        logger.info("打断→ %s", text)
        try:
            reply = hermes.send(text)
            logger.info("API → %s", reply)
            _speak_and_recover(recorder, tts, reply)
        except ConnectionError:
            logger.warning("打断处理: Hermes API 不可达")

    session_timeout = config.get("session_timeout", 30)

    shutdown_requested = False

    def shutdown(sig, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        logger.info("正在关闭 …")
        tts.stop()
        recorder.stop()
        kws.close()
        hermes.close()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    state = "LISTENING"
    logger.info("状态机 → 就绪，说\"小九\"唤醒我")

    while not shutdown_requested:
        try:
            if state == "LISTENING":
                recorder.set_wake_hook(kws.process_chunk)

                if not recorder.wait_for_wake_word():
                    continue

                detected = kws.last_keyword
                recorder.set_wake_hook(None)
                recorder.stop()
                kws.reset()
                play_beep()
                tts.speak("我在")

                logger.info("唤醒词 → 检测到\"%s\"，等待指令 …", detected)

                recorder.start()
                audio = recorder.read_utterance(
                    idle_timeout=session_timeout
                )
                if audio is None:
                    logger.info("状态机 → 没听到指令，回到待唤醒")
                    continue

                text = _to_simplified(stt.transcribe(audio))
                if not text or not text.strip():
                    continue

                logger.info("麦克风→ %s", text)

                reply = hermes.send(text)
                logger.info("API → %s", reply)
                logger.info("朗读 → %s", reply)
                interrupted = _speak_and_recover(recorder, tts, reply)
                state = "AWAKE"
                recorder.start()
                logger.info("状态机 → 进入跟随时窗 (%.0f秒)",
                            session_timeout)
                _process_interruption(recorder, stt, hermes, tts, interrupted)

            elif state == "AWAKE":
                audio = recorder.read_utterance(
                    idle_timeout=session_timeout
                )

                if audio is None:
                    logger.info("状态机 → 跟随时窗超时，回到待唤醒")
                    hermes.clear_context()
                    state = "LISTENING"
                    continue

                text = _to_simplified(stt.transcribe(audio))
                if not text or not text.strip():
                    continue

                logger.info("麦克风→ %s", text)

                reply = hermes.send(text)
                logger.info("API → %s", reply)
                logger.info("朗读 → %s", reply)
                interrupted = _speak_and_recover(recorder, tts, reply)
                _process_interruption(recorder, stt, hermes, tts, interrupted)

        except ConnectionError as e:
            logger.warning("Hermes API 不可达 (%s)，回到待唤醒", e)
            _speak_and_recover(recorder, tts, "请先启动 Hermes 服务")
            state = "LISTENING"

        except KeyboardInterrupt:
            break

        except Exception:
            logger.exception("意外错误，恢复中 …")
            state = "LISTENING"

    recorder.stop()
    hermes.close()


if __name__ == "__main__":
    main()
