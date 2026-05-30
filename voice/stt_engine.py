"""STT engine abstraction.

Add new engines by subclassing ``STTEngine``, implementing ``transcribe()``,
and registering in ``create_stt_engine()``.
"""


class STTEngine:
    """Abstract STT engine."""

    def transcribe(self, audio):
        """Transcribe float32 audio (16kHz mono) to text."""
        raise NotImplementedError


def create_stt_engine(config):
    """Create an STT engine from configuration dict.

    config keys: ``stt_engine``, ``stt_model_size``, ``stt_model_source``,
    ``stt_model_path``, ``stt_language``.
    """
    engine_type = config.get("stt_engine", "whisper")

    if engine_type == "whisper":
        from .stt_whisper import WhisperSTT
        return WhisperSTT(
            model_size=config.get("stt_model_size", "small"),
            model_source=config.get("stt_model_source", "modelscope"),
            model_path=config.get("stt_model_path") or None,
        )

    if engine_type == "sensevoice":
        from .stt_sensevoice import SenseVoiceSTT
        return SenseVoiceSTT(
            language=config.get("stt_language", "zh"),
        )

    raise ValueError(f"Unknown STT engine: {engine_type}")
