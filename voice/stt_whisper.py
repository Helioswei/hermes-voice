"""faster-whisper STT backend."""

from .models import resolve_whisper_model
from .stt_engine import STTEngine


class WhisperSTT(STTEngine):
    """faster-whisper based STT engine.

    Parameters
    ----------
    model_size : str
        ``tiny`` / ``base`` / ``small`` / ``medium``.
    model_source : str
        ``modelscope`` / ``huggingface`` / ``local``.
    model_path : str or None
        Path when ``model_source=local``, otherwise unused.
    """

    def __init__(self, model_size="small", model_source="modelscope",
                 model_path=None, device="cpu", compute_type="int8"):
        from faster_whisper import WhisperModel

        model_dir = resolve_whisper_model(model_size, model_source, model_path)
        self._model = WhisperModel(model_dir, device=device,
                                   compute_type=compute_type)

    def transcribe(self, audio):
        if audio is None or len(audio) == 0:
            return ""
        segments, _ = self._model.transcribe(
            audio, language="zh",
            beam_size=5, vad_filter=True,
        )
        return "".join(seg.text for seg in segments).strip()
