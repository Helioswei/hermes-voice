"""SenseVoice STT backend via sherpa-onnx."""

from .models import resolve_sensevoice_model
from .stt_engine import STTEngine


class SenseVoiceSTT(STTEngine):
    """SenseVoice STT engine via sherpa-onnx.

    Downloads the ONNX model from HF mirror on first use.
    """

    def __init__(self, language="zh"):
        import sherpa_onnx

        model_path, tokens_path = resolve_sensevoice_model()
        self._model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=2,
            sample_rate=16000,
            language=language,
            use_itn=True,
        )
        self._stream = self._model.create_stream()

    def transcribe(self, audio):
        if audio is None or len(audio) == 0:
            return ""
        self._stream.accept_waveform(16000, audio)
        self._model.decode_stream(self._stream)
        text = self._stream.result.text
        self._stream = self._model.create_stream()
        return text.strip()
