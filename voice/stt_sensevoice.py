"""SenseVoice STT backend via sherpa-onnx."""

import os

from .stt_engine import STTEngine


class SenseVoiceSTT(STTEngine):
    """SenseVoice STT engine via sherpa-onnx.

    Downloads the ONNX model from HF mirror on first use.
    """

    def __init__(self, language="zh"):
        import sherpa_onnx

        model_path, tokens_path = _resolve_sensevoice_model()
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


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def _resolve_sensevoice_model():
    from urllib.request import urlretrieve

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    model_dir = os.path.join(project_root, "models", "sensevoice")
    os.makedirs(model_dir, exist_ok=True)

    model_file = os.path.join(model_dir, "model.onnx")
    tokens_file = os.path.join(model_dir, "tokens.txt")

    if os.path.isfile(model_file) and os.path.isfile(tokens_file):
        return model_file, tokens_file

    base = "https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main"

    print("Downloading SenseVoice ONNX model …")
    urlretrieve(f"{base}/model.onnx", model_file)
    urlretrieve(f"{base}/tokens.txt", tokens_file)

    print(f"SenseVoice model cached at {model_dir}")
    return model_file, tokens_file
