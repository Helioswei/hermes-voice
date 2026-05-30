"""faster-whisper STT backend."""

import os

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

        model_dir = _resolve_whisper_model(model_size, model_source, model_path)
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


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def _resolve_whisper_model(model_size, model_source, model_path):
    if model_source == "local":
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"model_source=local but model_path not found: {model_path}"
            )
        return model_path

    if model_source == "modelscope":
        return _prepare_whisper_modelscope(model_size)

    # huggingface → faster-whisper downloads automatically
    return model_size


def _prepare_whisper_modelscope(model_size, repo=None):
    import shutil

    from modelscope import snapshot_download

    if not repo:
        repo = f"openai-mirror/whisper-{model_size}"

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    ct2_path = os.path.join(project_root, "models", f"whisper-{model_size}")
    os.makedirs(ct2_path, exist_ok=True)

    if os.path.isfile(os.path.join(ct2_path, "model.bin")):
        return ct2_path

    raw_path = snapshot_download(repo, ignore_file_pattern=["*.msgpack", "*.h5"])

    from ctranslate2.converters import TransformersConverter

    if os.path.isdir(ct2_path):
        shutil.rmtree(ct2_path)

    print(f"Converting {repo} to CTranslate2 format …")
    converter = TransformersConverter(
        model_name_or_path=raw_path,
        copy_files=["tokenizer.json", "preprocessor_config.json"],
    )
    converter.convert(
        output_dir=ct2_path,
        quantization="float16" if _has_gpu() else "float32",
    )
    print(f"Converted model saved to {ct2_path}")
    return ct2_path


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
