"""Model management utilities.

Centralizes model resolution, download, and caching for all STT engines.
"""

import os
import shutil
import glob


def _project_root():
    """Return absolute path to project root (parent of voice/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def model_dir(subdir: str) -> str:
    """Return absolute path to ``models/<subdir>/``."""
    path = os.path.join(_project_root(), "models", subdir)
    os.makedirs(path, exist_ok=True)
    return path


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ── Whisper ──────────────────────────────────────────────────────────

def resolve_whisper_model(model_size, source="modelscope", path=None):
    """Return CTranslate2 whisper model directory path.

    Parameters
    ----------
    model_size : str
        ``tiny`` / ``base`` / ``small`` / ``medium``.
    source : str
        ``modelscope`` (download + convert, default), ``huggingface`` (auto),
        or ``local`` (pre-downloaded CTranslate2 directory).
    path : str or None
        Model path when ``source="local"``, otherwise unused.

    Returns
    -------
    str
        Path to CTranslate2 model directory.
    """
    if source == "local":
        if not path or not os.path.isdir(path):
            raise FileNotFoundError(
                f"source=local but path not found: {path}"
            )
        return path

    if source == "modelscope":
        return _prepare_whisper_modelscope(model_size)

    # huggingface → let faster-whisper handle it automatically
    return model_size


def _prepare_whisper_modelscope(model_size, repo=None):
    """Download whisper from ModelScope, convert to CTranslate2, cache to models/."""
    from modelscope import snapshot_download

    if not repo:
        repo = f"openai-mirror/whisper-{model_size}"

    ct2_path = model_dir(f"whisper-{model_size}")

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


# ── SenseVoice ───────────────────────────────────────────────────────

def resolve_sensevoice_model():
    """Download SenseVoice ONNX model from HF mirror, cache to models/sensevoice/.

    Returns
    -------
    tuple[str, str]
        ``(model.onnx path, tokens.txt path)``
    """
    from urllib.request import urlretrieve

    mdir = model_dir("sensevoice")

    model_file = os.path.join(mdir, "model.onnx")
    tokens_file = os.path.join(mdir, "tokens.txt")

    if os.path.isfile(model_file) and os.path.isfile(tokens_file):
        return model_file, tokens_file

    base = (
        "https://hf-mirror.com/csukuangfj/"
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main"
    )

    print("Downloading SenseVoice ONNX model …")
    urlretrieve(f"{base}/model.onnx", model_file)
    urlretrieve(f"{base}/tokens.txt", tokens_file)

    print(f"SenseVoice model cached at {mdir}")
    return model_file, tokens_file
