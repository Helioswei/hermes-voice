import os
import re

from faster_whisper import WhisperModel


class WakeWordDetector:
    """faster-whisper wrapper with "小九" wake-word detection.

    Three model sources:

    * ``huggingface`` — download from HuggingFace Hub (default)
    * ``modelscope`` — download from ModelScope via ``snapshot_download``,
      then auto-convert to CTranslate2 format (only on first run)
    * ``local`` — use a pre-downloaded CTranslate2 model directory
    """

    def __init__(self, model_size="tiny", model_source="huggingface",
                 modelscope_repo=None, model_path=None,
                 device="cpu", compute_type="int8"):
        resolved = self._resolve_model(model_size, model_source,
                                       modelscope_repo, model_path)
        self.model = WhisperModel(resolved, device=device,
                                  compute_type=compute_type)

    # ------------------------------------------------------------------
    # Model resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model(model_size, model_source, modelscope_repo, model_path):
        if model_source == "local":
            if not model_path or not os.path.isdir(model_path):
                raise FileNotFoundError(
                    f"model_source=local but model_path not found: {model_path}"
                )
            return model_path

        if model_source == "modelscope":
            return WakeWordDetector._prepare_modelscope(model_size,
                                                        modelscope_repo)

        # huggingface → let faster-whisper handle it
        return model_size

    @staticmethod
    def _prepare_modelscope(model_size, repo=None):
        """Download whisper model from ModelScope and convert to CTranslate2.

        The ``openai-mirror/whisper-{size}`` repos on ModelScope contain the
        original OpenAI Whisper (PyTorch format). ``faster-whisper`` needs
        CTranslate2 format, so we convert it on first download.

        Both the download and the converted output are cached so subsequent
        runs are instant.
        """
        from modelscope import snapshot_download

        if not repo:
            repo = f"openai-mirror/whisper-{model_size}"

        model_name = repo.replace("/", "_")
        cache_root = os.path.join(
            os.path.expanduser("~"), ".cache", "hermes-voice"
        )
        os.makedirs(cache_root, exist_ok=True)

        # Path for the converted CTranslate2 model
        ct2_path = os.path.join(cache_root, f"{model_name}-ct2")

        # Already converted → return immediately
        if os.path.isdir(ct2_path) and os.path.isfile(
                os.path.join(ct2_path, "model.bin")):
            return ct2_path

        # Download original whisper model from ModelScope
        # Only download PyTorch format, skip flax/tensorflow (unused)
        raw_path = snapshot_download(
            repo,
            ignore_file_pattern=["*.msgpack", "*.h5"],
        )

        # Convert to CTranslate2 format
        from ctranslate2.converters import TransformersConverter

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

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(self, audio):
        if audio is None or len(audio) == 0:
            return ""

        segments, _ = self.model.transcribe(
            audio, language="zh",
            initial_prompt="你好，请问今天天气怎么样",
            beam_size=1, vad_filter=False,
        )
        text = "".join(seg.text for seg in segments).strip()
        return text

    # Characters whisper tiny often confuses with "九" (all pronounced jiǔ)
    _WAKE_HOMOPHONES = {"九", "酒", "就", "久", "舅", "救", "旧"}

    @staticmethod
    def contains_wake_word(text):
        if "小九" in text:
            return True
        idx = text.find("小")
        if idx >= 0 and idx + 1 < len(text):
            return text[idx + 1] in WakeWordDetector._WAKE_HOMOPHONES
        return False

    @staticmethod
    def strip_wake_word(text):
        stripped = re.sub(r"^小九[\s,，]*", "", text).strip()
        if stripped != text:
            return stripped
        idx = text.find("小")
        if idx >= 0 and idx + 1 < len(text) and \
           text[idx + 1] in WakeWordDetector._WAKE_HOMOPHONES:
            return text[idx + 2:].strip().lstrip("，,").strip()
        return text.strip()


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
