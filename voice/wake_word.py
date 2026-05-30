import os

from faster_whisper import WhisperModel


class WakeWordDetector:
    """faster-whisper wrapper for Chinese STT.

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

        Model is stored in ``models/whisper/`` under the project root.
        """
        from modelscope import snapshot_download

        if not repo:
            repo = f"openai-mirror/whisper-{model_size}"

        project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        ct2_path = os.path.join(project_root, "models", "whisper")
        os.makedirs(ct2_path, exist_ok=True)

        if os.path.isfile(os.path.join(ct2_path, "model.bin")):
            return ct2_path

        raw_path = snapshot_download(
            repo,
            ignore_file_pattern=["*.msgpack", "*.h5"],
        )

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
            beam_size=5, vad_filter=True,
        )
        text = "".join(seg.text for seg in segments).strip()
        return text


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
