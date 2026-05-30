"""STT text processing.

In v1 the actual speech-to-text inference is done by the wake-word detector.
This module handles the conditional text post-processing based on state.
"""

from .wake_word import WakeWordDetector


def process_transcription(text, state):
    """Post-process the transcribed text according to *state*.

    Parameters
    ----------
    text : str
        Raw transcription from the wake-word detector.
    state : str
        ``"LISTENING"`` or ``"AWAKE"``.

    Returns
    -------
    str
        Processed command text to send to Hermes.
    """
    if not text.strip():
        return ""

    if state == "LISTENING":
        return WakeWordDetector.strip_wake_word(text)
    # AWAKE → pass through verbatim
    return text
