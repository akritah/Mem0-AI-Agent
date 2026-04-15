"""Speech-to-text helpers for local and cloud transcription backends."""

import os
import traceback
from typing import Tuple, Optional


def transcribe_audio(audio_path: str) -> Tuple[str, Optional[str]]:
    """Transcribe an audio file to text.

    Args:
        audio_path: Absolute or relative path to an audio file.

    Returns:
        A tuple containing the transcript and an optional error message.
        On success: ``(<transcript>, None)``.
        On failure: ``("", <error_message>)``.
    """

    # If GROQ_API_KEY is set → prefer Groq (faster, no GPU needed)
    if os.getenv("GROQ_API_KEY"):
        return _transcribe_groq(audio_path)

    # Try local faster-whisper first
    try:
        return _transcribe_local(audio_path)
    except Exception as e:
        # Graceful degradation: try Groq if available
        if os.getenv("GROQ_API_KEY"):
            return _transcribe_groq(audio_path)
        return "", f"Local STT failed: {str(e)}\n\nFix: Install faster-whisper (`pip install faster-whisper`) or set GROQ_API_KEY in .env"


def _transcribe_local(audio_path: str) -> Tuple[str, Optional[str]]:
    """Run local transcription with faster-whisper.

    Args:
        audio_path: Path to the audio file to transcribe.

    Returns:
        A tuple containing transcript text and optional error details.
    """
    from faster_whisper import WhisperModel

    # Use 'base' model for speed; change to 'small' or 'medium' for accuracy
    model_size = os.getenv("WHISPER_MODEL_SIZE", "base")

    print(f"[STT] Loading faster-whisper ({model_size})...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(audio_path, beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments)

    if not text.strip():
        return "", "STT returned empty transcription. Check audio quality."

    print(f"[STT] Transcribed: {text[:100]}")
    return text.strip(), None


def _transcribe_groq(audio_path: str) -> Tuple[str, Optional[str]]:
    """Run transcription using Groq Whisper API.

    Args:
        audio_path: Path to the audio file to transcribe.

    Returns:
        A tuple containing transcript text and optional error details.
    """
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="text"
            )

        text = transcription if isinstance(transcription, str) else transcription.text
        print(f"[STT-Groq] Transcribed: {text[:100]}")
        return text.strip(), None

    except Exception as e:
        return "", f"Groq STT failed: {str(e)}"
