from faster_whisper import WhisperModel


def my_transcription_task():
    model = WhisperModel("tiny", device="cuda", compute_type="float16")
    segments, info = model.transcribe("california.mp3", beam_size=5)
    result = list(segments)
    print(f"Detected language: {info.language}")


if __name__ == "__main__":
    my_transcription_task()

# py-spy record -o profile.svg -- python pyspy_analysis.py
