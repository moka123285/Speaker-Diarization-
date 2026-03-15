import runpod
import base64
import tempfile
import os
import io
import torch
from pyannote.audio import Pipeline
from pydub import AudioSegment

# تحميل النموذج مرة واحدة عند بدء الـ Pod
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.environ["HF_TOKEN"]
)

if torch.cuda.is_available():
    pipeline = pipeline.to(torch.device("cuda"))


def handle_diarize(job_input: dict) -> dict:
    audio_bytes = base64.b64decode(job_input["audio_base64"])

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        diarization = pipeline(tmp_path)
    finally:
        os.unlink(tmp_path)

    speakers = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append({
            "start": round(turn.start, 3),
            "end":   round(turn.end, 3)
        })

    return {"status": "success", "speakers": speakers}


def handle_extract(job_input: dict) -> dict:
    audio_bytes = base64.b64decode(job_input["audio_base64"])
    speaker_id  = job_input["speaker_id"]
    segments    = job_input["segments"]  # list of {start, end}

    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))

    combined = AudioSegment.empty()
    for seg in segments:
        start_ms = int(seg["start"] * 1000)
        end_ms   = int(seg["end"]   * 1000)
        combined += audio[start_ms:end_ms]

    out = io.BytesIO()
    combined.export(out, format="wav")
    result_b64 = base64.b64encode(out.getvalue()).decode()

    return {
        "status":       "success",
        "speaker_id":   speaker_id,
        "audio_base64": result_b64,
        "duration_seconds": round(len(combined) / 1000, 2)
    }


def handler(job):
    job_input = job["input"]
    action    = job_input.get("action")

    if action == "diarize":
        return handle_diarize(job_input)
    elif action == "extract":
        return handle_extract(job_input)
    else:
        return {"status": "error", "message": f"unknown action: {action}"}


runpod.serverless.start({"handler": handler})
