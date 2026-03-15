from flask import Blueprint, request, jsonify, send_file
import base64
import json
import io

from services.runpod_service import send_diarize, send_extract_speaker

diarization_bp = Blueprint("diarization", __name__)


@diarization_bp.route("/diarize", methods=["POST"])
def diarize():
    if "file" not in request.files:
        return jsonify({"error": "لم يتم إرسال ملف"}), 400

    audio_b64 = base64.b64encode(
        request.files["file"].read()
    ).decode()

    response = send_diarize({"audio_base64": audio_b64})
    return jsonify(response.json()), response.status_code


@diarization_bp.route("/extract", methods=["POST"])
def extract():
    if "file" not in request.files:
        return jsonify({"error": "لم يتم إرسال ملف"}), 400

    speaker_id = request.form.get("speaker_id")
    segments   = request.form.get("segments")

    if not speaker_id or not segments:
        return jsonify({"error": "speaker_id و segments مطلوبان"}), 400

    audio_b64 = base64.b64encode(
        request.files["file"].read()
    ).decode()

    response = send_extract_speaker({
        "audio_base64": audio_b64,
        "speaker_id":   speaker_id,
        "segments":     json.loads(segments)
    })

    result = response.json()

    if result.get("status") != "success":
        return jsonify({"error": result.get("message")}), 500

    # إرجاع ملف WAV مباشرة للـ Flutter
    audio_bytes = base64.b64decode(result["audio_base64"])
    return send_file(
        io.BytesIO(audio_bytes),
        mimetype="audio/wav",
        as_attachment=True,
        download_name=f"{speaker_id}_extracted.wav"
    )
