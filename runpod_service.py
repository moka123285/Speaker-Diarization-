def send_diarize(payload: dict, timeout: int = 300) -> requests.Response:
    """
    إرسال طلب Diarization إلى RunPod Serverless.
    تتقبل action='diarize'
    """
    runpod_payload = {
        "input": {
            "action": "diarize",
            **payload
        }
    }

    response = requests.post(
        RUNPOD_RUN_URL,
        json=runpod_payload,
        headers=HEADERS,
        timeout=30
    )
    response.raise_for_status()
    runpod_data = response.json()
    normalized = _normalize_runpod_response(runpod_data)
    return _make_response(normalized, response.status_code)


def send_extract_speaker(payload: dict, timeout: int = 300) -> requests.Response:
    """
    إرسال طلب استخراج مقاطع متحدث محدد إلى RunPod Serverless.
    تتقبل action='extract'
    """
    runpod_payload = {
        "input": {
            "action": "extract",
            **payload
        }
    }

    response = requests.post(
        RUNPOD_RUN_URL,
        json=runpod_payload,
        headers=HEADERS,
        timeout=30
    )
    response.raise_for_status()
    runpod_data = response.json()
    normalized = _normalize_runpod_response(runpod_data)
    return _make_response(normalized, response.status_code)
