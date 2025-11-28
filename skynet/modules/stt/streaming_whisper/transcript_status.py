"""
Transcript Status Service

Handles:
1. Notifying external API when transcript starts/finishes
2. Uploading transcript files to MinIO/S3
"""

from pathlib import Path
from typing import Optional

from skynet.env import (
    transcript_status_api_url,
    transcript_status_api_key,
    transcript_status_api_enabled,
    streaming_whisper_output_dir,
    whisper_model_name,
    skynet_s3_bucket,
    use_s3,
)
from skynet.logs import get_logger
from skynet import http_client

log = get_logger(__name__)

# Store transcript status IDs per meeting
_transcript_status_ids: dict[str, str] = {}


async def start_transcript(session_id: str) -> Optional[str]:
    """
    Notify external API that transcript has started.
    Returns the transcript status ID from API response.
    """
    if not transcript_status_api_enabled or not transcript_status_api_url:
        log.debug(f'Transcript status API disabled, skipping start_transcript for {session_id}')
        return None

    try:
        url = f'{transcript_status_api_url}/api/TranscriptStatus/StartTranscript'
        headers = {'X-API-Key': transcript_status_api_key} if transcript_status_api_key else {}
        payload = {
            'SessionId': session_id,
            'RecordPath': skynet_s3_bucket or '',
            'TranscriptionModel': whisper_model_name or 'whisper',
            'TranscriptPath': f'{session_id}/transcript/{session_id}.srt'
        }

        log.info(f'Calling StartTranscript API for session {session_id}')
        response = await http_client.post(url, json=payload, headers=headers)

        transcript_id = response.get('Id') if isinstance(response, dict) else None
        if transcript_id:
            _transcript_status_ids[session_id] = transcript_id
            log.info(f'StartTranscript successful for {session_id}, got ID: {transcript_id}')
        else:
            log.warning(f'StartTranscript response missing Id for {session_id}: {response}')

        return transcript_id

    except Exception as e:
        log.error(f'Failed to call StartTranscript API for {session_id}: {e}')
        return None


async def finish_transcript(session_id: str, success: bool = True) -> bool:
    """
    Notify external API that transcript has finished.
    """
    if not transcript_status_api_enabled or not transcript_status_api_url:
        log.debug(f'Transcript status API disabled, skipping finish_transcript for {session_id}')
        return True

    transcript_id = _transcript_status_ids.get(session_id)
    if not transcript_id:
        log.warning(f'No transcript status ID found for session {session_id}, cannot finish')
        return False

    try:
        url = f'{transcript_status_api_url}/api/TranscriptStatus/FinishTranscript'
        headers = {'X-API-Key': transcript_status_api_key} if transcript_status_api_key else {}
        payload = {
            'Id': transcript_id,
            'IsSuccess': success
        }

        log.info(f'Calling FinishTranscript API for session {session_id}, success={success}')
        await http_client.post(url, json=payload, headers=headers)

        # Clean up stored ID
        _transcript_status_ids.pop(session_id, None)
        log.info(f'FinishTranscript successful for {session_id}')
        return True

    except Exception as e:
        log.error(f'Failed to call FinishTranscript API for {session_id}: {e}')
        _transcript_status_ids.pop(session_id, None)
        return False


async def upload_transcripts_to_minio(session_id: str) -> bool:
    """
    Upload transcript files to MinIO/S3.
    
    Path structure:
    - {session_id}/raw_transcript/{session_id}.jsonl
    - {session_id}/transcript/{session_id}.srt
    """
    if not use_s3:
        log.debug(f'S3 not configured, skipping MinIO upload for {session_id}')
        return True

    try:
        from skynet.modules.ttt.s3 import S3
        s3 = S3()

        meeting_dir = Path(streaming_whisper_output_dir) / session_id
        if not meeting_dir.exists():
            log.warning(f'Meeting directory not found: {meeting_dir}')
            return False

        success = True

        # Upload JSONL (raw transcript)
        jsonl_file = meeting_dir / f'{session_id}.jsonl'
        if jsonl_file.exists():
            s3_key = f'{session_id}/raw_transcript/{session_id}.jsonl'
            if not await s3.upload_file_with_key(str(jsonl_file), s3_key):
                success = False
        else:
            log.debug(f'JSONL file not found: {jsonl_file}')

        # Upload SRT (compiled transcript)
        srt_file = meeting_dir / f'{session_id}.srt'
        if srt_file.exists():
            s3_key = f'{session_id}/transcript/{session_id}.srt'
            if not await s3.upload_file_with_key(str(srt_file), s3_key):
                success = False
        else:
            log.debug(f'SRT file not found: {srt_file}')

        if success:
            log.info(f'Successfully uploaded transcripts to MinIO for {session_id}')
        else:
            log.warning(f'Some transcript uploads failed for {session_id}')

        return success

    except Exception as e:
        log.error(f'Failed to upload transcripts to MinIO for {session_id}: {e}')
        return False


def get_transcript_status_id(session_id: str) -> Optional[str]:
    """Get the stored transcript status ID for a session."""
    return _transcript_status_ids.get(session_id)


def clear_transcript_status_id(session_id: str):
    """Clear the stored transcript status ID for a session."""
    _transcript_status_ids.pop(session_id, None)

