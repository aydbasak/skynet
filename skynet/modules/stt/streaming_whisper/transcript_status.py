"""
Transcript Status Service

Handles:
1. Notifying external API when transcript starts/finishes
2. Uploading transcript files to MinIO/S3
"""

import json
from pathlib import Path
from typing import Optional

import aiohttp

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

log = get_logger(__name__)

# Store transcript status IDs per meeting
_transcript_status_ids: dict[str, str] = {}


async def start_transcript(meeting_id: str) -> Optional[str]:
    """
    Notify external API that transcript has started.
    Returns the transcript status ID from API response.
    """
    if not transcript_status_api_enabled or not transcript_status_api_url:
        log.debug(f'Transcript status API disabled, skipping start_transcript for {meeting_id}')
        return None

    try:
        url = f'{transcript_status_api_url}/api/TranscriptStatus/StartTranscript'
        headers = {'X-API-Key': transcript_status_api_key, 'Content-Type': 'application/json'} if transcript_status_api_key else {'Content-Type': 'application/json'}
        payload = {
            'SessionId': meeting_id,
            'RecordPath': skynet_s3_bucket or '',
            'TransciptionModel': whisper_model_name or 'whisper',
            'TranscriptPath': f'{meeting_id}/transcript/{meeting_id}.srt'
        }

        log.info(f'Calling StartTranscript API for meeting {meeting_id}')
        
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(url, json=payload, headers=headers) as resp:
                status = resp.status
                response_text = await resp.text()
                log.debug(f'StartTranscript response: status={status}, body={response_text}')
                
                if status >= 400:
                    log.error(f'StartTranscript API returned error {status}: {response_text}')
                    return None
                
                # Try to parse JSON response
                try:
                    response = json.loads(response_text) if response_text else {}
                    # API returns: { "data": { "id": "..." }, "success": true, ... }
                    data = response.get('data', {}) if isinstance(response, dict) else {}
                    transcript_id = data.get('id') if isinstance(data, dict) else None
                except json.JSONDecodeError:
                    log.warning(f'Could not parse JSON response: {response_text}')
                    transcript_id = None

        if transcript_id:
            _transcript_status_ids[meeting_id] = transcript_id
            log.info(f'StartTranscript successful for {meeting_id}, got ID: {transcript_id}')
        else:
            log.warning(f'StartTranscript response missing Id for {meeting_id}')

        return transcript_id

    except Exception as e:
        log.error(f'Failed to call StartTranscript API for {meeting_id}: {e}')
        return None


async def finish_transcript(meeting_id: str, success: bool = True) -> bool:
    """
    Notify external API that transcript has finished.
    """
    if not transcript_status_api_enabled or not transcript_status_api_url:
        log.debug(f'Transcript status API disabled, skipping finish_transcript for {meeting_id}')
        return True

    transcript_id = _transcript_status_ids.get(meeting_id)
    if not transcript_id:
        log.warning(f'No transcript status ID found for meeting {meeting_id}, cannot finish')
        return False

    try:
        url = f'{transcript_status_api_url}/api/TranscriptStatus/FinishTranscript'
        headers = {'X-API-Key': transcript_status_api_key, 'Content-Type': 'application/json'} if transcript_status_api_key else {'Content-Type': 'application/json'}
        payload = {
            'Id': transcript_id,
            'IsSuccess': success
        }

        log.info(f'Calling FinishTranscript API for meeting {meeting_id}, success={success}')
        
        async with aiohttp.ClientSession() as http_session:
            async with http_session.post(url, json=payload, headers=headers) as resp:
                status = resp.status
                response_text = await resp.text()
                log.debug(f'FinishTranscript response: status={status}, body={response_text}')
                
                if status >= 400:
                    log.error(f'FinishTranscript API returned error {status}: {response_text}')
                    _transcript_status_ids.pop(meeting_id, None)
                    return False

        # Clean up stored ID
        _transcript_status_ids.pop(meeting_id, None)
        log.info(f'FinishTranscript successful for {meeting_id}')
        return True

    except Exception as e:
        log.error(f'Failed to call FinishTranscript API for {meeting_id}: {e}')
        _transcript_status_ids.pop(meeting_id, None)
        return False


async def upload_transcripts_to_minio(meeting_id: str) -> bool:
    """
    Upload transcript files to MinIO/S3.
    
    Path structure:
    - {meeting_id}/raw_transcript/{meeting_id}.jsonl
    - {meeting_id}/transcript/{meeting_id}.srt
    """
    if not use_s3:
        log.debug(f'S3 not configured, skipping MinIO upload for {meeting_id}')
        return True

    try:
        from skynet.modules.ttt.s3 import S3
        s3 = S3()

        meeting_dir = Path(streaming_whisper_output_dir) / meeting_id
        if not meeting_dir.exists():
            log.warning(f'Meeting directory not found: {meeting_dir}')
            return False

        upload_success = True

        # Upload JSONL (raw transcript)
        jsonl_file = meeting_dir / f'{meeting_id}.jsonl'
        if jsonl_file.exists():
            s3_key = f'{meeting_id}/raw_transcript/{meeting_id}.jsonl'
            if not await s3.upload_file_with_key(str(jsonl_file), s3_key):
                upload_success = False
        else:
            log.debug(f'JSONL file not found: {jsonl_file}')

        # Upload SRT (compiled transcript)
        srt_file = meeting_dir / f'{meeting_id}.srt'
        if srt_file.exists():
            s3_key = f'{meeting_id}/transcript/{meeting_id}.srt'
            if not await s3.upload_file_with_key(str(srt_file), s3_key):
                upload_success = False
        else:
            log.debug(f'SRT file not found: {srt_file}')

        if upload_success:
            log.info(f'Successfully uploaded transcripts to MinIO for {meeting_id}')
        else:
            log.warning(f'Some transcript uploads failed for {meeting_id}')

        return upload_success

    except Exception as e:
        log.error(f'Failed to upload transcripts to MinIO for {meeting_id}: {e}')
        return False


def get_transcript_status_id(meeting_id: str) -> Optional[str]:
    """Get the stored transcript status ID for a meeting."""
    return _transcript_status_ids.get(meeting_id)


def clear_transcript_status_id(meeting_id: str):
    """Clear the stored transcript status ID for a meeting."""
    _transcript_status_ids.pop(meeting_id, None)
