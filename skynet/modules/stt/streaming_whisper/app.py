from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from skynet.logs import get_logger
from skynet.modules.stt.streaming_whisper.connection_manager import ConnectionManager
from skynet.modules.stt.streaming_whisper.utils import utils

log = get_logger(__name__)

ws_connection_manager = ConnectionManager()
app = FastAPI()


@app.websocket('/ws/{meeting_id}')
async def websocket_endpoint(
    websocket: WebSocket,
    meeting_id: str,
    auth_token: str | None = None,
    roomname: str | None = None,
):
    # Log incoming parameters for debugging
    log.info(f'WebSocket params: meeting_id={meeting_id}, roomname={roomname}')
    
    # If a Jitsi room name is provided, use it as the effective meeting identifier
    effective_id = roomname if roomname else meeting_id
    log.info(f'Using effective_id: {effective_id}')
    
    connected = await ws_connection_manager.connect(websocket, effective_id, auth_token)
    if not connected:
        return

    try:
        while True:
            try:
                chunk = await websocket.receive_bytes()
            except Exception as err:
                log.warning(f'Expected bytes, received something else, disconnecting {effective_id}. Error: \n{err}')
                await ws_connection_manager.graceful_disconnect(effective_id, success=False)
                break
            if len(chunk) == 1 and ord(b'' + chunk) == 0:
                log.info(f'Received disconnect message for {effective_id}')
                await ws_connection_manager.graceful_disconnect(effective_id, success=True)
                break
            await ws_connection_manager.process(effective_id, chunk, utils.now())
    except WebSocketDisconnect:
        log.info(f'Meeting {effective_id} has ended (WebSocket disconnect)')
        await ws_connection_manager.graceful_disconnect(effective_id, success=True)
    except Exception as e:
        log.error(f'Unexpected error in meeting {effective_id}: {e}')
        await ws_connection_manager.graceful_disconnect(effective_id, success=False)
