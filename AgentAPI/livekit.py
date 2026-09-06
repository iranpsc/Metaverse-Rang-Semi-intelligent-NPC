from __future__ import annotations

import datetime as dt
import json

import aiohttp
from django.conf import settings
from livekit import api


def create_participant_token(*, room_name: str, identity: str, session_id: str) -> str:
    """Create a short-lived, room-scoped Unity participant token."""
    grants = api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=False,
        can_publish_sources=["microphone"],
        can_update_own_metadata=False,
    )
    return (
        api.AccessToken(settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
        .with_identity(identity)
        .with_name("Unity client")
        .with_metadata(json.dumps({"session_id": session_id}, separators=(",", ":")))
        .with_grants(grants)
        .with_ttl(dt.timedelta(seconds=settings.LIVEKIT_TOKEN_TTL_SECONDS))
        .to_jwt()
    )


async def provision_room_and_agent(*, room_name: str, metadata: dict) -> str:
    """Create the isolated room and explicitly dispatch the named agent worker."""
    timeout = aiohttp.ClientTimeout(total=settings.LIVEKIT_API_TIMEOUT_SECONDS)
    async with api.LiveKitAPI(
        settings.LIVEKIT_API_URL,
        settings.LIVEKIT_API_KEY,
        settings.LIVEKIT_API_SECRET,
        timeout=timeout,
    ) as client:
        await client.room.create_room(
            api.CreateRoomRequest(
                name=room_name,
                empty_timeout=settings.LIVEKIT_ROOM_EMPTY_TIMEOUT_SECONDS,
                departure_timeout=settings.LIVEKIT_ROOM_DEPARTURE_TIMEOUT_SECONDS,
                max_participants=2,
                metadata=json.dumps(
                    {"session_id": metadata["session_id"]}, separators=(",", ":")
                ),
            )
        )
        try:
            dispatch = await client.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    room=room_name,
                    agent_name=settings.LIVEKIT_AGENT_NAME,
                    metadata=json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
                )
            )
        except Exception:
            try:
                await client.room.delete_room(api.DeleteRoomRequest(room=room_name))
            except Exception:
                pass
            raise
    return dispatch.id


async def delete_room(room_name: str) -> None:
    timeout = aiohttp.ClientTimeout(total=settings.LIVEKIT_API_TIMEOUT_SECONDS)
    async with api.LiveKitAPI(
        settings.LIVEKIT_API_URL,
        settings.LIVEKIT_API_KEY,
        settings.LIVEKIT_API_SECRET,
        timeout=timeout,
    ) as client:
        await client.room.delete_room(api.DeleteRoomRequest(room=room_name))

