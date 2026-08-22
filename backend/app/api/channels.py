"""Channel and agent REST endpoints.

The WebSocket carries the live conversation; these endpoints exist so the UI
can render before the socket connects, and so history survives a reload.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from app.agents.registry import get_agent_registry
from app.api.deps import require_api_key
from app.core.models import Message
from app.services.message_service import get_message_service

router = APIRouter(prefix="/api", tags=["channels"], dependencies=[Depends(require_api_key)])


@router.get("/channels")
async def list_channels() -> dict[str, Any]:
    service = get_message_service()
    registry = get_agent_registry()

    channels = []
    for channel in service.channels:
        payload = channel.model_dump(mode="json")
        payload["agents"] = service.agents_for_channel(channel.id, registry.ids)
        payload["message_count"] = len(service.history(channel.id))
        channels.append(payload)

    return {"channels": channels}


@router.get("/channels/{channel_id}/messages")
async def channel_messages(
    channel_id: str,
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    service = get_message_service()
    service.get_channel(channel_id)
    messages: list[Message] = service.history(channel_id, limit=limit)
    return {
        "channel_id": channel_id,
        "messages": [m.model_dump(mode="json") for m in messages],
    }


@router.get("/agents")
async def list_agents() -> dict[str, Any]:
    return {"agents": get_agent_registry().public()}


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str) -> dict[str, Any]:
    definition = get_agent_registry().get(agent_id)
    payload = definition.public()
    # The full prompt is useful when tuning an agent and contains no secrets.
    payload["system_prompt"] = definition.full_system_prompt()
    payload["auto_retrieve"] = definition.auto_retrieve
    return payload
