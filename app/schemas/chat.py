"""
Pydantic request / response schemas for the chat API.
"""
from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: Annotated[
        str,
        Field(min_length=1, max_length=2000, description="用户的问题"),
    ]
    session_id: Annotated[
        str | None,
        Field(
            default=None,
            max_length=64,
            description=(
                "会话 ID。首次请求留空，服务端会生成并在响应中返回。"
                "后续请求传入该值以延续同一会话。"
            ),
        ),
    ] = None
    source: str = Field(
        default="api",
        max_length=32,
        description="渠道标签，如 wechat / wecom / api",
    )


class ChatResponse(BaseModel):
    answer:     str
    session_id: str   = Field(description="会话 ID，客户端应持久化并在后续请求中回传")
    latency_s:  float = Field(description="服务端处理耗时（秒）")


class HealthResponse(BaseModel):
    status:   str
    pipeline: str
