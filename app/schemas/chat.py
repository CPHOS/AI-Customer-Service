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
                "会话 ID（兼容字段，迁移期可选）。"
                "服务端优先使用 HttpOnly Cookie；该字段后续将下线。"
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
    session_id: str   = Field(description="会话 ID（兼容字段，后续将废弃；以 Cookie 为准）")
    latency_s:  float = Field(description="服务端处理耗时（秒）")


class HealthResponse(BaseModel):
    status:   str
    pipeline: str
