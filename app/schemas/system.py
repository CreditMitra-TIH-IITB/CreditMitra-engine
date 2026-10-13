from typing import Any, Optional
from pydantic import BaseModel, Field


class NodeStatus(BaseModel):
    name: str
    status: str = Field("online", description="Status: online, degraded, offline, ready")
    host: str
    latency_ms: Optional[int] = None
    details: Optional[str] = None
    model: Optional[str] = None
    is_active: bool = True
    capabilities: list[str] = Field(default_factory=list)


class SystemNodesResponse(BaseModel):
    engine: NodeStatus
    ollama: NodeStatus
    merchant_server: NodeStatus
    docling: NodeStatus
    classifier: NodeStatus
    timestamp: float
    overall_health: str = "healthy"


class ConnectionTestRequest(BaseModel):
    component: str = Field(..., description="engine, ollama, or merchant")
    url: str
    api_key: Optional[str] = None


class ConnectionTestResponse(BaseModel):
    success: bool
    component: str
    url: str
    latency_ms: int
    details: str


class NodeConfigUpdateRequest(BaseModel):
    ollama_host: Optional[str] = None
    ollama_model: Optional[str] = None
    merchant_url: Optional[str] = None
    merchant_api_key: Optional[str] = None
