from pydantic import BaseModel, Field


class NodeStatus(BaseModel):
    name: str
    status: str = Field("online", description="Status: online, degraded, offline, ready")
    host: str
    latency_ms: int | None = None
    details: str | None = None
    model: str | None = None
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
    api_key: str | None = None


class ConnectionTestResponse(BaseModel):
    success: bool
    component: str
    url: str
    latency_ms: int
    details: str


class NodeConfigUpdateRequest(BaseModel):
    ollama_host: str | None = None
    ollama_model: str | None = None
    merchant_url: str | None = None
    merchant_api_key: str | None = None
