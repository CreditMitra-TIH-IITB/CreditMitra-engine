from pydantic import BaseModel


class Transaction(BaseModel):
    date: str
    particulars: str
    deposits: str
    withdrawals: str
    balance: str
    payee: str | None = None
    payee_type: str | None = None  # "person" or "merchant"
    payee_confidence: float | None = None  # 0.0-1.0


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    transactions: list[Transaction] | None = None
    error: str | None = None
