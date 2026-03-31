from pydantic import BaseModel
from typing import Optional

class Observation(BaseModel):
    data_preview: str
    missing_values: int
    duplicates: int

class Action(BaseModel):
    action_type: str
    column: Optional[str] = None

class Reward(BaseModel):
    score: float
    message: str