"""FaceONNX schema."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

import numpy as np
from pydantic import BaseModel, Field, field_validator


class FaceONNXResultSchema(BaseModel):
    """FaceONNX detection result schema."""

    face: np.ndarray = Field(...)
    boxes: List[int] = Field([], example=[0, 0, 100, 100])
    scores: float = Field(..., example=0.99)

    @field_validator("scores")
    def scores_validator(cls, v):
        """Round scores to 2 decimal places."""
        return round(v, 2)

    class Config:
        arbitrary_types_allowed = True
