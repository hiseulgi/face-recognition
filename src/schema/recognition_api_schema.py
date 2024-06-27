"""Schema for the recognition API."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List, Union

from fastapi import UploadFile
from pydantic import BaseModel, Field, field_validator

from src.schema.enums_schema import DetectionModel, RecognitionModel


class BaseAPIResponse(BaseModel):
    """Base schema for API response."""

    timestamp: str = Field(..., description="Response timestamp")
    status: str = Field(..., description="Response status")
    detail: str = Field(..., description="Response detail")
    data: dict = Field({}, description="Response data")


# [GET]/api/face
class FaceEmbeddingData(BaseModel):
    """Face embedding data schema."""

    id: int = Field(..., description="ID of the face embedding")
    profile_id: int = Field(..., description="ID of the profile")
    name: str = Field(..., description="Name of the person")
    recognition_model: str = Field(..., description="Recognition model used")
    detection_model: str = Field(..., description="Detection model used")


class GetAllFaceEmbeddingsResponse(BaseAPIResponse):
    """Get all face embeddings response schema."""

    data: List[FaceEmbeddingData] = Field([], description="List of face embeddings")


# [POST]/api/face/register
class PostRegisterFaceRequest(BaseModel):
    """Register face request schema."""

    name: str = Field(..., description="Name of the person")
    detection_model: DetectionModel = Field(..., description="Detection model used")
    recognition_model: RecognitionModel = Field(
        ..., description="Recognition model used"
    )
    image: UploadFile = Field(..., description="Image file")


# [POST]/api/face/recognize
class PostRecognizeFaceRequest(BaseModel):
    """Recognize face request schema."""

    detection_model: DetectionModel = Field(..., description="Detection model used")
    recognition_model: RecognitionModel = Field(
        ..., description="Recognition model used"
    )
    image: UploadFile = Field(..., description="Image file")


class RecognizeData(BaseModel):
    """Recognize data schema."""

    id: int = Field(..., description="ID of the face embedding")
    profile_id: int = Field(..., description="ID of the recognized profile")
    name: str = Field(..., description="Name of the recognized person")
    distance: float = Field(..., description="Distance of the recognized profile")

    @field_validator("distance")
    def distance_to_float(cls, v: float):
        """Two decimal places."""
        return round(v, 4)


class PostRecognizeFaceResponse(BaseAPIResponse):
    """Recognize face response schema."""

    data: Union[RecognizeData, List[RecognizeData]] = Field(
        [], description="Recognized data"
    )
