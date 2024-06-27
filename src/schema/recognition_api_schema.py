import rootutils

ROOT = rootutils.autosetup()

from typing import List, Union

from fastapi import UploadFile
from pydantic import BaseModel, Field, field_validator

from src.schema.enums_schema import DetectionModel, RecognitionModel


class BaseAPIResponse(BaseModel):
    timestamp: str = Field(..., description="Response timestamp")
    status: str = Field(..., description="Response status")
    message: str = Field(..., description="Response message")
    data: dict = Field({}, description="Response data")


# [GET]/api/face
class FaceEmbeddingData(BaseModel):
    id: int = Field(..., description="ID of the face embedding")
    profile_id: int = Field(..., description="ID of the profile")
    name: str = Field(..., description="Name of the person")
    recognition_model: str = Field(..., description="Recognition model used")
    detection_model: str = Field(..., description="Detection model used")


class GetAllFaceEmbeddingsResponse(BaseAPIResponse):
    data: List[FaceEmbeddingData] = Field([], description="List of face embeddings")


# [POST]/api/face/register
class PostRegisterFaceRequest(BaseModel):
    name: str = Field(..., description="Name of the person")
    detection_model: DetectionModel = Field(..., description="Detection model used")
    recognition_model: RecognitionModel = Field(
        ..., description="Recognition model used"
    )
    image: UploadFile = Field(..., description="Image file")


# [POST]/api/face/recognize
class PostRecognizeFaceRequest(BaseModel):
    detection_model: DetectionModel = Field(..., description="Detection model used")
    recognition_model: RecognitionModel = Field(
        ..., description="Recognition model used"
    )
    image: UploadFile = Field(..., description="Image file")


class RecognizeData(BaseModel):
    id: int = Field(..., description="ID of the face embedding")
    profile_id: int = Field(..., description="ID of the recognized profile")
    name: str = Field(..., description="Name of the recognized person")
    distance: float = Field(..., description="Distance of the recognized profile")

    @field_validator("distance")
    def distance_to_float(cls, v: float):
        """Two decimal places."""
        return round(v, 4)


class PostRecognizeFaceResponse(BaseAPIResponse):
    data: Union[RecognizeData, List[RecognizeData]] = Field(
        [], description="Recognized data"
    )
