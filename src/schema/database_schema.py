"""Database schema for the application."""

import rootutils

ROOT = rootutils.autosetup()

from pgvector.sqlalchemy import Vector
from pydantic import BaseModel


# Profile
class ProfileBase(BaseModel):
    """Base schema for Profile."""

    name: str


class ProfileCreate(ProfileBase):
    """Create schema for Profile."""

    pass


class Profile(ProfileBase):
    """Schema for Profile ORM."""

    id: int
    created_at: str

    class Config:
        orm_mode = True


# FaceEmbedding
class FaceEmbeddingBase(BaseModel):
    """Base schema for FaceEmbedding."""

    embedding: list
    detection_model: str
    recognition_model: str


class FaceEmbeddingCreate(FaceEmbeddingBase):
    """Create schema for FaceEmbedding."""

    profile_id: int


class FaceEmbedding(FaceEmbeddingBase):
    """Schema for FaceEmbedding ORM."""

    id: int
    profile_id: int
    created_at: str

    class Config:
        orm_mode = True
