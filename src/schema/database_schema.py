import rootutils

ROOT = rootutils.autosetup()

from pgvector.sqlalchemy import Vector
from pydantic import BaseModel


class ProfileBase(BaseModel):
    name: str


class ProfileCreate(ProfileBase):
    pass


class Profile(ProfileBase):
    id: int
    created_at: str

    class Config:
        orm_mode = True


class FaceEmbeddingBase(BaseModel):
    embedding: list
    detection_model: str
    recognition_model: str


class FaceEmbeddingCreate(FaceEmbeddingBase):
    profile_id: int


class FaceEmbedding(FaceEmbeddingBase):
    id: int
    profile_id: int
    created_at: str

    class Config:
        orm_mode = True
