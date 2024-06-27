"""Database models table schema."""

import rootutils

ROOT = rootutils.autosetup()

from pgvector.sqlalchemy import Vector
from sqlalchemy import TIMESTAMP, Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from src.database.database import Base


class Profile(Base):
    """Profile table schema."""

    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    created_at = Column(TIMESTAMP, nullable=False, server_default="now()")

    face_embeddings = relationship("FaceEmbedding", back_populates="profile")


class FaceEmbedding(Base):
    """FaceEmbedding table schema."""

    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    profile_id = Column(Integer, ForeignKey("profiles.id"))
    embedding = Column(Vector(512))
    detection_model = Column(String)
    recognition_model = Column(String)
    created_at = Column(TIMESTAMP, nullable=False, server_default="now()")

    profile = relationship("Profile", back_populates="face_embeddings")
