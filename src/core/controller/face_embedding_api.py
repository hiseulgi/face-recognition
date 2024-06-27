from typing import Union

import rootutils

ROOT = rootutils.autosetup()

from sqlalchemy.orm import Session
from sqlalchemy.sql import select

from src.database.models import FaceEmbedding
from src.schema.database_schema import FaceEmbeddingCreate
from src.schema.enums_schema import DistanceMetric


class FaceEmbeddingAPI:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_face_embedding(
        self, face_embedding: FaceEmbeddingCreate
    ) -> FaceEmbedding:
        db_face_embedding = FaceEmbedding(
            profile_id=face_embedding.profile_id,
            embedding=face_embedding.embedding,
            detection_model=face_embedding.detection_model,
            recognition_model=face_embedding.recognition_model,
        )
        self.db.add(db_face_embedding)
        self.db.commit()
        self.db.refresh(db_face_embedding)
        return db_face_embedding

    def get_face_embedding_by_id(self, face_embedding_id: int) -> FaceEmbedding:
        return self.db.query(FaceEmbedding).get(face_embedding_id)

    def get_all_face_embeddings(self) -> list[FaceEmbedding]:
        return self.db.query(FaceEmbedding).all()

    def get_face_embeddings_by_recognizer(
        self, recognizer_model: str
    ) -> list[FaceEmbedding]:
        return (
            self.db.query(FaceEmbedding)
            .filter(FaceEmbedding.recognition_model == recognizer_model)
            .all()
        )

    def find_nearest_face_embedding(
        self, embedding: list, recognizer_model: str, distance_metric: str
    ) -> FaceEmbedding:
        if distance_metric == DistanceMetric.COSINE:
            nearest_embedding = (
                self.db.query(FaceEmbedding)
                .filter(FaceEmbedding.recognition_model == recognizer_model)
                .order_by(FaceEmbedding.embedding.op("<=>")(embedding))
                .first()
            )
        else:
            raise NotImplementedError(
                f"Distance metric {distance_metric} is not implemented."
            )
        return nearest_embedding

    def delete_face_embedding(self, face_embedding_id: int) -> FaceEmbedding:
        face_embedding = self.get_face_embedding_by_id(face_embedding_id)
        self.db.delete(face_embedding)
        self.db.commit()
        return face_embedding
