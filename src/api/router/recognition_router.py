import rootutils

from database.database import Base, Database

ROOT = rootutils.autosetup()

import time
from io import BytesIO
from typing import Any, List

import cv2
import numpy as np
from fastapi import APIRouter, Depends, Request
from omegaconf import DictConfig
from PIL import Image

from src.core.controller.face_embedding_api import FaceEmbeddingAPI
from src.core.controller.profile_api import ProfileAPI
from src.core.detector.faceonnx_core import FaceOnnxCore
from src.core.recognizer.arcface_core import ArcfaceOnnxCore
from src.schema.database_schema import FaceEmbeddingCreate, ProfileCreate
from src.schema.enums_schema import DetectionModel, RecognitionModel
from src.schema.recognition_api_schema import (
    BaseAPIResponse,
    GetAllFaceEmbeddingsResponse,
    PostRecognizeFaceRequest,
    PostRecognizeFaceResponse,
    PostRegisterFaceRequest,
    RecognizeData,
)
from src.utils.logger import get_logger
from src.utils.math import find_cosine_distance

log = get_logger()


class RecognitionRouter:

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.router = APIRouter()

        self.setup_model()
        self.setup_db()
        self.init_routes()

    def setup_db(self) -> None:
        self.db = Database(**self.cfg.database.postgres)

        self.db.connect()
        self.db.setup_pgvector()

        self.session = self.db.SessionLocal()

        self.face_embedding_api = FaceEmbeddingAPI(self.session)
        self.profile_api = ProfileAPI(self.session)

        log.info("Database tables created")

    def setup_model(self) -> None:
        self.detector = FaceOnnxCore(**self.cfg.detector.faceonnx)
        self.recognizer = ArcfaceOnnxCore(**self.cfg.recognizer.arcface)

        self.detector.setup()
        self.recognizer.setup()

    def init_routes(self) -> None:
        self.router.get(
            "/api/face",
            response_model=GetAllFaceEmbeddingsResponse,
            tags=["face"],
            summary="Get all face embeddings",
        )(self.get_all_face_embeddings)

        self.router.post(
            "/api/face/register",
            response_model=BaseAPIResponse,
            tags=["face"],
            summary="Register face",
        )(self.register_face)

        self.router.post(
            "/api/face/recognize",
            response_model=PostRecognizeFaceResponse,
            tags=["face"],
            summary="Recognize face",
        )(self.recognize_face)

        self.router.delete(
            "/api/face/{id}",
            response_model=BaseAPIResponse,
            tags=["face"],
            summary="Delete face embedding",
        )(self.delete_face_embedding)

    # [GET]/api/face
    async def get_all_face_embeddings(self) -> GetAllFaceEmbeddingsResponse:
        try:
            face_embeddings = self.face_embedding_api.get_all_face_embeddings()
            data = []
            for face_embedding in face_embeddings:
                data.append(
                    {
                        "id": face_embedding.id,
                        "profile_id": face_embedding.profile.id,
                        "name": face_embedding.profile.name,
                        "recognition_model": face_embedding.recognition_model,
                        "detection_model": face_embedding.detection_model,
                    }
                )
            return GetAllFaceEmbeddingsResponse(
                timestamp=str(time.time()),
                status="success",
                message="Face embeddings retrieved",
                data=data,
            )
        except Exception as e:
            log.error(f"Error: {e}")
            return GetAllFaceEmbeddingsResponse(
                timestamp=str(time.time()),
                status="error",
                message="Failed to retrieve face embeddings",
            )

    # [POST]/api/face/register
    async def register_face(
        self, request: Request, form: PostRegisterFaceRequest = Depends()
    ) -> BaseAPIResponse:
        try:
            profile = self.profile_api.get_profile_by_name(form.name)
            if not profile:

                profile = self.profile_api.create_profile(ProfileCreate(name=form.name))

            img_bytes = await form.image.read()
            img = await self.preprocess_img_bytes(img_bytes)

            if form.detection_model == DetectionModel.FACEONNX:
                results = self.detector.detect_face(img)

            if results is None or len(results) == 0:
                return PostRecognizeFaceResponse(
                    timestamp=str(time.time()),
                    status="error",
                    message="No face detected",
                )

            if len(results) > 1:
                return PostRecognizeFaceResponse(
                    timestamp=str(time.time()),
                    status="error",
                    message="Multiple faces detected",
                )

            if len(results) == 1:
                face = results[0].face

            if form.recognition_model == RecognitionModel.ARCFACE:
                embedding = self.recognizer.get_embedding(face)

            if embedding:
                face_embedding = self.face_embedding_api.create_face_embedding(
                    FaceEmbeddingCreate(
                        embedding=embedding,
                        profile_id=profile.id,
                        detection_model=form.detection_model,
                        recognition_model=form.recognition_model,
                    )
                )
                return BaseAPIResponse(
                    timestamp=str(time.time()),
                    status="success",
                    message="Face registered",
                    data={"id": face_embedding.id, "profile_id": profile.id},
                )

            return BaseAPIResponse(
                timestamp=str(time.time()),
                status="error",
                message="Failed to register face",
            )

        except Exception as e:
            log.error(f"Error: {e}")
            return BaseAPIResponse(
                timestamp=str(time.time()),
                status="error",
                message="Failed to register face",
            )

    # [POST]/api/face/recognize
    async def recognize_face(
        self, request: Request, form: PostRecognizeFaceRequest = Depends()
    ) -> PostRecognizeFaceResponse:
        try:
            img_bytes = await form.image.read()
            img = await self.preprocess_img_bytes(img_bytes)

            if form.detection_model == DetectionModel.FACEONNX:
                results = self.detector.detect_face(img)

            if results is None or len(results) == 0:
                return PostRecognizeFaceResponse(
                    timestamp=str(time.time()),
                    status="error",
                    message="No face detected",
                )

            if len(results) > 1:
                return PostRecognizeFaceResponse(
                    timestamp=str(time.time()),
                    status="error",
                    message="Multiple faces detected",
                )

            if len(results) == 1:
                face = results[0].face

            if form.recognition_model == RecognitionModel.ARCFACE:
                embedding = self.recognizer.get_embedding(face)

            if embedding:
                # find nearest face embedding
                nearest_face_embedding = FaceEmbeddingAPI(
                    self.session
                ).find_nearest_face_embedding(
                    embedding, form.recognition_model, self.cfg.recognizer.dist_method
                )

                if nearest_face_embedding:
                    distance = find_cosine_distance(
                        embedding, nearest_face_embedding.embedding
                    )

                    if distance > self.cfg.recognizer.min_dist_threshold:
                        return PostRecognizeFaceResponse(
                            timestamp=str(time.time()),
                            status="error",
                            message="Face not recognized",
                        )

                    data = RecognizeData(
                        id=nearest_face_embedding.id,
                        profile_id=nearest_face_embedding.profile.id,
                        name=nearest_face_embedding.profile.name,
                        distance=distance,
                    )
                    return PostRecognizeFaceResponse(
                        timestamp=str(time.time()),
                        status="success",
                        message="Face recognized",
                        data=data,
                    )

            return PostRecognizeFaceResponse(
                timestamp=str(time.time()),
                status="error",
                message="Failed to recognize face",
            )

        except Exception as e:
            log.error(f"Error: {e}")
            return PostRecognizeFaceResponse(
                timestamp=str(time.time()),
                status="error",
                message="Failed to recognize face",
            )

    # [DELETE]/api/face/{id}
    async def delete_face_embedding(self, id: int) -> BaseAPIResponse:
        try:
            face_embedding = self.face_embedding_api.get_face_embedding_by_id(id)
            if not face_embedding:
                return BaseAPIResponse(
                    timestamp=str(time.time()),
                    status="error",
                    message="Face embedding not found",
                )

            self.face_embedding_api.delete_face_embedding(id)
            return BaseAPIResponse(
                timestamp=str(time.time()),
                status="success",
                message="Face embedding deleted",
            )
        except Exception as e:
            log.error(f"Error: {e}")
            return BaseAPIResponse(
                timestamp=str(time.time()),
                status="error",
                message="Failed to delete face embedding",
            )

    async def preprocess_img_bytes(self, img_bytes: bytes) -> np.ndarray:
        """Preprocess image bytes."""

        img = Image.open(BytesIO(img_bytes))
        img = np.array(img)

        # if PNG, convert to RGB
        if img.shape[-1] == 4:
            img = img[..., :3]

        return img
