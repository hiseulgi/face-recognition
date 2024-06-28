"""Recognition router module."""

import rootutils

ROOT = rootutils.autosetup()

import time
from io import BytesIO

import numpy as np
from fastapi import APIRouter, Depends
from omegaconf import DictConfig
from PIL import Image

from src.core.controller.face_embedding_api import FaceEmbeddingAPI
from src.core.controller.profile_api import ProfileAPI
from src.core.detector.faceonnx_core import FaceOnnxCore
from src.core.recognizer.arcface_core import ArcfaceOnnxCore
from src.database.database import Database
from src.schema.database_schema import FaceEmbeddingCreate, ProfileCreate
from src.schema.enums_schema import DetectionModel, DistanceMetric, RecognitionModel
from src.schema.recognition_api_schema import (
    BaseAPIResponse,
    GetAllFaceEmbeddingsResponse,
    PostRecognizeFaceRequest,
    PostRecognizeFaceResponse,
    PostRegisterFaceRequest,
    RecognizeData,
)
from src.utils.exception import APIExceptions
from src.utils.logger import get_logger
from src.utils.math import find_cosine_distance

exceptions = APIExceptions()

log = get_logger()


class RecognitionRouter:
    """Recognition router."""

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize recognition router.

        Args:
            cfg (DictConfig): Configurations.

        Returns:
            None
        """
        self.cfg = cfg
        self.router = APIRouter()

        self.setup_model()
        self.setup_db()
        self.init_routes()

    def setup_db(self) -> None:
        """Setup database."""

        self.db = Database(**self.cfg.database.postgres)

        self.db.connect()
        self.db.setup_pgvector()

        self.session = self.db.SessionLocal()

        self.face_embedding_api = FaceEmbeddingAPI(self.session)
        self.profile_api = ProfileAPI(self.session)

        log.info("Database tables created")

    def setup_model(self) -> None:
        """Setup models."""

        self.detector = FaceOnnxCore(**self.cfg.detector.faceonnx)
        self.recognizer = ArcfaceOnnxCore(**self.cfg.recognizer.arcface)

        self.detector.setup()
        self.recognizer.setup()

    def init_routes(self) -> None:
        """Initialize routes."""

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
            status_code=201,
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
        """
        Get all face embeddings.
        """

        # get all face embeddings
        face_embeddings = self.face_embedding_api.get_all_face_embeddings()

        # iterate through face embeddings
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
            detail="Face embeddings retrieved",
            data=data,
        )

    # [POST]/api/face/register
    async def register_face(
        self, form: PostRegisterFaceRequest = Depends()
    ) -> BaseAPIResponse:
        """
        Register face from raw image.

        Args:
            form(PostRegisterFaceRequest): PostRegisterFaceRequest object

        Raises:
            exceptions.NotFound: No face detected
            exceptions.BadRequest: Multiple faces detected
        """

        # read image bytes and preprocess
        img_bytes = await form.image.read()
        img = await self.preprocess_img_bytes(img_bytes)

        # detect face according to detection model
        if form.detection_model == DetectionModel.FACEONNX:
            results = self.detector.detect_face(img)

        if results is None or len(results) == 0:
            raise exceptions.NotFound("No face detected")

        if len(results) > 1:
            raise exceptions.BadRequest("Multiple faces detected")

        face = results[0].face

        # get face embedding according to recognition model
        if form.recognition_model == RecognitionModel.ARCFACE:
            embedding = self.recognizer.get_embedding(face)

        # get profile by name
        profile = self.profile_api.get_profile_by_name(form.name)

        # create profile if not exists
        if not profile:
            profile = self.profile_api.create_profile(ProfileCreate(name=form.name))

        # create face embedding
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
            detail="Face registered",
            data={
                "id": face_embedding.id,
                "profile_id": profile.id,
                "name": profile.name,
            },
        )

    # [POST]/api/face/recognize
    async def recognize_face(
        self, form: PostRecognizeFaceRequest = Depends()
    ) -> PostRecognizeFaceResponse:
        """
        Recognize face from raw image.

        Args:
            form (PostRecognizeFaceRequest): PostRecognizeFaceRequest object

        Raises:
            exceptions.NotFound: No face detected or Face not recognized
            exceptions.BadRequest: Multiple faces detected
        """

        # read image bytes and preprocess
        img_bytes = await form.image.read()
        img = await self.preprocess_img_bytes(img_bytes)

        # detect face according to detection model
        if form.detection_model == DetectionModel.FACEONNX:
            results = self.detector.detect_face(img)

        if results is None or len(results) == 0:
            raise exceptions.NotFound("No face detected")

        if len(results) > 1:
            raise exceptions.BadRequest("Multiple faces detected")

        face = results[0].face

        # get face embedding according to recognition model
        if form.recognition_model == RecognitionModel.ARCFACE:
            embedding = self.recognizer.get_embedding(face)

        # find nearest face embedding
        nearest_face_embedding = FaceEmbeddingAPI(
            self.session
        ).find_nearest_face_embedding(
            embedding, form.recognition_model, self.cfg.recognizer.dist_method
        )

        # if nearest face embedding not found
        if not nearest_face_embedding:
            raise exceptions.NotFound("Face not recognized")

        # calculate distance
        if self.cfg.recognizer.dist_method == DistanceMetric.COSINE:
            distance = find_cosine_distance(embedding, nearest_face_embedding.embedding)

        # if distance is less than threshold
        if distance > self.cfg.recognizer.min_dist_threshold:
            raise exceptions.NotFound("Face not recognized")

        data = RecognizeData(
            id=nearest_face_embedding.id,
            profile_id=nearest_face_embedding.profile.id,
            name=nearest_face_embedding.profile.name,
            distance=distance,
        )

        return PostRecognizeFaceResponse(
            timestamp=str(time.time()),
            status="success",
            detail="Face recognized",
            data=data,
        )

    # [DELETE]/api/face/{id}
    async def delete_face_embedding(self, id: int) -> BaseAPIResponse:
        """
        Delete face embedding by ID.

        Args:
            id (int): Face embedding ID

        Raises:
            exceptions.NotFound: Face embedding not found
        """

        # get face embedding by ID
        face_embedding = self.face_embedding_api.get_face_embedding_by_id(id)

        if not face_embedding:
            raise exceptions.NotFound("Face embedding not found")

        self.face_embedding_api.delete_face_embedding(id)
        return BaseAPIResponse(
            timestamp=str(time.time()),
            status="success",
            detail="Face embedding deleted",
        )

    async def preprocess_img_bytes(self, img_bytes: bytes) -> np.ndarray:
        """Preprocess image bytes."""

        img = Image.open(BytesIO(img_bytes))
        img = np.array(img)

        # if PNG, convert to RGB
        if img.shape[-1] == 4:
            img = img[..., :3]

        return img
