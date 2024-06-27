"""Arcface recognizer core module"""

import rootutils

ROOT = rootutils.autosetup()

from typing import List, Union

import cv2
import numpy as np

from src.core.onnx_base import OnnxBase


class ArcfaceOnnxCore(OnnxBase):

    def __init__(
        self,
        engine_path: str,
        provider: str = "cpu",
    ) -> None:
        """
        Initialize Arcface recognizer core.

        Args:
            engine_path (str): Path to the engine file.
            provider (str): Inference provider (default: "cpu").

        Returns:
            None
        """
        super().__init__(engine_path, provider)

    def get_embedding(self, face: np.ndarray) -> Union[List[float], None]:
        """
        Get face embedding from the engine.

        Args:
            face (np.ndarray): Face image.

        Returns:
            Union[List[float], None]: Face embedding or None.
        """
        face = self.preprocess(face)
        outputs = self.engine.run(None, {self.metadata[0].input_name: face})[
            0
        ].tolist()[0]
        return outputs

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Preprocess input image.

        Args:
            img (np.ndarray): Input image.

        Returns:
            np.ndarray: Preprocessed image.
        """

        dst_h, dst_w = self.img_shape
        resized_img = np.zeros((1, dst_h, dst_w, 3), dtype=np.float32)
        img = cv2.resize(img, dsize=(dst_w, dst_h))
        resized_img[0] = img
        resized_img = resized_img.transpose((0, 3, 1, 2))

        return resized_img


"""debug session"""
if __name__ == "__main__":
    engine = ArcfaceOnnxCore("static/recognizer/arcfaceresnet100-11-int8.onnx")
    engine.setup()

    img = cv2.imread("tmp/1.jpg")
    embedding = engine.get_embedding(img)

    if embedding:
        print(embedding)
