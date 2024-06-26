import rootutils

ROOT = rootutils.autosetup()

from typing import List, Union

import cv2
import numpy as np

from src.core.onnx_base import OnnxBase
from src.schema.faceonnx_schema import FaceONNXResultSchema
from src.utils.logger import get_logger
from src.utils.nms import hard_nms

log = get_logger()


class FaceOnnxCore(OnnxBase):
    """Only support 1 batch size for now (according to the original model)"""

    def __init__(
        self,
        engine_path: str,
        provider: str = "cpu",
    ) -> None:
        super().__init__(engine_path, provider)

    def detect_face(
        self,
        img: np.ndarray,
        prob_threshold: float = 0.7,
        iou_threshold: float = 0.5,
        top_k: int = -1,
    ) -> Union[List[FaceONNXResultSchema], None]:

        # preprocess
        preprocessed_img = self.preprocess(img)

        # inference
        outputs = self.engine.run(None, {self.metadata[0].input_name: preprocessed_img})

        # postprocess
        results = self.postprocess(
            raw_img=img,
            outputs=outputs,
            prob_threshold=prob_threshold,
            iou_threshold=iou_threshold,
            top_k=top_k,
        )

        return results

    def preprocess(self, img: np.ndarray) -> np.ndarray:

        dst_h, dst_w = self.img_shape
        resized_img = np.zeros((1, dst_h, dst_w, 3), dtype=np.float32)

        img = cv2.resize(img, dsize=(dst_w, dst_h))
        img_mean = np.array([127, 127, 127])
        img = (img - img_mean) / 128
        resized_img[0] = img

        resized_img = resized_img.transpose((0, 3, 1, 2))

        return resized_img

    def postprocess(
        self,
        raw_img: np.ndarray,
        outputs: np.ndarray,
        prob_threshold: float,
        iou_threshold: float = 0.5,
        top_k: int = -1,
    ) -> Union[List[FaceONNXResultSchema], None]:

        height, width, _ = raw_img.shape

        confidences, boxes = outputs
        boxes = boxes[0]
        confidences = confidences[0]

        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]

            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = hard_nms(
                box_probs,
                iou_threshold=iou_threshold,
                top_k=top_k,
            )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return None
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height

        boxes_outputs = picked_box_probs[:, :4].astype(np.int32)
        scores_outputs = picked_box_probs[:, 4]

        results = []
        for i in range(len(boxes_outputs)):
            result = FaceONNXResultSchema(
                face=raw_img[
                    boxes_outputs[i][1] : boxes_outputs[i][3],
                    boxes_outputs[i][0] : boxes_outputs[i][2],
                ],
                boxes=boxes_outputs[i],
                scores=scores_outputs[i],
            )
            results.append(result)

        return results


if __name__ == "__main__":
    engine_path = "static/detector/face_detector_640.onnx"
    face_detector = FaceOnnxCore(engine_path)
    face_detector.setup()

    img = cv2.imread("tmp/1.jpg")

    results = face_detector.detect_face(img)

    if results:
        for result in results:
            cv2.imshow("face", result.face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
