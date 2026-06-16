from typing import Any, Optional

import numpy as np

class Face:
    bbox: np.ndarray
    det_score: float
    kps: np.ndarray
    embedding: np.ndarray
    normed_embedding: np.ndarray

class FaceAnalysis:
    def __init__(self, name: str = ..., providers: list[str] = ...) -> None: ...
    def prepare(self, ctx_id: int = ..., det_size: tuple[int, int] = ...) -> None: ...
    def get(self, img: np.ndarray) -> list[Face]: ...

class SwapperModel:
    def get(self, img: np.ndarray, face: Face, old_face: Face, paste_back: bool = True) -> np.ndarray: ...
