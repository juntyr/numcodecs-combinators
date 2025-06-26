import numpy as np


class ChunkedNdArray(np.ndarray):
    __slots__ = ()

    def __new__(cls, array):
        return np.asarray(array).view(cls)

    @property
    def chunked(self) -> bool:
        return True
