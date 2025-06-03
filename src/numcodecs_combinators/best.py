"""
This module defines the [`PickBestCodec`][numcodecs_combinators.best.PickBestCodec] class, which picks the codec that encoded the data best.
"""

__all__ = ["PickBestCodec"]

from io import BytesIO
from typing import Callable, Optional

import numcodecs
import numcodecs.compat
import numcodecs.registry
import numpy as np
import varint
from numcodecs.abc import Codec
from typing_extensions import Buffer, Self  # MSPV 3.12

from .abc import CodecCombinatorMixin


class PickBestCodec(Codec, CodecCombinatorMixin, tuple[Codec]):
    """
    A codec that tries encoding with all combined codecs and then picks the one with the fewest bytes.

    The inner codecs must all encode to 1D byte arrays. To use a codec not
    encoding to bytes with this combinator, you can wrap it using
    [`FramedCodecStack(codec)`][numcodecs_combinators.framed.FramedCodecStack]
    combinator.

    This combinator uses the ULEB128 variable length integer encoding to encode
    the index of the codec that was chosen to encode and uses this index as a
    header before the encoded bytes. The header index is only included if this
    combinator wraps at least two codecs. If this combinator wraps zero codecs,
    it passes the original data through unchanged.
    """

    __slots__ = ()

    codec_id: str = "combinators.best"  # type: ignore

    def __init__(self, *args: dict | Codec):
        pass

    def __new__(cls, *args: dict | Codec) -> Self:
        return super(PickBestCodec, cls).__new__(
            cls,
            tuple(
                codec
                if isinstance(codec, Codec)
                else numcodecs.registry.get_codec(codec)
                for codec in args
            ),
        )

    def encode(self, buf: Buffer) -> bytes:
        """Encode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : bytes
            Encoded and data as a bytestring.
        """

        if len(self) == 0:
            return buf

        data = numcodecs.compat.ensure_ndarray(buf)

        best_size = np.inf
        best_index = None
        best_encoded = None

        for i, codec in enumerate(self):
            encoded = numcodecs.compat.ensure_ndarray(codec.encode(np.copy(data)))
            assert encoded.dtype == np.dtype("uint8"), (
                f"codec best[{i}] must encode to bytes"
            )
            assert encoded.ndim <= 1, f"codec best[{i}] must encode to 1D bytes"

            if encoded.nbytes < best_size:
                best_size = encoded.nbytes
                best_index = i
                best_encoded = encoded

        encoded_index = varint.encode(best_index)
        encoded_bytes = numcodecs.compat.ensure_bytes(best_encoded)

        if len(self) == 1:
            return encoded_bytes

        return encoded_index + encoded_bytes

    def decode(self, buf: Buffer, out: Optional[Buffer] = None) -> Buffer:
        """Decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. Must be an object representing a bytestring, e.g.
            [`bytes`][bytes] or a 1D array of [`np.uint8`][numpy.uint8]s etc.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this buffer must
            be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style
            buffer protocol.
        """

        if len(self) == 0:
            return numcodecs.compat.ndarray_copy(buf, out)

        b = numcodecs.compat.ensure_bytes(buf)
        b_io = BytesIO(b)

        if len(self) == 1:
            best_index = 0
        else:
            best_index = varint.decode_stream(b_io)

        return self[best_index].decode(b_io.read(), out=out)

    def get_config(self) -> dict:
        """
        Returns the configuration of the best codec combinator.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this combinator from the returned config.

        Returns
        -------
        config : dict
            Configuration of the best codec combinator.
        """

        return dict(
            id=type(self).codec_id,
            codecs=tuple(codec.get_config() for codec in self),
        )

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """
        Instantiate the best codec combinator from a configuration [`dict`][dict].

        Parameters
        ----------
        config : dict
            Configuration of the best codec combinator.

        Returns
        -------
        best : PickBestCodec
            Instantiated best codec combinator.
        """

        return cls(*config["codecs"])

    def __repr__(self) -> str:
        repr = ", ".join(f"{codec!r}" for codec in self)

        return f"{type(self).__name__}({repr})"

    def map(self, mapper: Callable[[Codec], Codec]) -> "PickBestCodec":
        """
        Apply the `mapper` to all codecs that are in this combinator.
        In the returned combinator, each codec is replaced by its mapped codec.

        The `mapper` should recursively apply itself to any inner codecs that
        also implement the [`CodecCombinatorMixin`][numcodecs_combinators.abc.CodecCombinatorMixin]
        mixin.

        To automatically handle the recursive application as a caller, you can
        use
        ```python
        numcodecs_combinators.map_codec(best, mapper)
        ```
        instead.

        Parameters
        ----------
        mapper : Callable[[Codec], Codec]
            The callable that should be applied to each codec to map over this
            best codec combinator.

        Returns
        -------
        mapped : PickBestCodec
            The mapped best codec combinator.
        """

        return PickBestCodec(*map(mapper, self))

    def __add__(self, other) -> "PickBestCodec":
        return PickBestCodec(*tuple.__add__(self, other))

    def __mul__(self, other) -> "PickBestCodec":
        return PickBestCodec(*tuple.__mul__(self, other))

    def __rmul__(self, other) -> "PickBestCodec":
        return PickBestCodec(*tuple.__rmul__(self, other))


numcodecs.registry.register_codec(PickBestCodec)
