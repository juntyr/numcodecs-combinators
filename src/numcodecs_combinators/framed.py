"""
This module defines the [`FramedCodecStack`][numcodecs_combinators.framed.FramedCodecStack] class, which exposes a framed stack of codecs as a combined codec.
"""

__all__ = ["FramedCodecStack"]

from io import BytesIO
from sys import byteorder
from typing import Callable, Optional

import numcodecs
import numcodecs.compat
import numcodecs.registry
import numpy as np
import varint
from numcodecs.abc import Codec
from typing_extensions import Buffer, Self  # MSPV 3.12

from .abc import CodecCombinatorMixin


class FramedCodecStack(Codec, CodecCombinatorMixin, tuple[Codec]):
    """
    A framed stack of codecs, which makes up a combined codec.

    On encoding, the result of applying the codecs from left to right to encode
    is framed s.t. the data types and shapes of all arrays (input,
    intermediary, encoded) are stored as part of the encoding, which is output
    as a bytestring.

    On decoding, this framing information is used to apply the codecs from
    right to left to decode into known output data types and shapes.

    Therefore, the [`FramedCodecStack`][numcodecs_combinators.framed.FramedCodecStack]
    can be used to combine codecs which require knowing the output data type
    and shape during decoding. It can also be used to encode arrays into
    bytestrings.

    Unlike the [`CodecStack`][numcodecs_combinators.stack.CodecStack], this
    class does *not* provide an additional `encode_decode(buf)` method, since
    it is equivalent to `framed.decode(stack.encode(buf))` due to the framing.
    """

    __slots__ = ()

    codec_id: str = "combinators.framed"  # type: ignore

    def __init__(self, *args: dict | Codec):
        pass

    def __new__(cls, *args: dict | Codec) -> Self:
        return super(FramedCodecStack, cls).__new__(
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
            Encoded and framed data as a bytestring.
        """

        encoded = buf
        encoded_ndarray = np.asarray(
            numcodecs.compat.ensure_contiguous_ndarray_like(encoded, flatten=False)
        )

        frames = [(encoded_ndarray.dtype, encoded_ndarray.shape)]

        for codec in self:
            encoded = codec.encode(encoded_ndarray)
            encoded_ndarray = np.asarray(
                numcodecs.compat.ensure_contiguous_ndarray_like(encoded, flatten=False)
            )
            frames.append((encoded_ndarray.dtype, encoded_ndarray.shape))

        # convert the encoded array to little endian bytes
        encoded_byteorder = encoded_ndarray.dtype.byteorder
        encoded_byteorder = (
            encoded_byteorder
            if encoded_byteorder in ("<", ">")
            else ("<" if (byteorder == "little") else ">")
        )
        if encoded_byteorder != "<":
            encoded_ndarray = encoded_ndarray.byteswap()
        encoded_bytes = encoded_ndarray.tobytes()

        message = [varint.encode(len(frames))]

        for dtype, shape in frames:
            message.append(varint.encode(len(dtype.str)))
            message.append(dtype.str.encode("ascii"))

            message.append(varint.encode(len(shape)))
            for s in shape:
                message.append(varint.encode(s))

        message.append(encoded_bytes)

        return b"".join(message)

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

        b = numcodecs.compat.ensure_bytes(buf)

        b_io = BytesIO(b)

        n_frames = varint.decode_stream(b_io)
        assert n_frames == len(self) + 1, (
            f"encoded data must contain {len(self) + 1} frames, found {n_frames}"
        )

        frames = []
        for _ in range(n_frames):
            dtype = np.dtype(b_io.read(varint.decode_stream(b_io)).decode("ascii"))
            shape = tuple(
                varint.decode_stream(b_io) for _ in range(varint.decode_stream(b_io))
            )
            frames.append((dtype, shape))

        # read the decoded array from the little endian bytes
        decoded = np.frombuffer(
            b_io.read(np.prod(shape, dtype=int) * dtype.itemsize),
            dtype=dtype.newbyteorder("<"),
            count=np.prod(shape, dtype=int),
        ).reshape(shape)
        dtype_byteorder = dtype.byteorder
        dtype_byteorder = (
            dtype_byteorder
            if dtype_byteorder in ("<", ">")
            else ("<" if (byteorder == "little") else ">")
        )
        if dtype_byteorder != "<":
            decoded = decoded.byteswap()

        for codec, (dtype, shape) in zip(reversed(self), frames[:-1][::-1]):
            decoded = (
                codec.decode(
                    decoded,
                    out=np.empty(shape, dtype),
                )
                .view(dtype)
                .reshape(shape)
            )

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of the framed codec stack.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this stack from the returned config.

        Returns
        -------
        config : dict
            Configuration of the framed codec stack.
        """

        return dict(
            id=type(self).codec_id,
            codecs=tuple(codec.get_config() for codec in self),
        )

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """
        Instantiate the framed codec stack from a configuration [`dict`][dict].

        Parameters
        ----------
        config : dict
            Configuration of the framed codec stack.

        Returns
        -------
        stack : FramedCodecStack
            Instantiated framed codec stack.
        """

        return cls(*config["codecs"])

    def __repr__(self) -> str:
        repr = ", ".join(f"{codec!r}" for codec in self)

        return f"{type(self).__name__}({repr})"

    def map(self, mapper: Callable[[Codec], Codec]) -> "FramedCodecStack":
        """
        Apply the `mapper` to all codecs that are in this framed stack.
        In the returned stack, each codec is replaced by its mapped codec.

        The `mapper` should recursively apply itself to any inner codecs that
        also implement the [`CodecCombinatorMixin`][numcodecs_combinators.abc.CodecCombinatorMixin]
        mixin.

        To automatically handle the recursive application as a caller, you can
        use
        ```python
        numcodecs_combinators.map_codec(stack, mapper)
        ```
        instead.

        Parameters
        ----------
        mapper : Callable[[Codec], Codec]
            The callable that should be applied to each codec to map over this
            framed codec stack.

        Returns
        -------
        mapped : FramedCodecStack
            The mapped framed codec stack.
        """

        return FramedCodecStack(*map(mapper, self))

    def __add__(self, other) -> "FramedCodecStack":
        return FramedCodecStack(*tuple.__add__(self, other))

    def __mul__(self, other) -> "FramedCodecStack":
        return FramedCodecStack(*tuple.__mul__(self, other))

    def __rmul__(self, other) -> "FramedCodecStack":
        return FramedCodecStack(*tuple.__rmul__(self, other))


numcodecs.registry.register_codec(FramedCodecStack)
