"""
This module defines the [`CodecStack`][numcodecs_combinators.stack.CodecStack] class, which exposes a stack of codecs as a combined codec.
"""

__all__ = ["CodecStack"]

from typing import Optional, Callable
from typing_extensions import Buffer, Self  # MSPV 3.12

import numcodecs
import numcodecs.compat
import numcodecs.registry
import numpy as np

try:
    import xarray as xr
except ImportError:
    pass

from numcodecs.abc import Codec

from .abc import CodecCombinatorMixin


class CodecStack(Codec, CodecCombinatorMixin, tuple[Codec]):
    """
    A stack of codecs, which makes up a combined codec.

    On encoding, the codecs are applied to encode from left to right, i.e.
    ```python
    CodecStack(a, b, c).encode(buf)
    ```
    computes
    ```python
    c.encode(b.encode(a.encode(buf)))
    ```

    On decoding, the codecs are applied to decode from right to left, i.e.
    ```python
    CodecStack(a, b, c).decode(buf)
    ```
    computes
    ```python
    a.decode(b.decode(c.decode(buf)))
    ```

    The [`CodecStack`][numcodecs_combinators.stack.CodecStack] provides the
    additional
    [`encode_decode(buf)`][numcodecs_combinators.stack.CodecStack.encode_decode]
    method that computes
    ```python
    stack.decode(stack.encode(buf))
    ```
    but makes use of knowing the shapes and dtypes of all intermediary encoding
    stages.
    """

    __slots__ = ()

    codec_id: str = "combinators.stack"  # type: ignore

    def __init__(self, *args: dict | Codec):
        pass

    def __new__(cls, *args: dict | Codec) -> Self:
        return super(CodecStack, cls).__new__(
            cls,
            tuple(
                codec
                if isinstance(codec, Codec)
                else numcodecs.registry.get_codec(codec)
                for codec in args
            ),
        )

    def encode(self, buf: Buffer) -> Buffer:
        """Encode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        enc : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        """

        encoded = buf
        for codec in self:
            encoded = codec.encode(
                numcodecs.compat.ensure_contiguous_ndarray_like(encoded, flatten=False)
            )
        return encoded

    def decode(self, buf: Buffer, out: Optional[Buffer] = None) -> Buffer:
        """Decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. May be any object supporting the new-style buffer
            protocol.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this buffer must
            be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style
            buffer protocol.
        """

        decoded = buf
        for codec in reversed(self):
            decoded = codec.decode(
                numcodecs.compat.ensure_contiguous_ndarray_like(decoded, flatten=False),
                out=None,
            )
        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore

    def encode_decode(self, buf: Buffer) -> Buffer:
        """
        Encode, then decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Data to be encoded. May be any object supporting the new-style
            buffer protocol.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style
            buffer protocol.
        """

        encoded = numcodecs.compat.ensure_contiguous_ndarray_like(buf, flatten=False)
        silhouettes = []

        for codec in self:
            silhouettes.append((encoded.shape, np.dtype(encoded.dtype.name)))
            encoded = numcodecs.compat.ensure_contiguous_ndarray_like(
                codec.encode((encoded)), flatten=False
            )

        decoded = encoded

        for codec in reversed(self):
            shape, dtype = silhouettes.pop()
            out = np.empty(shape=shape, dtype=dtype)
            decoded = codec.decode(decoded, out).reshape(shape)

        if isinstance(decoded, type(buf)):
            return decoded

        return type(buf)(decoded)  # type: ignore

    def encode_decode_data_array(self, da: "xr.DataArray") -> "xr.DataArray":
        """
        Encode, then decode the data array in `da`.

        The encode-decode computation may be deferred until the
        [`compute`][xarray.DataArray.compute] method is called on the result.

        This method requires the optional [`xarray`][xarray] dependency to be
        installed.

        Parameters
        ----------
        da : xr.DataArray
            Data to be encoded.

        Returns
        -------
        dec : xr.DataArray
            Decoded data.
        """

        import xarray as xr

        def encode_decode_data_array_single_chunk(
            da: xr.DataArray,
        ) -> xr.DataArray:
            single_chunk = {dim: -1 for dim in da.dims}

            # return early for zero-sized arrays
            if da.size == 0:
                return da.copy(deep=False).chunk(single_chunk)

            # eagerly compute the input chunk and encode and decode it
            decoded = self.encode_decode(da.values)  # type: ignore

            return da.copy(deep=False, data=decoded).chunk(single_chunk)

        return xr.map_blocks(encode_decode_data_array_single_chunk, da)

    def get_config(self) -> dict:
        """
        Returns the configuration of the codec stack.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this stack from the returned config.

        Returns
        -------
        config : dict
            Configuration of the codec stack.
        """

        return dict(
            id=type(self).codec_id,
            codecs=tuple(codec.get_config() for codec in self),
        )

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """
        Instantiate the codec stack from a configuration [`dict`][dict].

        Parameters
        ----------
        config : dict
            Configuration of the codec stack.

        Returns
        -------
        stack : CodecStack
            Instantiated codec stack.
        """

        return cls(*config["codecs"])

    def __repr__(self) -> str:
        repr = ", ".join(f"{codec!r}" for codec in self)

        return f"{type(self).__name__}({repr})"

    def map(self, mapper: Callable[[Codec], Codec]) -> "CodecStack":
        """
        Apply the `mapper` to all codecs that are in this stack.
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
            codec stack.

        Returns
        -------
        mapped : CodecStack
            The mapped codec stack.
        """

        return CodecStack(*map(mapper, self))

    def __add__(self, other) -> "CodecStack":
        return CodecStack(*tuple.__add__(self, other))

    def __mul__(self, other) -> "CodecStack":
        return CodecStack(*tuple.__mul__(self, other))

    def __rmul__(self, other) -> "CodecStack":
        return CodecStack(*tuple.__rmul__(self, other))


numcodecs.registry.register_codec(CodecStack)
