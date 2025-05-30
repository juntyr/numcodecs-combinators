import numcodecs
import numcodecs.compat
import numpy as np

import numcodecs_combinators
from numcodecs_combinators.framed import FramedCodecStack


def assert_config_roundtrip(codec: numcodecs.abc.Codec):
    config = codec.get_config()
    codec2 = numcodecs.get_codec(config)
    assert codec2 == codec


def test_init_config():
    stack = FramedCodecStack()
    assert len(stack) == 0
    assert_config_roundtrip(stack)

    stack = FramedCodecStack(dict(id="zlib", level=9))
    assert len(stack) == 1
    assert_config_roundtrip(stack)

    stack = FramedCodecStack(dict(id="zlib", level=9), numcodecs.CRC32())
    assert len(stack) == 2
    assert_config_roundtrip(stack)


def test_encode_decode():
    for stack in [
        FramedCodecStack(),
        FramedCodecStack(dict(id="zlib", level=9)),
        FramedCodecStack(numcodecs.Zlib(level=9), numcodecs.CRC32()),
    ]:
        for data in [
            b"abc",
            np.array(3),
            np.linspace(1, 100, 100).reshape(10, 10),
            np.linspace(1, 100, 100).reshape(10, 10).byteswap(),
        ]:
            encoded = stack.encode(data)
            assert isinstance(encoded, bytes)
            decoded = stack.decode(encoded)
            assert np.all(decoded == numcodecs.compat.ensure_ndarray_like(data))


def test_map():
    stack = FramedCodecStack(numcodecs.Zlib(level=9), numcodecs.CRC32())

    mapped = numcodecs_combinators.map_codec(stack, lambda c: c)
    assert mapped == stack

    mapped = numcodecs_combinators.map_codec(stack, lambda c: FramedCodecStack(c))
    assert mapped == FramedCodecStack(
        FramedCodecStack(
            FramedCodecStack(numcodecs.Zlib(level=9)),
            FramedCodecStack(numcodecs.CRC32()),
        )
    )

    mapped = numcodecs_combinators.map_codec(mapped, lambda c: FramedCodecStack(c))
    assert mapped == FramedCodecStack(
        FramedCodecStack(
            FramedCodecStack(
                FramedCodecStack(
                    FramedCodecStack(
                        FramedCodecStack(FramedCodecStack(numcodecs.Zlib(level=9)))
                    ),
                    FramedCodecStack(
                        FramedCodecStack(FramedCodecStack(numcodecs.CRC32()))
                    ),
                )
            )
        )
    )
