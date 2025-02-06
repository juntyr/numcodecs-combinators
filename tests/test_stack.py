import numcodecs
import numpy as np
import xarray as xr

import numcodecs_combinators
from numcodecs_combinators.stack import CodecStack


def assert_config_roundtrip(codec: numcodecs.abc.Codec):
    config = codec.get_config()
    codec2 = numcodecs.get_codec(config)
    assert codec2 == codec


def test_init_config():
    stack = CodecStack()
    assert len(stack) == 0
    assert_config_roundtrip(stack)

    stack = CodecStack(dict(id="zlib", level=9))
    assert len(stack) == 1
    assert_config_roundtrip(stack)

    stack = CodecStack(dict(id="zlib", level=9), numcodecs.CRC32())
    assert len(stack) == 2
    assert_config_roundtrip(stack)


def test_encode_decode():
    stack = CodecStack(numcodecs.Zlib(level=9), numcodecs.CRC32())

    stack_encoded = stack.encode(b"abc")
    encoded = numcodecs.CRC32().encode(numcodecs.Zlib(level=9).encode(b"abc"))
    assert type(stack_encoded) is type(encoded)
    assert (np.array(stack_encoded) == np.array(encoded)).all()

    stack_decoded = stack.decode(stack_encoded)
    decoded = numcodecs.Zlib(level=9).decode(numcodecs.CRC32().decode(encoded))
    assert stack_decoded == decoded
    assert stack_decoded == b"abc"

    encoded_decoded = stack.encode_decode(b"abc")
    assert encoded_decoded == b"abc"

    encoded_decoded = stack.encode_decode_data_array(xr.DataArray([1.0, 2.0, 3.0]))
    assert encoded_decoded.equals(xr.DataArray([1.0, 2.0, 3.0]))

    encoded_decoded = stack.encode_decode_data_array(
        xr.DataArray([1.0, 2.0, 3.0]).chunk(1)
    )
    assert encoded_decoded.equals(xr.DataArray([1.0, 2.0, 3.0]))


def test_map():
    stack = CodecStack(numcodecs.Zlib(level=9), numcodecs.CRC32())

    mapped = numcodecs_combinators.map_codec(stack, lambda c: c)
    assert mapped == stack

    mapped = numcodecs_combinators.map_codec(stack, lambda c: CodecStack(c))
    assert mapped == CodecStack(
        CodecStack(CodecStack(numcodecs.Zlib(level=9)), CodecStack(numcodecs.CRC32()))
    )

    mapped = numcodecs_combinators.map_codec(mapped, lambda c: CodecStack(c))
    assert mapped == CodecStack(
        CodecStack(
            CodecStack(
                CodecStack(
                    CodecStack(CodecStack(CodecStack(numcodecs.Zlib(level=9)))),
                    CodecStack(CodecStack(CodecStack(numcodecs.CRC32()))),
                )
            )
        )
    )
