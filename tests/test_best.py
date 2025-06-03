import numcodecs
import numcodecs.compat
import numpy as np

import numcodecs_combinators
from numcodecs_combinators.best import PickBestCodec
from numcodecs_combinators.framed import FramedCodecStack


def assert_config_roundtrip(codec: numcodecs.abc.Codec):
    config = codec.get_config()
    codec2 = numcodecs.get_codec(config)
    assert codec2 == codec


def test_init_config():
    best = PickBestCodec()
    assert len(best) == 0
    assert_config_roundtrip(best)

    best = PickBestCodec(dict(id="zlib", level=9))
    assert len(best) == 1
    assert_config_roundtrip(best)

    best = PickBestCodec(dict(id="zlib", level=9), numcodecs.CRC32())
    assert len(best) == 2
    assert_config_roundtrip(best)


def test_encode_decode():
    for best in [
        PickBestCodec(),
        PickBestCodec(dict(id="combinators.framed", codecs=[dict(id="zlib", level=9)])),
        PickBestCodec(
            FramedCodecStack(numcodecs.Zlib(level=9)),
            FramedCodecStack(numcodecs.CRC32()),
        ),
        PickBestCodec(
            FramedCodecStack(numcodecs.Zlib(level=9)),
            FramedCodecStack(numcodecs.CRC32()),
            FramedCodecStack(numcodecs.Zstd(level=20)),
        ),
    ]:
        for data in [
            np.zeros(shape=(0,)),
            np.array(3),
            np.array([97, 98, 99], dtype=np.uint8),
            np.linspace(1, 100, 100).reshape(10, 10),
            np.linspace(1, 100, 100).reshape(10, 10).byteswap(),
        ]:
            encoded = best.encode(data)
            if len(best) > 0:
                assert isinstance(encoded, bytes)
            decoded = best.decode(encoded)
            print(best)
            assert np.all(decoded == data)


def test_map():
    best = PickBestCodec(numcodecs.Zlib(level=9), numcodecs.CRC32())

    mapped = numcodecs_combinators.map_codec(best, lambda c: c)
    assert mapped == best

    mapped = numcodecs_combinators.map_codec(best, lambda c: PickBestCodec(c))
    assert mapped == PickBestCodec(
        PickBestCodec(
            PickBestCodec(numcodecs.Zlib(level=9)),
            PickBestCodec(numcodecs.CRC32()),
        )
    )

    mapped = numcodecs_combinators.map_codec(mapped, lambda c: PickBestCodec(c))
    assert mapped == PickBestCodec(
        PickBestCodec(
            PickBestCodec(
                PickBestCodec(
                    PickBestCodec(
                        PickBestCodec(PickBestCodec(numcodecs.Zlib(level=9)))
                    ),
                    PickBestCodec(PickBestCodec(PickBestCodec(numcodecs.CRC32()))),
                )
            )
        )
    )
