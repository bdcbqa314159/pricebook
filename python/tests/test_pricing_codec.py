"""Tests for pricing codec: encode/decode, framing, compression."""

from __future__ import annotations

import json

import pytest

from pricebook.pricing_codec import (
    Codec, CodecFormat, Compression, HEADER_SIZE,
)
from pricebook.pricing_schema import (
    PricingRequest, PricingConfig, QuoteMsg,
    irs_trade, quotes_market_data,
)


def _sample_request_dict():
    return PricingRequest(
        valuation_date="2026-04-28",
        trades=[irs_trade("T1", "USD", 0.035, "2031-04-28")],
        market_data=quotes_market_data([
            QuoteMsg("swap_rate", "5Y", 0.035).to_dict(),
        ]),
    ).to_dict()


def _large_request_dict():
    """Request with many quotes (for compression testing)."""
    quotes = [QuoteMsg("swap_rate", f"{i}Y", 0.03 + i * 0.001).to_dict()
              for i in range(1, 31)]
    return PricingRequest(
        valuation_date="2026-04-28",
        trades=[irs_trade(f"T{i}", "USD", 0.035, f"{2027+i}-04-28")
                for i in range(20)],
        market_data=quotes_market_data(quotes),
    ).to_dict()


# ---- JSON codec ----

class TestJSONCodec:

    def test_round_trip(self):
        codec = Codec(CodecFormat.JSON, Compression.NONE)
        msg = _sample_request_dict()
        data = codec.encode(msg)
        result = codec.decode(data)
        assert result == msg

    def test_framing(self):
        codec = Codec(CodecFormat.JSON, Compression.NONE)
        msg = _sample_request_dict()
        data = codec.encode(msg)
        # First 4 bytes = total length
        assert len(data) == Codec.read_frame_length(data[:4])

    def test_header_content(self):
        codec = Codec(CodecFormat.JSON, Compression.NONE)
        msg = _sample_request_dict()
        data = codec.encode(msg)
        # Header: format=1 (JSON), compression=0 (NONE)
        assert data[4] == 1   # JSON
        assert data[5] == 0   # NONE

    def test_encode_raw(self):
        codec = Codec(CodecFormat.JSON, Compression.NONE)
        msg = _sample_request_dict()
        raw = codec.encode_raw(msg)
        result = codec.decode_raw(raw)
        assert result == msg

    def test_json_compact(self):
        """JSON should use compact separators (no spaces)."""
        codec = Codec(CodecFormat.JSON)
        msg = {"key": "value", "number": 42}
        raw = codec.encode_raw(msg)
        assert b" " not in raw  # no spaces


# ---- Compression ----

class TestCompression:

    def test_zstd_round_trip(self):
        pytest.importorskip("zstandard")
        codec = Codec(CodecFormat.JSON, Compression.ZSTD)
        msg = _large_request_dict()
        data = codec.encode(msg)
        result = codec.decode(data)
        assert result == msg

    def test_zstd_smaller(self):
        pytest.importorskip("zstandard")
        msg = _large_request_dict()
        raw = Codec(CodecFormat.JSON, Compression.NONE).encode(msg)
        compressed = Codec(CodecFormat.JSON, Compression.ZSTD).encode(msg)
        assert len(compressed) < len(raw)

    def test_lz4_round_trip(self):
        pytest.importorskip("lz4.frame")
        codec = Codec(CodecFormat.JSON, Compression.LZ4)
        msg = _large_request_dict()
        data = codec.encode(msg)
        result = codec.decode(data)
        assert result == msg

    def test_lz4_smaller(self):
        pytest.importorskip("lz4.frame")
        msg = _large_request_dict()
        raw = Codec(CodecFormat.JSON, Compression.NONE).encode(msg)
        compressed = Codec(CodecFormat.JSON, Compression.LZ4).encode(msg)
        assert len(compressed) < len(raw)


# ---- MessagePack ----

class TestMsgpackCodec:

    def test_round_trip(self):
        pytest.importorskip("msgpack")
        codec = Codec(CodecFormat.MSGPACK, Compression.NONE)
        msg = _sample_request_dict()
        data = codec.encode(msg)
        result = codec.decode(data)
        assert result == msg

    def test_smaller_than_json(self):
        pytest.importorskip("msgpack")
        msg = _large_request_dict()
        json_data = Codec(CodecFormat.JSON).encode(msg)
        msgpack_data = Codec(CodecFormat.MSGPACK).encode(msg)
        assert len(msgpack_data) < len(json_data)

    def test_header_format_byte(self):
        pytest.importorskip("msgpack")
        codec = Codec(CodecFormat.MSGPACK)
        data = codec.encode({"test": 1})
        assert data[4] == 2  # MSGPACK


# ---- Protobuf stub ----

class TestProtobufStub:
    def test_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Protobuf"):
            Codec(CodecFormat.PROTOBUF)


# ---- Frame parsing ----

class TestFrameParsing:

    def test_short_frame_raises(self):
        with pytest.raises(ValueError, match="too short"):
            Codec._unframe(b"\x00\x00")

    def test_incomplete_frame_raises(self):
        codec = Codec()
        data = codec.encode({"test": 1})
        with pytest.raises(ValueError, match="Incomplete"):
            Codec._unframe(data[:10])

    def test_read_frame_length(self):
        codec = Codec()
        data = codec.encode({"test": 1})
        length = Codec.read_frame_length(data[:4])
        assert length == len(data)

    def test_multiple_frames(self):
        """Simulate two messages concatenated (TCP stream)."""
        codec = Codec()
        msg1 = {"id": 1}
        msg2 = {"id": 2}
        frame1 = codec.encode(msg1)
        frame2 = codec.encode(msg2)
        stream = frame1 + frame2

        # Parse first frame
        len1 = Codec.read_frame_length(stream[:4])
        result1 = codec.decode(stream[:len1])
        assert result1 == msg1

        # Parse second frame
        result2 = codec.decode(stream[len1:])
        assert result2 == msg2


# ---- Missing deps ----

class TestMissingDeps:

    def test_msgpack_not_installed(self):
        """If msgpack isn't installed, should get ImportError."""
        # Can't truly test this without mocking, just verify codec creation works when available
        try:
            import msgpack  # noqa: F401
            codec = Codec(CodecFormat.MSGPACK)
            assert codec.format == CodecFormat.MSGPACK
        except ImportError:
            with pytest.raises(ImportError, match="msgpack"):
                Codec(CodecFormat.MSGPACK)
