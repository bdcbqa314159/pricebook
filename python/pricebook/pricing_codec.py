"""Codec layer: encode/decode + compression + wire framing.

Pluggable format (JSON now, MessagePack/Protobuf later) with optional
compression. Wire frame: [4-byte length][1-byte format][1-byte compression][payload].

    from pricebook.pricing_codec import Codec, CodecFormat, Compression

    codec = Codec(format=CodecFormat.JSON, compression=Compression.NONE)
    data = codec.encode(request.to_dict())   # → bytes (framed)
    msg = codec.decode(data)                 # → dict

For socket transport, use frame/unframe for TCP stream framing:
    framed = codec.frame(payload)            # → bytes with length prefix
    payload = codec.unframe(framed)          # → raw payload bytes
"""

from __future__ import annotations

import json
import struct
from enum import IntEnum
from typing import Any


class CodecFormat(IntEnum):
    """Wire format for payload encoding."""
    JSON = 1
    MSGPACK = 2
    PROTOBUF = 3  # future


class Compression(IntEnum):
    """Compression algorithm for payload."""
    NONE = 0
    ZSTD = 1
    LZ4 = 2


# Header: [4 bytes length][1 byte format][1 byte compression]
HEADER_SIZE = 6
HEADER_STRUCT = struct.Struct(">IBB")  # big-endian: uint32 + uint8 + uint8


def _has_msgpack() -> bool:
    try:
        import msgpack  # noqa: F401
        return True
    except ImportError:
        return False


def _has_zstd() -> bool:
    try:
        import zstandard  # noqa: F401
        return True
    except ImportError:
        return False


def _has_lz4() -> bool:
    try:
        import lz4.frame  # noqa: F401
        return True
    except ImportError:
        return False


class Codec:
    """Encode/decode messages with pluggable format and compression.

    The codec handles three concerns:
    1. Serialization: dict ↔ bytes (JSON or MessagePack)
    2. Compression: raw bytes ↔ compressed bytes (optional)
    3. Framing: length-prefixed for TCP stream transport

    Args:
        format: CodecFormat.JSON (default) or CodecFormat.MSGPACK.
        compression: Compression.NONE (default), ZSTD, or LZ4.
    """

    def __init__(
        self,
        format: CodecFormat = CodecFormat.JSON,
        compression: Compression = Compression.NONE,
    ):
        if format == CodecFormat.MSGPACK and not _has_msgpack():
            raise ImportError("msgpack not installed: pip install msgpack")
        if format == CodecFormat.PROTOBUF:
            raise NotImplementedError("Protobuf codec not yet implemented")
        if compression == Compression.ZSTD and not _has_zstd():
            raise ImportError("zstandard not installed: pip install zstandard")
        if compression == Compression.LZ4 and not _has_lz4():
            raise ImportError("lz4 not installed: pip install lz4")

        self.format = format
        self.compression = compression

    def encode(self, msg: dict[str, Any]) -> bytes:
        """Encode a dict to framed bytes: [header][compressed_payload]."""
        payload = self._serialize(msg)
        compressed = self._compress(payload)
        return self._frame(compressed)

    def decode(self, data: bytes) -> dict[str, Any]:
        """Decode framed bytes back to a dict."""
        fmt, comp, payload = self._unframe(data)
        decompressed = self._decompress(payload, comp)
        return self._deserialize(decompressed, fmt)

    def encode_raw(self, msg: dict[str, Any]) -> bytes:
        """Encode without framing (for embedding in other protocols)."""
        payload = self._serialize(msg)
        return self._compress(payload)

    def decode_raw(self, data: bytes) -> dict[str, Any]:
        """Decode without framing."""
        decompressed = self._decompress(data, self.compression)
        return self._deserialize(decompressed, self.format)

    # ---- Framing ----

    def _frame(self, payload: bytes) -> bytes:
        """Add length-prefix header: [4B length][1B format][1B compression][payload].

        Length includes header (6 bytes) + payload.
        """
        total_len = HEADER_SIZE + len(payload)
        header = HEADER_STRUCT.pack(total_len, int(self.format), int(self.compression))
        return header + payload

    @staticmethod
    def _unframe(data: bytes) -> tuple[CodecFormat, Compression, bytes]:
        """Extract header and payload from framed data."""
        if len(data) < HEADER_SIZE:
            raise ValueError(f"Frame too short: {len(data)} < {HEADER_SIZE}")
        total_len, fmt_byte, comp_byte = HEADER_STRUCT.unpack(data[:HEADER_SIZE])
        if len(data) < total_len:
            raise ValueError(f"Incomplete frame: got {len(data)}, expected {total_len}")
        payload = data[HEADER_SIZE:total_len]
        return CodecFormat(fmt_byte), Compression(comp_byte), payload

    @staticmethod
    def read_frame_length(header_bytes: bytes) -> int:
        """Read total frame length from the first 4 bytes (for streaming)."""
        if len(header_bytes) < 4:
            raise ValueError("Need at least 4 bytes for length")
        return struct.unpack(">I", header_bytes[:4])[0]

    # ---- Serialization ----

    def _serialize(self, msg: dict) -> bytes:
        if self.format == CodecFormat.JSON:
            return json.dumps(msg, separators=(",", ":"), sort_keys=False).encode("utf-8")
        elif self.format == CodecFormat.MSGPACK:
            import msgpack
            return msgpack.packb(msg, use_bin_type=True)
        raise ValueError(f"Unsupported format: {self.format}")

    @staticmethod
    def _deserialize(data: bytes, fmt: CodecFormat) -> dict:
        if fmt == CodecFormat.JSON:
            return json.loads(data.decode("utf-8"))
        elif fmt == CodecFormat.MSGPACK:
            import msgpack
            return msgpack.unpackb(data, raw=False)
        raise ValueError(f"Unsupported format: {fmt}")

    # ---- Compression ----

    def _compress(self, data: bytes) -> bytes:
        if self.compression == Compression.NONE:
            return data
        elif self.compression == Compression.ZSTD:
            import zstandard
            return zstandard.ZstdCompressor(level=3).compress(data)
        elif self.compression == Compression.LZ4:
            import lz4.frame
            return lz4.frame.compress(data)
        raise ValueError(f"Unsupported compression: {self.compression}")

    @staticmethod
    def _decompress(data: bytes, comp: Compression) -> bytes:
        if comp == Compression.NONE:
            return data
        elif comp == Compression.ZSTD:
            import zstandard
            return zstandard.ZstdDecompressor().decompress(data)
        elif comp == Compression.LZ4:
            import lz4.frame
            return lz4.frame.decompress(data)
        raise ValueError(f"Unsupported compression: {comp}")
