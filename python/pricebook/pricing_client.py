"""Pricing client: synchronous TCP client for the pricing server.

    from pricebook.pricing_client import PricingClient

    client = PricingClient("127.0.0.1", 9090)
    client.connect()
    response = client.price(request)
    client.close()

Handles framing, codec, and connection lifecycle.
"""

from __future__ import annotations

import socket
import struct
from typing import Any

from pricebook.pricing_codec import Codec, CodecFormat, Compression, HEADER_SIZE
from pricebook.pricing_schema import PricingRequest, PricingResponse


class PricingClient:
    """Synchronous TCP client for the pricing server.

    Args:
        host: server address.
        port: server port.
        codec: Codec instance (must match server's codec).
        timeout: socket timeout in seconds.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9090,
        codec: Codec | None = None,
        timeout: float = 30.0,
    ):
        self.host = host
        self.port = port
        self.codec = codec or Codec()
        self.timeout = timeout
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        """Connect to the pricing server."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))

    def close(self) -> None:
        """Close the connection."""
        if self._sock:
            self._sock.close()
            self._sock = None

    def price(self, request: PricingRequest) -> PricingResponse:
        """Send a pricing request and receive the response."""
        if self._sock is None:
            raise ConnectionError("Not connected. Call connect() first.")

        # Encode and send
        data = self.codec.encode(request.to_dict())
        self._sock.sendall(data)

        # Receive response: first read 4 bytes for length
        length_bytes = self._recv_exact(4)
        total_len = Codec.read_frame_length(length_bytes)

        # Read the rest
        remaining = self._recv_exact(total_len - 4)
        frame = length_bytes + remaining

        # Decode
        msg = self.codec.decode(frame)
        return PricingResponse.from_dict(msg)

    def price_dict(self, request_dict: dict) -> dict:
        """Send a raw dict request and receive a raw dict response."""
        if self._sock is None:
            raise ConnectionError("Not connected. Call connect() first.")

        data = self.codec.encode(request_dict)
        self._sock.sendall(data)

        length_bytes = self._recv_exact(4)
        total_len = Codec.read_frame_length(length_bytes)
        remaining = self._recv_exact(total_len - 4)
        frame = length_bytes + remaining

        return self.codec.decode(frame)

    def _recv_exact(self, n: int) -> bytes:
        """Receive exactly n bytes from the socket."""
        data = b""
        while len(data) < n:
            chunk = self._sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed by server")
            data += chunk
        return data

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
