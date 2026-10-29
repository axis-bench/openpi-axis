import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy

# Reserved observation key carrying caller-supplied sampler noise. Defined here and imported by the
# server so the two halves cannot drift apart into a silent no-op.
NOISE_KEY = "__noise__"


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    # The server runs inference synchronously in its asyncio handler, so
                    # during a slow call (notably the first, which JIT-compiles the model
                    # -- 30-90s for pi0.5) it cannot answer keepalive pings. With the
                    # default 20s ping_timeout the client would drop the connection with
                    # "sent 1011 keepalive ping timeout" mid-inference. Disable
                    # client-initiated pings; a genuinely dead server is still detected
                    # via the socket closing.
                    ping_interval=None,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict, *, noise=None) -> Dict:  # noqa: UP006
        if noise is not None:
            # msgpack_numpy round-trips ndarrays, so the wire format needs no change. The reserved
            # key is popped by the server before the obs reaches any transform.
            obs = {**obs, NOISE_KEY: noise}
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
