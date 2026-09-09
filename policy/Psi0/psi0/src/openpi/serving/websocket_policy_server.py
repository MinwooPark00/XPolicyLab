import asyncio
import http
import logging
import time
import traceback
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)

INIT_BASE_HEIGHT = 0.74

class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

        self.torso_rpyh = np.array([0, 0, 0, INIT_BASE_HEIGHT], dtype=np.float32)
        self.serve_time = time.monotonic()

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                infer_time = time.monotonic()
                print(f"Received instruction: {obs['prompt']}")
                if infer_time - self.serve_time > 30 or obs.get("reset", False): # if idle more than 30s or reset flag is set, reset torso_rpyh
                    self.torso_rpyh = np.array([0, 0, 0, INIT_BASE_HEIGHT], dtype=np.float32)
                    print("Reset torso_rpyh to default.") 
                print(f"Torso rpyh: {self.torso_rpyh}")

                obs["states"] = np.concatenate([obs["states"], self.torso_rpyh], axis=0)

                action = self._policy.infer(obs)
                # print(f"Sending action: {action}")
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                
                self.torso_rpyh = action["actions"][-1, 28:32]

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time
                self.serve_time = time.monotonic()

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
