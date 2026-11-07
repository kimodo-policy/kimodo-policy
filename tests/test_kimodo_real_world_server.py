from __future__ import annotations

import base64
import importlib.util
import json
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path

import numpy as np


SERVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "real-world"
    / "deploy"
    / "real_world_server.py"
)
SPEC = importlib.util.spec_from_file_location("real_world_server", SERVER_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _FakeRuntime:
    control_fps = 50.0

    def __init__(self):
        self.reset_calls = []
        self.infer_calls = 0
        self.session_ids = []

    def reset(self, seed=None, session_id=None):
        self.reset_calls.append(seed)
        self.session_ids.append(session_id)

    def infer(self, payload):
        # Validation now belongs to the runtime, including non-HTTP callers.
        MODULE._validate_infer_payload(payload)
        self.infer_calls += 1
        action = np.zeros((3, 40), dtype=np.float32)
        action[:, 2] = 0.85
        action[:, 3:9] = [1, 0, 0, 1, 0, 0]
        return action


def _request(connection: HTTPConnection, method: str, path: str, payload=None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if body is not None else {}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    data = json.loads(response.read().decode("utf-8"))
    return response.status, data


class RealWorldServerTest(unittest.TestCase):
    def setUp(self):
        self.runtime = _FakeRuntime()
        handler = type("TestHandler", (MODULE.RequestHandler,), {})
        handler.runtime = self.runtime
        self.server = MODULE.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)

    def tearDown(self):
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    @staticmethod
    def _payload():
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        return {
            "task": "pick and place",
            "observation": {
                "images": {
                    "front": {
                        "shape": list(image.shape),
                        "dtype": "uint8",
                        "data_b64": base64.b64encode(image.tobytes()).decode("ascii"),
                    }
                },
                "state": np.zeros(64, dtype=np.float32).tolist(),
                "state_history": np.zeros((2, 64), dtype=np.float32).tolist(),
            },
        }

    def test_health_reset_and_infer_contract(self):
        status, health = _request(self.connection, "GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["mode"], "real_world")
        self.assertEqual(health["control_fps"], 50.0)
        status, result = _request(self.connection, "POST", "/reset", {"seed": 7, "session_id": "test-session"})
        self.assertEqual(status, 200)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(self.runtime.reset_calls, [7])
        self.assertEqual(self.runtime.session_ids, ["test-session"])
        status, result = _request(self.connection, "POST", "/infer", self._payload())
        self.assertEqual(status, 200)
        self.assertEqual(np.asarray(result["action_chunk"]).shape, (3, 40))
        self.assertEqual(self.runtime.infer_calls, 1)

    def test_invalid_payload_is_client_error(self):
        status, result = _request(self.connection, "POST", "/infer", {"task": "x"})
        self.assertEqual(status, 400)
        self.assertIn("observation", result["error"])
        self.assertEqual(self.runtime.infer_calls, 0)


if __name__ == "__main__":
    unittest.main()
