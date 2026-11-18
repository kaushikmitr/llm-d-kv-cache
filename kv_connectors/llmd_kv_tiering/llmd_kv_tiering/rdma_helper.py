# Copyright 2026 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RDMA helper for kvtierd.

Runs next to the daemon, one per node. It attaches to the daemon's pool,
maps it, and registers it with NIXL. The daemon then asks it, over a Unix
socket, to read blocks from a peer's pool straight into local pool slots
with RDMA. Peers exchange NIXL metadata through the daemons' own peer
protocol, so the helper opens no network port of its own.

Requests and responses are JSON lines:

    {"op": "info"}
        -> {"name", "metadata" (base64), "base", "slot_bytes"}
    {"op": "read", "peer", "peer_metadata" (base64), "remote_base",
     "slot_bytes", "pairs": [[local_slot, remote_slot], ...]}
        -> {} on success, {"error": "..."} on failure

Run it as: python -m llmd_kv_tiering.rdma_helper --daemon-socket S --listen-socket L
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import socket
import threading
import time

from llmd_kv_tiering.client import DaemonClient, DaemonError

logger = logging.getLogger("llmd_kv_tiering.rdma_helper")

HELPER_NAMESPACE = "__rdma_helper__"


def coalesce(pairs: list[tuple[int, int]], slot_bytes: int) -> list[tuple[int, int, int]]:
    """Merge runs where both the local and the remote slot advance by one.

    Returns (local_offset, remote_offset, length) in bytes. Fewer, larger
    descriptors make for fewer RDMA work requests.
    """
    out: list[tuple[int, int, int]] = []
    for local, remote in sorted(pairs):
        lo, ro = local * slot_bytes, remote * slot_bytes
        if out and out[-1][0] + out[-1][2] == lo and out[-1][1] + out[-1][2] == ro:
            prev = out[-1]
            out[-1] = (prev[0], prev[1], prev[2] + slot_bytes)
        else:
            out.append((lo, ro, slot_bytes))
    return out


class Helper:
    def __init__(self, daemon_socket: str, name: str, backends: list[str], timeout_s: float) -> None:
        self.name = name
        self.timeout = timeout_s
        self._lock = threading.Lock()
        self._known_peers: set[str] = set()
        self.client = self._attach(daemon_socket)
        arena = self.client.arena
        self.base, self.size, self.slot_bytes = arena.base, arena.size, arena.slot_bytes

        from nixl._api import nixl_agent, nixl_agent_config

        self.agent = nixl_agent(name, nixl_agent_config(backends=backends))
        self._reg = self.agent.register_memory([(self.base, self.size, 0, "")], "DRAM")
        self.metadata = self.agent.get_agent_metadata()
        logger.info(
            "RDMA helper %s: registered %d bytes of pool at 0x%x (slot %d bytes) with %s",
            name,
            self.size,
            self.base,
            self.slot_bytes,
            backends,
        )

    def _attach(self, daemon_socket: str) -> DaemonClient:
        # The pool exists once the first engine has connected; wait for it.
        while True:
            try:
                return DaemonClient(
                    daemon_socket,
                    HELPER_NAMESPACE,
                    0,
                    f"rdma-helper/{self.name}",
                    on_event=lambda _msg: None,
                    on_disconnect=self._lost,
                    timeout_s=self.timeout,
                )
            except (OSError, DaemonError) as e:
                logger.info("waiting for the daemon's pool: %s", e)
                time.sleep(2)

    def _lost(self) -> None:
        # The pool mapping stays valid, but a restarted daemon has a new pool.
        logger.error("lost the daemon; exiting so the helper restarts with it")
        os._exit(1)

    def info(self) -> dict:
        return {
            "name": self.name,
            "metadata": base64.b64encode(self.metadata).decode(),
            "base": self.base,
            "slot_bytes": self.slot_bytes,
        }

    def read(self, req: dict) -> dict:
        slot_bytes = int(req["slot_bytes"])
        if slot_bytes != self.slot_bytes:
            return {"error": f"slot size {slot_bytes} does not match local {self.slot_bytes}"}
        peer = req["peer"]
        remote_base = int(req["remote_base"])
        runs = coalesce([(int(local), int(remote)) for local, remote in req["pairs"]], slot_bytes)
        for lo, _ro, n in runs:
            if lo + n > self.size:
                return {"error": "local slot outside the pool"}
        with self._lock:
            if peer not in self._known_peers:
                added = self.agent.add_remote_agent(base64.b64decode(req["peer_metadata"]))
                name = added.decode() if isinstance(added, bytes) else added
                self._known_peers.add(name)
            local = self.agent.get_xfer_descs([(self.base + lo, n, 0) for lo, _ro, n in runs], "DRAM")
            remote = self.agent.get_xfer_descs([(remote_base + ro, n, 0) for _lo, ro, n in runs], "DRAM")
            handle = self.agent.initialize_xfer("READ", local, remote, peer)
            try:
                state = self.agent.transfer(handle)
                deadline = time.monotonic() + self.timeout
                while state not in ("DONE", "ERR"):
                    if time.monotonic() > deadline:
                        return {"error": "RDMA read timed out"}
                    state = self.agent.check_xfer_state(handle)
                if state == "ERR":
                    return {"error": "RDMA read failed"}
            finally:
                self.agent.release_xfer_handle(handle)
        return {}


def serve(helper: Helper, path: str) -> None:
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    os.chmod(path, 0o660)
    srv.listen(16)
    logger.info("RDMA helper listening on %s", path)

    def handle(conn: socket.socket) -> None:
        with conn, conn.makefile("rwb") as f:
            for line in f:
                try:
                    req = json.loads(line)
                    op = req.get("op")
                    if op == "info":
                        resp = helper.info()
                    elif op == "read":
                        resp = helper.read(req)
                    else:
                        resp = {"error": f"unknown op {op!r}"}
                except Exception as e:  # noqa: BLE001 - every failure becomes a response
                    logger.exception("request failed")
                    resp = {"error": str(e)}
                f.write(json.dumps(resp).encode() + b"\n")
                f.flush()

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--daemon-socket", required=True)
    ap.add_argument("--listen-socket", required=True)
    ap.add_argument("--name", default=f"kvtier-{socket.gethostname()}")
    ap.add_argument("--backends", default="UCX", help="comma-separated NIXL backends")
    ap.add_argument("--timeout-s", type=float, default=30.0)
    ap.add_argument("--log-level", default="INFO")
    a = ap.parse_args()
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    helper = Helper(a.daemon_socket, a.name, a.backends.split(","), a.timeout_s)
    serve(helper, a.listen_socket)


if __name__ == "__main__":
    main()
