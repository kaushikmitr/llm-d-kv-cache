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
"""Zero-copy peer reads into the engine's own CPU tier.

The local daemon leases the source daemon's pins on the blocks and returns
where they sit in the source's pool. This reader then does an RDMA READ
from those pool slots straight into the engine's CPU-tier chunks, with
NIXL, from inside the engine process. The blocks never pass through the
local pool, so there is no second copy.
"""

from __future__ import annotations

import base64
import logging
import os
import socket
import threading
import time
from collections.abc import Sequence

from llmd_kv_tiering.client import FETCH_OK, DaemonClient, DaemonError
from llmd_kv_tiering.rdma_helper import coalesce, split_runs

logger = logging.getLogger("vllm.llmd_kv_tiering.zero_copy")


class ZeroCopyReader:
    """Reads peer pool slots into a registered local memory region."""

    def __init__(self, region_base: int, region_size: int, block_bytes: int, parallel: int, timeout_s: float):
        from nixl._api import nixl_agent, nixl_agent_config

        self._base = region_base
        self._size = region_size
        self._block = block_bytes
        self._parallel = max(1, parallel)
        self._timeout = timeout_s
        self._lock = threading.Lock()
        self._peers: dict[str, str] = {}  # source agent name -> registered name
        name = f"vllm-{socket.gethostname()}-{os.getpid()}"
        self._agent = nixl_agent(name, nixl_agent_config(backends=["UCX"]))
        self._reg = self._agent.register_memory([(region_base, region_size, 0, "")], "DRAM")
        logger.info("zero-copy reader %s: registered %d bytes at 0x%x", name, region_size, region_base)

    def read(
        self,
        client: DaemonClient,
        endpoint: str,
        keys: Sequence[bytes],
        chunk_ids: Sequence[int],
    ) -> tuple[list[bytes], list[bytes]]:
        """Read keys from the source at endpoint into the given chunks.

        Returns (loaded, missing). Raises DaemonError when the source cannot
        serve a remote read, so the caller can fall back to a pooled fetch.
        """
        resp = client.call("remote_get", keys=list(keys), endpoint=endpoint)
        lease = resp["lease"]
        read_error = ""
        loaded: list[bytes] = []
        missing: list[bytes] = []
        try:
            if resp.get("slot_bytes") != self._block:
                raise DaemonError(f"pool slot {resp.get('slot_bytes')} bytes, engine block {self._block}")
            pairs: list[tuple[int, int]] = []
            for key, cid, st, slot in zip(keys, chunk_ids, resp["status"], resp["slots"]):
                if st != FETCH_OK or slot < 0:
                    missing.append(key)
                    continue
                if not 0 <= cid * self._block < self._size:
                    raise DaemonError(f"chunk {cid} outside the registered region")
                pairs.append((int(cid), int(slot)))
                loaded.append(key)
            if pairs:
                self._transfer(resp["agent"], resp["agent_meta"], int(resp["base"]), pairs)
        except Exception as e:
            read_error = str(e) or type(e).__name__
            raise
        finally:
            try:
                client.call("remote_release", lease=lease, error=read_error)
            except DaemonError as e:
                logger.warning("releasing lease %d: %s", lease, e)
        return loaded, missing

    def _transfer(self, agent: str, agent_meta: bytes, remote_base: int, pairs: list[tuple[int, int]]) -> None:
        runs = coalesce(pairs, self._block)
        with self._lock:
            peer = self._peers.get(agent)
            if peer is None:
                meta = agent_meta if isinstance(agent_meta, bytes) else base64.b64decode(agent_meta)
                added = self._agent.add_remote_agent(meta)
                peer = added.decode() if isinstance(added, bytes) else added
                self._peers[agent] = peer
            handles = []
            try:
                for group in split_runs(runs, self._parallel):
                    local = self._agent.get_xfer_descs([(self._base + lo, n, 0) for lo, _ro, n in group], "DRAM")
                    remote = self._agent.get_xfer_descs([(remote_base + ro, n, 0) for _lo, ro, n in group], "DRAM")
                    handle = self._agent.initialize_xfer("READ", local, remote, peer)
                    handles.append(handle)
                    if self._agent.transfer(handle) == "ERR":
                        raise DaemonError("RDMA read failed to start")
                deadline = time.monotonic() + self._timeout
                pending = list(handles)
                while pending:
                    if time.monotonic() > deadline:
                        raise DaemonError("RDMA read timed out")
                    still = []
                    for handle in pending:
                        state = self._agent.check_xfer_state(handle)
                        if state == "ERR":
                            raise DaemonError("RDMA read failed")
                        if state != "DONE":
                            still.append(handle)
                    pending = still
            finally:
                for handle in handles:
                    self._agent.release_xfer_handle(handle)
