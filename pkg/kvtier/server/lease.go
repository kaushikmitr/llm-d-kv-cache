// Copyright 2026 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package server

import (
	"context"
	"errors"
	"time"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/peer"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/pool"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/protocol"
)

// leaseTTL bounds how long an engine may hold a source's pins before the
// daemon releases them on its behalf.
const leaseTTL = 30 * time.Second

type lease struct {
	remote *peer.RemoteLease
	owner  pool.SessionID
	timer  *time.Timer
}

// remoteGet leases a source's pins on req.Keys for an engine that reads the
// blocks itself, and answers with where they sit in the source's pool.
func (s *Server) remoteGet(ss *session, req, resp *protocol.Message) {
	if s.peer == nil {
		resp.Error = "peer transfer is disabled on this daemon"
		return
	}
	arena := s.store.Arena()
	if arena == nil {
		resp.Error = "pool not initialised"
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), leaseTTL)
	defer cancel()
	rl, err := s.peer.LeaseRemote(ctx, req.Endpoint, ss.ns, req.Keys, arena.SlotBytes())
	if err != nil {
		if errors.Is(err, peer.ErrNoRemoteRDMA) {
			resp.Error = "no_rdma"
		} else {
			resp.Error = err.Error()
		}
		return
	}
	s.leaseMu.Lock()
	s.nextLease++
	id := s.nextLease
	l := &lease{remote: rl, owner: ss.id}
	l.timer = time.AfterFunc(leaseTTL, func() { s.endLease(id, "lease expired") })
	s.leases[id] = l
	s.leaseMu.Unlock()

	resp.Lease = id
	resp.Status = rl.Status
	resp.Slots = rl.Slots
	resp.Agent = rl.Agent
	resp.AgentMeta = rl.AgentMeta
	resp.Base = rl.Base
	resp.SlotBytes = arena.SlotBytes()
}

// remoteRelease ends a lease. A non-empty req.Error reports a failed read.
func (s *Server) remoteRelease(ss *session, req, resp *protocol.Message) {
	s.leaseMu.Lock()
	l, ok := s.leases[req.Lease]
	s.leaseMu.Unlock()
	if !ok || l.owner != ss.id {
		resp.Error = "unknown lease"
		return
	}
	s.endLease(req.Lease, req.Error)
}

func (s *Server) endLease(id uint64, readErr string) {
	s.leaseMu.Lock()
	l, ok := s.leases[id]
	delete(s.leases, id)
	s.leaseMu.Unlock()
	if !ok {
		return
	}
	l.timer.Stop()
	if err := l.remote.Release(readErr); err != nil {
		s.log.Warn("releasing remote lease", "lease", id, "source", l.remote.Endpoint, "err", err)
	}
}

// releaseLeases ends every lease a disconnected session still holds.
func (s *Server) releaseLeases(owner pool.SessionID) {
	s.leaseMu.Lock()
	var ids []uint64
	for id, l := range s.leases {
		if l.owner == owner {
			ids = append(ids, id)
		}
	}
	s.leaseMu.Unlock()
	for _, id := range ids {
		s.endLease(id, "client disconnected")
	}
}
