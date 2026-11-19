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

package peer

import (
	"context"
	"errors"
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvtier/protocol"
)

// ErrNoRemoteRDMA means the source cannot serve its slots for a remote
// read. The caller should fetch the blocks into its pool instead.
var ErrNoRemoteRDMA = errors.New("source does not serve RDMA")

// RemoteLease holds a source daemon's pins on blocks that a local engine
// reads straight out of the source's pool with RDMA. The source keeps the
// blocks pinned until Release, or until its own timeout passes.
type RemoteLease struct {
	Endpoint  string
	Status    []uint8 // protocol.Fetch* per requested key
	Slots     []int64 // source slot per key; -1 when absent
	Agent     string  // source helper's NIXL agent name
	AgentMeta []byte  // source helper's NIXL metadata
	Base      uint64  // address of the source pool's slot 0

	node *Node
	conn net.Conn
	once sync.Once
}

// LeaseRemote asks the source to pin keys and describe where they sit, so
// the caller can read them with RDMA. It does not touch the local pool.
func (n *Node) LeaseRemote(ctx context.Context, endpoint, ns string, keys [][]byte, slotBytes uint64,
) (*RemoteLease, error) {
	conn, err := n.get(ctx, endpoint)
	if err != nil {
		return nil, err
	}
	if err := conn.SetDeadline(time.Now().Add(n.opts.Timeout)); err != nil {
		n.put(endpoint, conn, false)
		return nil, err
	}
	req := &protocol.Message{
		ID: 1, Op: protocol.OpPeerGet, Namespace: ns, Keys: keys, SlotBytes: slotBytes, RDMA: true,
	}
	if err := writeFrame(conn, req); err != nil {
		n.put(endpoint, conn, false)
		return nil, err
	}
	resp, err := protocol.Read(conn)
	if err != nil {
		n.put(endpoint, conn, false)
		return nil, err
	}
	if resp.Error != "" {
		n.put(endpoint, conn, true)
		return nil, fmt.Errorf("source %s: %s", endpoint, resp.Error)
	}
	if !resp.RDMA {
		// The source is streaming bytes instead; this connection cannot be
		// reused mid-stream.
		n.put(endpoint, conn, false)
		return nil, ErrNoRemoteRDMA
	}
	if len(resp.Status) != len(keys) || len(resp.Slots) != len(keys) {
		n.put(endpoint, conn, false)
		return nil, fmt.Errorf("source %s returned %d statuses and %d slots for %d keys",
			endpoint, len(resp.Status), len(resp.Slots), len(keys))
	}
	return &RemoteLease{
		Endpoint: endpoint, Status: resp.Status, Slots: resp.Slots,
		Agent: resp.Agent, AgentMeta: resp.AgentMeta, Base: resp.Base,
		node: n, conn: conn,
	}, nil
}

// Release tells the source the read is over, so it drops its pins. A
// non-empty readErr tells it the read failed, so it keeps the blocks even
// on an evicting fetch. Release is safe to call more than once.
func (l *RemoteLease) Release(readErr string) error {
	var err error
	l.once.Do(func() {
		msg := &protocol.Message{ID: 2, Op: protocol.OpPeerRelease, Error: readErr}
		if err = l.conn.SetWriteDeadline(time.Now().Add(l.node.opts.Timeout)); err != nil {
			l.node.put(l.Endpoint, l.conn, false)
			return
		}
		err = writeFrame(l.conn, msg)
		l.node.put(l.Endpoint, l.conn, err == nil)
	})
	return err
}
