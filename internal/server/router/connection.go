package router

import (
	"context"
	"fmt"
	"sync"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/gin-gonic/gin"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type ConnectionRouter struct {
	watcher *connectionWatcher
}

func NewConnectionRouter(ctx context.Context, client client.WithWatch) (*ConnectionRouter, error) {
	watcher, err := newConnectionWatcher(ctx, client)
	if err != nil {
		return nil, fmt.Errorf("create connection watcher: %w", err)
	}
	return &ConnectionRouter{watcher: watcher}, nil
}

func (cr *ConnectionRouter) Get(ctx *gin.Context) {
	name := ctx.Query("name")
	namespace := ctx.Query("namespace")

	req := types.NamespacedName{Name: name, Namespace: namespace}
	conn := cr.watcher.get(ctx, req)
	if conn == nil {
		ctx.JSON(404, gin.H{"error": "connection not found"})
		return
	}

	if conn.Status.Phase == tfv1.WorkerRunning {
		ctx.String(200, conn.Status.ConnectionURL)
		return
	}

	// Subscribe to connection updates
	ch, cancelFunc := cr.watcher.subscribe(req)
	defer cancelFunc()

	// Wait for connection updates
	for conn := range ch {
		if conn.Status.Phase == tfv1.WorkerRunning {
			ctx.String(200, conn.Status.ConnectionURL)
			return
		}
	}
}

type connectionChannel chan *tfv1.TensorFusionConnection
type connectionSet map[connectionChannel]struct{}
type connectionSubscribers map[types.NamespacedName]connectionSet

type connectionWatcher struct {
	client client.WithWatch

	mu   sync.RWMutex
	subs connectionSubscribers
}

func newConnectionWatcher(ctx context.Context, client client.WithWatch) (*connectionWatcher, error) {
	cw := &connectionWatcher{
		client: client,
		subs:   make(connectionSubscribers),
	}
	watcher, err := cw.client.Watch(ctx, &tfv1.TensorFusionConnectionList{})
	if err != nil {
		return nil, fmt.Errorf("watch connections: %w", err)
	}
	go cw.watchConnections(ctx, watcher)
	return cw, nil
}

func (cw *connectionWatcher) get(ctx context.Context, req types.NamespacedName) *tfv1.TensorFusionConnection {
	conn := &tfv1.TensorFusionConnection{}
	if err := cw.client.Get(ctx, req, conn); err != nil {
		return nil
	}
	return conn
}

// Subscribe returns a channel that will be closed when the connection is deleted
func (cw *connectionWatcher) subscribe(req types.NamespacedName) (connectionChannel, func()) {
	ch := make(connectionChannel, 1)

	cw.mu.Lock()
	if _, exists := cw.subs[req]; !exists {
		cw.subs[req] = make(connectionSet)
	}
	cw.subs[req][ch] = struct{}{}
	cw.mu.Unlock()

	cancelFunc := func() {
		cw.mu.Lock()
		defer cw.mu.Unlock()

		if chans, exists := cw.subs[req]; exists {
			delete(chans, ch)
			close(ch)

			// If no more subscribers, remove the key
			if len(chans) == 0 {
				delete(cw.subs, req)
			}
		}
	}

	return ch, cancelFunc
}

func (cw *connectionWatcher) watchConnections(ctx context.Context, watcher watch.Interface) {
	// Watch for changes
	defer watcher.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case event, ok := <-watcher.ResultChan():
			if !ok {
				return
			}

			conn, ok := event.Object.(*tfv1.TensorFusionConnection)
			if !ok {
				continue
			}

			// Get the list of subscribers for this connection
			cw.mu.RLock()
			key := types.NamespacedName{Name: conn.Name, Namespace: conn.Namespace}
			if subscribers, exists := cw.subs[key]; exists {
				// Copy subscribers to avoid holding lock during channel send
				for ch := range subscribers {
					select {
					case ch <- conn:
					default:
						// Skip if channel is full
					}
				}
			}
			cw.mu.RUnlock()
		}
	}
}
