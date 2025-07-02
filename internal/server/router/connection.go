package router

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/gin-gonic/gin"
	v1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/cache"
	"k8s.io/apimachinery/pkg/watch"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const JWTTokenCacheDuration = time.Minute * 30
const JWTTokenCacheSize = 20000
const BearerPrefix = "Bearer "

type ConnectionRouter struct {
	watcher  *connectionWatcher
	lruCache *cache.LRUExpireCache

	client client.WithWatch
}

func NewConnectionRouter(ctx context.Context, client client.WithWatch) (*ConnectionRouter, error) {
	watcher, err := newConnectionWatcher(ctx, client)
	if err != nil {
		return nil, fmt.Errorf("create connection watcher: %w", err)
	}
	return &ConnectionRouter{
		watcher:  watcher,
		client:   client,
		lruCache: cache.NewLRUExpireCache(JWTTokenCacheSize),
	}, nil
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

	if os.Getenv(constants.DisableConnectionAuthEnv) != constants.TrueStringValue {
		if !cr.authenticatePodConnection(ctx, conn) {
			ctx.JSON(401, gin.H{"error": "unauthorized"})
			return
		}
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
	watcherChan := watcher.ResultChan()
	for {

		select {
		case <-ctx.Done():
			return
		case event, ok := <-watcherChan:
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

func (cr *ConnectionRouter) authenticatePodConnection(ctx *gin.Context, conn *tfv1.TensorFusionConnection) bool {
	if len(conn.OwnerReferences) == 0 {
		log.FromContext(ctx).Error(nil, "connection owner reference is empty")
		return false
	}
	token := ctx.GetHeader(constants.AuthorizationHeader)

	if token == "" || !strings.HasPrefix(token, BearerPrefix) {
		return false
	}
	token = token[len(BearerPrefix):]

	// use cache to avoid repeated token review and unnecessary API calls to API server
	if value, exists := cr.lruCache.Get(token); exists {
		if value.(bool) {
			log.FromContext(ctx).Info("token authentication successful for connection from cache", "connection", conn.Name)
			return true
		}
		log.FromContext(ctx).Info("token authentication failed for connection from cache", "connection", conn.Name)
		return false
	}

	tokenReview := &v1.TokenReview{
		ObjectMeta: metav1.ObjectMeta{
			Name: conn.Name,
		},
		Spec: v1.TokenReviewSpec{
			Token: token,
		},
	}
	if err := cr.client.Create(ctx, tokenReview); err != nil {
		log.FromContext(ctx).Error(err, "token authentication failed, auth endpoint error", "connection", conn.Name)
		return false
	}
	if !tokenReview.Status.Authenticated {
		cr.lruCache.Add(token, false, JWTTokenCacheDuration)
		log.FromContext(ctx).Error(nil, "token authentication failed, invalid token", "connection", conn.Name)
		return false
	}
	if tokenReview.Status.User.Extra == nil ||
		len(tokenReview.Status.User.Extra[constants.ExtraVerificationInfoPodIDKey]) == 0 {
		cr.lruCache.Add(token, false, JWTTokenCacheDuration)
		log.FromContext(ctx).Error(nil, "token authentication failed, no valid pod UID in token", "connection", conn.Name)
		return false
	}
	// verified pod ID, the connection is valid
	if string(conn.OwnerReferences[0].UID) == tokenReview.Status.User.Extra[constants.ExtraVerificationInfoPodIDKey][0] {
		cr.lruCache.Add(token, true, JWTTokenCacheDuration)
		log.FromContext(ctx).Info("token authentication successful for connection", "connection", conn.Name)
		return true
	}

	cr.lruCache.Add(token, false, JWTTokenCacheDuration)
	log.FromContext(ctx).Error(nil, "token authentication failed, pod ID not matched", "connection", conn.Name)
	return false
}
