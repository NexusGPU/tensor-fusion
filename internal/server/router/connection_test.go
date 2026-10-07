package router

import (
	"context"
	"net/http/httptest"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/gin-gonic/gin"
	authv1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestConnectionAuthCacheIsScopedToPodOwner(t *testing.T) {
	for _, firstOwner := range []types.UID{"pod-a", "pod-b"} {
		t.Run(string(firstOwner), func(t *testing.T) {
			scheme := runtime.NewScheme()
			if err := authv1.AddToScheme(scheme); err != nil {
				t.Fatal(err)
			}
			reviews := 0
			c := fake.NewClientBuilder().WithScheme(scheme).WithInterceptorFuncs(interceptor.Funcs{
				Create: func(_ context.Context, _ client.WithWatch, obj client.Object, _ ...client.CreateOption) error {
					reviews++
					review := obj.(*authv1.TokenReview)
					review.Status.Authenticated = true
					review.Status.User.Extra = map[string]authv1.ExtraValue{
						constants.ExtraVerificationInfoPodIDKey: {"pod-a"},
					}
					return nil
				},
			}).Build()
			router := &ConnectionRouter{client: c, lruCache: cache.NewLRUExpireCache(10)}
			authenticate := func(owner types.UID) bool {
				ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
				ctx.Request = httptest.NewRequest("GET", "/api/connection", nil)
				ctx.Request.Header.Set(constants.AuthorizationHeader, "Bearer token-for-pod-a")
				return router.authenticatePodConnection(ctx, &tfv1.TensorFusionConnection{
					ObjectMeta: metav1.ObjectMeta{OwnerReferences: []metav1.OwnerReference{{UID: owner}}},
				})
			}
			if got := authenticate(firstOwner); got != (firstOwner == "pod-a") {
				t.Fatalf("unexpected initial authentication result: %v", got)
			}
			if !authenticate("pod-a") {
				t.Fatal("a denied request for another owner poisoned the valid owner's cache")
			}
			if authenticate("pod-b") {
				t.Fatal("cached token for pod-a authorized a connection owned by pod-b")
			}
			if !authenticate("pod-a") || reviews > 2 {
				t.Fatalf("same-owner authentication should reuse its result, reviews=%d", reviews)
			}
		})
	}
}

func TestConnectionWatchRechecksOwnerBeforeReturningURL(t *testing.T) {
	t.Setenv(constants.DisableConnectionAuthEnv, "false")
	scheme := runtime.NewScheme()
	if err := tfv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := authv1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	initial := &tfv1.TensorFusionConnection{ObjectMeta: metav1.ObjectMeta{
		Name: "conn", Namespace: "ns", OwnerReferences: []metav1.OwnerReference{{UID: "pod-a"}},
	}}
	authenticated := make(chan struct{}, 1)
	c := fake.NewClientBuilder().WithScheme(scheme).WithObjects(initial).WithInterceptorFuncs(interceptor.Funcs{
		Create: func(_ context.Context, _ client.WithWatch, obj client.Object, _ ...client.CreateOption) error {
			review := obj.(*authv1.TokenReview)
			review.Status.Authenticated = true
			review.Status.User.Extra = map[string]authv1.ExtraValue{constants.ExtraVerificationInfoPodIDKey: {"pod-a"}}
			select {
			case authenticated <- struct{}{}:
			default:
			}
			return nil
		},
	}).Build()
	watcher := &connectionWatcher{client: c, subs: make(connectionSubscribers)}
	router := &ConnectionRouter{client: c, watcher: watcher, lruCache: cache.NewLRUExpireCache(10)}
	response := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(response)
	ctx.Request = httptest.NewRequest("GET", "/api/connection?name=conn&namespace=ns", nil)
	ctx.Request.Header.Set(constants.AuthorizationHeader, "Bearer token-for-pod-a")
	done := make(chan struct{})
	go func() { defer close(done); router.Get(ctx) }()
	select {
	case <-authenticated:
	case <-time.After(time.Second):
		t.Fatal("initial authentication did not finish")
	}
	replacement := initial.DeepCopy()
	replacement.OwnerReferences[0].UID = "pod-b"
	replacement.Status.Phase = tfv1.WorkerRunning
	replacement.Status.ConnectionURL = "private-worker-b"
	watcher.mu.RLock()
	for ch := range watcher.subs[types.NamespacedName{Name: "conn", Namespace: "ns"}] {
		ch <- replacement
	}
	watcher.mu.RUnlock()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("connection request did not finish")
	}
	if response.Code != 401 {
		t.Fatalf("changed owner must be rejected, got %d: %s", response.Code, response.Body.String())
	}
}
