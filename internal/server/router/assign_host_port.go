package router

import (
	"context"
	"fmt"
	"net/http"

	"github.com/NexusGPU/tensor-fusion/internal/portallocator"
	"github.com/gin-gonic/gin"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type AssignHostPortRouter struct {
	allocator *portallocator.PortAllocator
}

func NewAssignHostPortRouter(ctx context.Context, allocator *portallocator.PortAllocator) (*AssignHostPortRouter, error) {
	return &AssignHostPortRouter{allocator: allocator}, nil
}

func (r *AssignHostPortRouter) AssignHostPort(ctx *gin.Context) {
	// TODO verify service account token, issuer must be the same as current instance
	// namely the request must comes from peer operator Pod

	podName := ctx.Query("podName")
	port, err := r.allocator.AssignClusterLevelHostPort(podName)
	if err != nil {
		ctx.String(http.StatusInternalServerError, err.Error())
		return
	}
	log.FromContext(ctx).Info("assigned host port", "podName", podName, "port", port)
	ctx.String(http.StatusOK, fmt.Sprintf("%d", port))
}
