package router

import (
	"context"
	"fmt"
	"net/http"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/portallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/gin-gonic/gin"
	v1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const assignPortTokenReviewName = "assign-host-port-token-review"

type AssignHostPortRouter struct {
	allocator *portallocator.PortAllocator
}

func NewAssignHostPortRouter(ctx context.Context, allocator *portallocator.PortAllocator) (*AssignHostPortRouter, error) {
	return &AssignHostPortRouter{allocator: allocator}, nil
}

func (r *AssignHostPortRouter) AssignHostPort(ctx *gin.Context) {
	podName := ctx.Query("podName")
	token := ctx.Request.Header.Get(constants.AuthorizationHeader)

	if token == "" {
		log.FromContext(ctx).Error(nil, "assigned host port failed, missing token", "podName", podName)
		ctx.String(http.StatusUnauthorized, "missing authorization header")
		return
	}
	tokenReview := &v1.TokenReview{
		ObjectMeta: metav1.ObjectMeta{
			Name: assignPortTokenReviewName,
		},
		Spec: v1.TokenReviewSpec{
			Token: token,
		},
	}
	if err := r.allocator.Client.Create(ctx, tokenReview); err != nil {
		log.FromContext(ctx).Error(err, "assigned host port failed, auth endpoint error", "podName", podName)
		ctx.String(http.StatusInternalServerError, "auth endpoint error")
		return
	}
	if !tokenReview.Status.Authenticated || tokenReview.Status.User.Username != utils.GetSelfServiceAccountNameFull() {
		log.FromContext(ctx).Error(nil, "assigned host port failed, token invalid", "podName", podName)
		ctx.String(http.StatusUnauthorized, "token authentication failed")
		return
	}

	port, err := r.allocator.AssignClusterLevelHostPort(podName)
	if err != nil {
		log.FromContext(ctx).Error(err, "assigned host port failed, port allocation failed", "podName", podName)
		ctx.String(http.StatusInternalServerError, err.Error())
		return
	}
	log.FromContext(ctx).Info("assigned host port successfully", "podName", podName, "port", port)
	ctx.String(http.StatusOK, fmt.Sprintf("%d", port))
}
