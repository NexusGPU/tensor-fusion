package router

import (
	"context"
	"fmt"
	"net/http"

	"github.com/NexusGPU/tensor-fusion/internal/constants"
	"github.com/NexusGPU/tensor-fusion/internal/indexallocator"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
	"github.com/gin-gonic/gin"
	v1 "k8s.io/api/authentication/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const assignIndexTokenReviewName = "assign-index-token-review"

type AssignIndexRouter struct {
	allocator *indexallocator.IndexAllocator
}

func NewAssignIndexRouter(ctx context.Context, allocator *indexallocator.IndexAllocator) (*AssignIndexRouter, error) {
	return &AssignIndexRouter{allocator: allocator}, nil
}

func (r *AssignIndexRouter) AssignIndex(ctx *gin.Context) {
	podName := ctx.Query("podName")
	token := ctx.Request.Header.Get(constants.AuthorizationHeader)

	if token == "" {
		log.FromContext(ctx).Error(nil, "assigned index failed, missing token", "podName", podName)
		ctx.String(http.StatusUnauthorized, "missing authorization header")
		return
	}
	tokenReview := &v1.TokenReview{
		ObjectMeta: metav1.ObjectMeta{
			Name: assignIndexTokenReviewName,
		},
		Spec: v1.TokenReviewSpec{
			Token: token,
		},
	}
	if err := r.allocator.Client.Create(ctx, tokenReview); err != nil {
		log.FromContext(ctx).Error(err, "assigned index failed, auth endpoint error", "podName", podName)
		ctx.String(http.StatusInternalServerError, "auth endpoint error")
		return
	}
	if !tokenReview.Status.Authenticated || tokenReview.Status.User.Username != utils.GetSelfServiceAccountNameFull() {
		log.FromContext(ctx).Error(nil, "assigned index failed, token invalid", "podName", podName)
		ctx.String(http.StatusUnauthorized, "token authentication failed")
		return
	}

	index, err := r.allocator.AssignIndex(podName)
	if err != nil {
		log.FromContext(ctx).Error(err, "assigned index failed", "podName", podName)
		ctx.String(http.StatusInternalServerError, err.Error())
		return
	}
	log.FromContext(ctx).Info("assigned index successfully", "podName", podName, "index", index)
	ctx.String(http.StatusOK, fmt.Sprintf("%d", index))
}

