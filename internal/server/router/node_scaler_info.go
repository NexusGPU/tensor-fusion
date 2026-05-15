package router

import (
	"context"
	"net/http"

	"github.com/NexusGPU/tensor-fusion/internal/scheduler/expander"
	"github.com/gin-gonic/gin"
)

type NodeScalerInfoRouter struct {
	nodeExpander *expander.NodeExpander
}

func NewNodeScalerInfoRouter(
	ctx context.Context,
	nodeExpander *expander.NodeExpander,
) (*NodeScalerInfoRouter, error) {
	return &NodeScalerInfoRouter{nodeExpander: nodeExpander}, nil
}

func (r *NodeScalerInfoRouter) Get(ctx *gin.Context) {
	if r.nodeExpander == nil {
		ctx.JSON(http.StatusServiceUnavailable, gin.H{
			"error": "node scaler not enabled",
		})
		return
	}
	ctx.JSON(http.StatusOK, gin.H{
		"data": r.nodeExpander.GetNodeScalerInfo(),
	})
}
