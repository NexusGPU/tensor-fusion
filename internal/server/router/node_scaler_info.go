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
	ctx.JSON(http.StatusOK, gin.H{
		"data": r.nodeExpander.GetNodeScalerInfo(),
	})
}
