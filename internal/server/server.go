package server

import (
	"net/http"

	"github.com/NexusGPU/tensor-fusion/internal/server/router"
	"github.com/gin-contrib/gzip"
	"github.com/gin-gonic/gin"
)

func NewHTTPServer(
	cr *router.ConnectionRouter,
	ahp *router.AssignHostPortRouter,
	alc *router.AllocatorInfoRouter,
	leaderChan <-chan struct{},
) *gin.Engine {

	r := gin.New()
	r.Use(gzip.Gzip(gzip.DefaultCompression))
	r.Use(gin.Recovery())
	r.Use(gin.Logger())

	apiGroup := r.Group("/api")
	apiGroup.GET("/connection", cr.Get)
	apiGroup.GET("/allocation", alc.Get)
	apiGroup.POST("/simulate-schedule", alc.SimulateScheduleOnePod)
	apiGroup.POST("/assign-host-port", func(ctx *gin.Context) {
		if leaderChan == nil {
			ctx.String(http.StatusServiceUnavailable, "current instance is not the leader")
			return
		}
		select {
		case <-ctx.Done():
			return
		case <-leaderChan:
		}
		// suspend API call utils it becomes leader
		ahp.AssignHostPort(ctx)
	})
	return r
}
