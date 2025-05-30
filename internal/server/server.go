package server

import (
	"github.com/NexusGPU/tensor-fusion/internal/server/router"
	"github.com/gin-contrib/gzip"
	"github.com/gin-gonic/gin"
)

func NewHTTPServer(
	cr *router.ConnectionRouter,
	ahp *router.AssignHostPortRouter,
) *gin.Engine {

	r := gin.New()
	r.Use(gzip.Gzip(gzip.DefaultCompression))
	r.Use(gin.Recovery())
	r.Use(gin.Logger())

	apiGroup := r.Group("/api")
	apiGroup.GET("/connection", cr.Get)
	apiGroup.POST("/assign-host-port", ahp.AssignHostPort)
	return r
}
