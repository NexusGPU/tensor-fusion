/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package server

import (
	"context"
	"fmt"
	"net/http"

	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/framework"
	"github.com/NexusGPU/tensor-fusion/internal/hypervisor/server/handlers"
	"github.com/gin-gonic/gin"
	"k8s.io/klog/v2"
)

// MetricsRecorder interface for metrics
type MetricsRecorder interface {
	Start()
}

// Server represents the hypervisor HTTP server
type Server struct {
	deviceController framework.DeviceController
	workerController framework.WorkerController
	metricsRecorder  MetricsRecorder
	backend          framework.Backend
	ctx              context.Context
	router           *gin.Engine
	httpServer       *http.Server

	// Handlers
	healthHandler *handlers.HealthHandler
	deviceHandler *handlers.DeviceHandler
	workerHandler *handlers.WorkerHandler
	legacyHandler *handlers.LegacyHandler
}

// NewServer creates a new hypervisor HTTP server
func NewServer(
	ctx context.Context,
	deviceController framework.DeviceController,
	workerController framework.WorkerController,
	metricsRecorder MetricsRecorder,
	backend framework.Backend,
	port int,
) *Server {
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Logger(), gin.Recovery())

	// Initialize handlers
	healthHandler := handlers.NewHealthHandler()
	deviceHandler := handlers.NewDeviceHandler(deviceController)
	workerHandler := handlers.NewWorkerHandler(workerController)
	legacyHandler := handlers.NewLegacyHandler(workerController, backend)

	s := &Server{
		deviceController: deviceController,
		workerController: workerController,
		metricsRecorder:  metricsRecorder,
		backend:          backend,
		ctx:              ctx,
		router:           router,
		httpServer: &http.Server{
			Addr:    fmt.Sprintf(":%d", port),
			Handler: router,
		},
		healthHandler: healthHandler,
		deviceHandler: deviceHandler,
		workerHandler: workerHandler,
		legacyHandler: legacyHandler,
	}

	s.setupRoutes()
	return s
}

func (s *Server) setupRoutes() {
	// Health check routes
	s.router.GET("/healthz", s.healthHandler.HandleHealthz)
	s.router.GET("/readyz", func(c *gin.Context) {
		s.healthHandler.HandleReadyz(c, s.deviceController, s.workerController)
	})

	// RESTful API routes
	// TODO: add authentication and authorization for worker APIs
	apiV1 := s.router.Group("/api/v1")
	{
		// Device routes
		apiV1.GET("/devices", s.deviceHandler.HandleGetDevices)
		apiV1.GET("/devices/:uuid", s.deviceHandler.HandleGetDevice)
		apiV1.POST("/devices/discover", s.deviceHandler.HandleDiscoverDevices)

		// Worker routes
		apiV1.GET("/workers", s.workerHandler.HandleGetWorkers)
		apiV1.GET("/workers/:id", s.workerHandler.HandleGetWorker)
		apiV1.POST("/workers/:id/snapshot", s.workerHandler.HandleSnapshotWorker)
		apiV1.POST("/workers/:id/resume", s.workerHandler.HandleResumeWorker)

		// Legacy routes
		apiV1.GET("/limiter", s.legacyHandler.HandleGetLimiter)
		apiV1.POST("/trap", s.legacyHandler.HandleTrap)
		apiV1.GET("/pod", s.legacyHandler.HandleGetPods)
		// TODO: should eliminate this API from limiter: apiV1.GET("/process", s.legacyHandler.HandleGetProcesses)
	}
}

// Start starts the HTTP server
func (s *Server) Start() error {
	klog.Infof("Starting hypervisor HTTP server on %s", s.httpServer.Addr)
	return s.httpServer.ListenAndServe()
}

// Stop stops the HTTP server
func (s *Server) Stop(ctx context.Context) error {
	return s.httpServer.Shutdown(ctx)
}
