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

package controller

import (
	"context"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/provider"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// ProviderConfigReconciler reconciles a ProviderConfig object
type ProviderConfigReconciler struct {
	client.Client
	Scheme *runtime.Scheme

	ProviderManager *provider.Manager
}

// +kubebuilder:rbac:groups=tensor-fusion.ai,resources=providerconfigs,verbs=get;list;watch;create;update;patch;delete

// Reconcile handles ProviderConfig create/update/delete events
func (r *ProviderConfigReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var providerConfig tfv1.ProviderConfig
	if err := r.Get(ctx, req.NamespacedName, &providerConfig); err != nil {
		if errors.IsNotFound(err) {
			// ProviderConfig was deleted, remove from manager by name
			// Since we index by vendor, we need to scan and remove
			logger.Info("ProviderConfig deleted", "name", req.Name)
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Update the provider manager cache
	r.ProviderManager.UpdateProvider(&providerConfig)
	logger.Info("ProviderConfig synced", "vendor", providerConfig.Spec.Vendor)

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager
func (r *ProviderConfigReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&tfv1.ProviderConfig{}).
		Named("provider-config").
		Complete(r)
}
