package utils_test

import (
	"context"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/stretchr/testify/require"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

func TestFindRootOwnerReference(t *testing.T) {
	// Prepare the scheme
	sch := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(sch))
	require.NoError(t, appsv1.AddToScheme(sch))

	t.Run("no owner returns self", func(t *testing.T) {
		pod := &corev1.Pod{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "v1",
				Kind:       "Pod",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mypod",
				Namespace: "default",
				UID:       "uid-pod",
			},
		}

		// build fake client with only pod
		c := fake.NewClientBuilder().WithScheme(sch).WithObjects(pod).Build()

		rootRef, err := utils.FindRootOwnerReference(context.TODO(), c, "default", pod)
		require.NoError(t, err)
		require.Nil(t, rootRef)
	})

	t.Run("hierarchy returns deployment", func(t *testing.T) {
		controller := true
		deployment := &appsv1.Deployment{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mydeploy",
				Namespace: "default",
				UID:       "uid-deploy",
			},
		}

		rs := &appsv1.ReplicaSet{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "apps/v1",
				Kind:       "ReplicaSet",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      "myrs",
				Namespace: "default",
				UID:       "uid-rs",
				OwnerReferences: []metav1.OwnerReference{
					{
						APIVersion: "apps/v1",
						Kind:       "Deployment",
						Name:       "mydeploy",
						UID:        deployment.UID,
						Controller: &controller,
					},
				},
			},
		}

		pod := &corev1.Pod{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "v1",
				Kind:       "Pod",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mypod",
				Namespace: "default",
				UID:       "uid-pod",
				OwnerReferences: []metav1.OwnerReference{
					{
						APIVersion: "apps/v1",
						Kind:       "ReplicaSet",
						Name:       "myrs",
						UID:        rs.UID,
						Controller: &controller,
					},
				},
			},
		}

		c := fake.NewClientBuilder().WithScheme(sch).WithObjects(pod, rs, deployment).Build()

		rootRef, err := utils.FindRootOwnerReference(context.TODO(), c, "default", pod)
		require.NoError(t, err)
		require.NotNil(t, rootRef)
		require.Equal(t, "mydeploy", rootRef.Name)
		require.Equal(t, "Deployment", rootRef.Kind)
	})

	t.Run("missing owner returns ownerRef", func(t *testing.T) {
		// Pod refers to a ReplicaSet that doesn't exist in fake client
		controller := true
		pod := &corev1.Pod{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "v1",
				Kind:       "Pod",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mypod",
				Namespace: "default",
				UID:       "uid-pod",
				OwnerReferences: []metav1.OwnerReference{
					{
						APIVersion: "apps/v1",
						Kind:       "ReplicaSet",
						Name:       "missing-rs",
						UID:        "uid-missing",
						Controller: &controller,
					},
				},
			},
		}

		c := fake.NewClientBuilder().WithScheme(sch).WithObjects(pod).Build()

		rootRef, err := utils.FindRootOwnerReference(context.TODO(), c, "default", pod)
		require.NoError(t, err)
		require.NotNil(t, rootRef)
		require.Equal(t, "missing-rs", rootRef.Name)
		require.Equal(t, "ReplicaSet", rootRef.Kind)
	})
}
