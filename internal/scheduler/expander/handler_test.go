package expander

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestNodeExpander(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "NodeExpander Test Suite")
}

var _ = Describe("NodeExpander", func() {
	var expander *NodeExpander

	BeforeEach(func() {
		expander = &NodeExpander{}
	})

	Describe("getGPUCount", func() {
		It("should return correct count for node with 4 GPUs", func() {
			node := &corev1.Node{
				Status: corev1.NodeStatus{
					Capacity: corev1.ResourceList{
						"nvidia.com/gpu": resource.MustParse("4"),
					},
				},
			}
			result := expander.getGPUCount(node)
			Expect(result).To(Equal(int64(4)))
		})

		It("should return 0 for node with no GPUs", func() {
			node := &corev1.Node{
				Status: corev1.NodeStatus{
					Capacity: corev1.ResourceList{
						"cpu": resource.MustParse("8"),
					},
				},
			}
			result := expander.getGPUCount(node)
			Expect(result).To(Equal(int64(0)))
		})
	})

	Describe("selectBestCandidateNode", func() {
		It("should select node with smallest GPU count", func() {
			node1 := &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node1"},
				Status: corev1.NodeStatus{
					Capacity: corev1.ResourceList{
						"nvidia.com/gpu": resource.MustParse("8"),
					},
				},
			}

			node2 := &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node2"},
				Status: corev1.NodeStatus{
					Capacity: corev1.ResourceList{
						"nvidia.com/gpu": resource.MustParse("4"),
					},
				},
			}

			node3 := &corev1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "node3"},
				Status: corev1.NodeStatus{
					Capacity: corev1.ResourceList{
						"nvidia.com/gpu": resource.MustParse("2"),
					},
				},
			}

			nodes := []*corev1.Node{node1, node2, node3}
			bestNode := expander.selectBestCandidateNode(nodes)

			Expect(bestNode.Name).To(Equal("node3"))
		})
	})
})
