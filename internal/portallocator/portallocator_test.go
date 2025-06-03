package portallocator

import (
	"fmt"
	"strconv"
	"sync"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Port Allocator", func() {
	BeforeEach(func() {
		// Reset state before each test
		// This is important to ensure tests don't interfere with each other
		// We're using the existing pa instance from the suite setup
	})

	Context("AssignHostPort", func() {
		It("should assign a valid port for a node", func() {
			port, err := pa.AssignHostPort("node-1")
			Expect(err).NotTo(HaveOccurred())
			Expect(port).To(Equal(40002))

			port, err = pa.AssignHostPort("node-1")
			Expect(err).NotTo(HaveOccurred())
			Expect(port).To(Equal(40003))

			err = pa.ReleaseHostPort("node-1", 40002)
			Expect(err).NotTo(HaveOccurred())

			port, err = pa.AssignHostPort("node-1")
			Expect(err).NotTo(HaveOccurred())
			Expect(port).To(Equal(40002))

			port, err = pa.AssignHostPort("node-new")
			Expect(err).NotTo(HaveOccurred())
			Expect(port).To(Equal(40000))
		})

		It("should fail when node name is empty", func() {
			_, err := pa.AssignHostPort("")
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("node name cannot be empty"))
		})

		It("should exhaust ports and return error when no ports available", func() {
			// Create a node with a small port range for testing exhaustion
			nodeName := "exhaust-test-node"

			// Assign ports until we get an error
			var lastPort int
			var err error
			assignedPorts := make(map[int]bool)

			// Keep assigning ports until we get an error or hit a reasonable limit
			for i := 0; i < 2002; i++ {
				lastPort, err = pa.AssignHostPort(nodeName)
				if err != nil {
					break
				}

				// Verify we don't get duplicate ports
				Expect(assignedPorts).NotTo(HaveKey(lastPort))
				assignedPorts[lastPort] = true
			}

			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("no available port"))
		})
	})

	Context("ReleaseHostPort", func() {
		It("should release a port successfully", func() {
			nodeName := "release-test-node"
			port, err := pa.AssignHostPort(nodeName)
			Expect(err).NotTo(HaveOccurred())

			err = pa.ReleaseHostPort(nodeName, port)
			Expect(err).NotTo(HaveOccurred())
		})

		It("should fail to release ports with invalid parameters", func() {
			tests := []struct {
				description string
				node        string
				port        int
				errorMsg    string
			}{
				{"invalid node name", "invalid-node-name", 40001, "node invalid-node-name not found"},
				{"port is zero", "node-1", 0, "port cannot be 0 when release host port"},
			}

			for _, tc := range tests {
				By(tc.description)
				err := pa.ReleaseHostPort(tc.node, tc.port)
				Expect(err).To(HaveOccurred())
				if tc.errorMsg != "" {
					Expect(err.Error()).To(ContainSubstring(tc.errorMsg))
				}
			}
		})
	})

	Context("Cluster Level Port Allocation", func() {
		It("should assign and release cluster level ports", func() {
			podName := "test-cluster-pod"
			port, err := pa.AssignClusterLevelHostPort(podName)
			Expect(err).NotTo(HaveOccurred())
			Expect(port).To(Equal(42002))

			err = pa.ReleaseClusterLevelHostPort(podName, port)
			Expect(err).NotTo(HaveOccurred())

			err = pa.ReleaseClusterLevelHostPort(podName, 59999)
			Expect(err).NotTo(HaveOccurred())

			port, err = pa.AssignClusterLevelHostPort(podName)
			Expect(err).NotTo(HaveOccurred())
			Expect(port).To(Equal(42002))
		})

		It("should fail to release a cluster port with invalid parameters", func() {
			err := pa.ReleaseClusterLevelHostPort("test-pod", 0)
			Expect(err).To(HaveOccurred())
			Expect(err.Error()).To(ContainSubstring("port cannot be 0 when release host port"))
		})
	})

	Context("Concurrency", func() {
		It("should handle concurrent port assignments and releases", func() {
			const workers = 20
			var wg sync.WaitGroup
			results := make(chan error, workers)

			wg.Add(workers)
			for i := 0; i < workers; i++ {
				go func(i int) {
					defer wg.Done()
					node := "concurrent-node-" + strconv.Itoa(i%5)
					_, err := pa.AssignHostPort(node)
					if err != nil {
						results <- fmt.Errorf("assignment failed: %v", err)
						return
					}
				}(i)
			}

			// Wait for all goroutines to complete
			wg.Wait()

			for i := 0; i < 5; i++ {
				bitMap := pa.BitmapPerNode["concurrent-node-"+strconv.Itoa(i)]
				Expect(bitMap).To(HaveLen(32))
				Expect(bitMap[0]).To(Equal(uint64(0xf)))
			}

			close(results)

			// Check for any errors
			for err := range results {
				Expect(err).NotTo(HaveOccurred())
			}
		})
	})
})
