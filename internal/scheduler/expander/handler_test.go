package expander

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestNodeExpander(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "NodeExpander Test Suite")
}

var _ = Describe("NodeExpander", func() {

	BeforeEach(func() {
	})

	Describe("selectBestCandidateNode", func() {
		It("should select node with smallest GPU count", func() {
			// TODO
		})
	})
})
