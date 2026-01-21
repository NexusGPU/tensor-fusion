package gpuresources

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestGPUResources(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "GPUResources Suite")
}
