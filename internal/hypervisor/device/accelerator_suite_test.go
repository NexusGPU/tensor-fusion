package device

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestAccelerator(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Accelerator Suite")
}
