package external_dp

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestDetector(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "External Device Plugin Detector Suite")
}
