package indexallocator

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestIndexAllocator(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "IndexAllocator Suite")
}
