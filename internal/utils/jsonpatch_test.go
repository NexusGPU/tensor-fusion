package utils_test

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

var _ = Describe("EscapeJSONPointer", func() {
	DescribeTable("escapes JSON pointer special characters correctly",
		func(input, expected string) {
			result := utils.EscapeJSONPointer(input)
			Expect(result).To(Equal(expected))
		},
		Entry("empty string", "", ""),
		Entry("no special characters", "simple-key", "simple-key"),
		Entry("tilde escape", "key~with~tildes", "key~0with~0tildes"),
		Entry("slash escape", "key/with/slashes", "key~1with~1slashes"),
		Entry("mixed special characters", "key~/mixed~chars/", "key~0~1mixed~0chars~1"),
		Entry("already escaped tilde should double escape", "~0already~1escaped", "~00already~01escaped"),
		Entry("consecutive special chars", "~~//~~", "~0~0~1~1~0~0"),
		Entry("special chars at boundaries", "~start/end~", "~0start~1end~0"),
		Entry("annotation key with domain", "tensor-fusion.ai/index", "tensor-fusion.ai~1index"),
	)
})
