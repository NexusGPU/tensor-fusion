package utils

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/pkg/constants"
)

func TestLabelSafeValue(t *testing.T) {
	cases := []struct {
		name      string
		in        string
		unchanged bool
	}{
		{"empty", "", true},
		{"short", "gpu-node-01", true},
		{"exactly 63", strings.Repeat("a", 63), true},
		{"64", strings.Repeat("a", 64), false},
		{"100", "kwok-" + strings.Repeat("n", 95), false},
		{"253", strings.Repeat("b", 253), false},
		{"dotted fqdn", strings.Repeat("ip-10-0-0-1.", 8) + "compute.internal", false},
		{"cut lands on separators", strings.Repeat("a", 50) + "-.-" + strings.Repeat("c", 30), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := LabelSafeValue(tc.in)
			assert.LessOrEqual(t, len(got), MaxLabelValueLength)
			assert.Empty(t, validation.IsValidLabelValue(got), "must be a valid label value: %q", got)
			assert.Equal(t, got, LabelSafeValue(tc.in), "must be stable")
			if tc.unchanged {
				assert.Equal(t, tc.in, got, "a name that fits must not change")
				return
			}
			assert.NotEqual(t, tc.in, got)
			assert.True(t, strings.HasPrefix(tc.in, got[:20]), "keeps a readable prefix")
		})
	}

	t.Run("long names that share the kept prefix stay distinct", func(t *testing.T) {
		seen := map[string]string{}
		for i := range 500 {
			in := strings.Repeat("x", 80) + fmt.Sprintf("-%04d", i)
			got := LabelSafeValue(in)
			if prev, dup := seen[got]; dup {
				t.Fatalf("%q and %q map to the same value %q", prev, in, got)
			}
			seen[got] = in
		}
	})

	t.Run("a truncated value is not the plain 63 character prefix", func(t *testing.T) {
		long := strings.Repeat("y", 100)
		assert.NotEqual(t, long[:MaxLabelValueLength], LabelSafeValue(long))
	})
}

func TestGPUOwnerSelectorAndNodeName(t *testing.T) {
	short := "gpu-node-01"
	long := "kwok-" + strings.Repeat("n", 95)

	assert.Equal(t, short, GPUOwnerSelector(short)[constants.LabelKeyOwner])
	assert.Equal(t, LabelSafeValue(long), GPUOwnerSelector(long)[constants.LabelKeyOwner])

	owned := func(node string) *tfv1.GPU {
		return &tfv1.GPU{ObjectMeta: metav1.ObjectMeta{
			Labels: GPUOwnerSelector(node),
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: tfv1.GroupVersion.String(), Kind: gpuNodeKind, Name: node,
			}},
		}}
	}
	assert.Equal(t, long, GPUNodeNameOfGPU(owned(long)), "the full name comes from the owner reference")
	assert.Equal(t, short, GPUNodeNameOfGPU(owned(short)))

	// Objects written without an owner reference (tests, very old versions).
	legacy := &tfv1.GPU{ObjectMeta: metav1.ObjectMeta{Labels: GPUOwnerSelector(short)}}
	assert.Equal(t, short, GPUNodeNameOfGPU(legacy))
	assert.Empty(t, GPUNodeNameOfGPU(&tfv1.GPU{}))
	assert.Empty(t, GPUNodeNameOfGPU(nil))
}
