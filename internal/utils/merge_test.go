package utils_test

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

// Test structs for MergeStructFields tests
type SimpleStruct struct {
	BoolField   bool
	StringField string
	IntField    int
	Int32Field  int32
	Int64Field  int64
	UintField   uint
	Uint32Field uint32
	Float32     float32
	Float64     float64
}

type NestedStruct struct {
	Name   string
	Inner  InnerStruct
	PtrInt *int
}

type InnerStruct struct {
	Value    string
	Count    int
	Enabled  bool
	unexport string //nolint:unused
}

type StructWithSliceMap struct {
	Tags     []string
	Metadata map[string]string
}

var _ = Describe("MergeStructFields", func() {
	Describe("edge cases", func() {
		It("should be no-op for non-pointer inputs", func() {
			dst := SimpleStruct{StringField: "original"}
			src := SimpleStruct{StringField: "new"}

			utils.MergeStructFields(dst, src)

			Expect(dst.StringField).To(Equal("original"))
		})

		It("should be no-op for non-struct pointers", func() {
			dst := "original"
			src := "new"

			utils.MergeStructFields(&dst, &src)

			Expect(dst).To(Equal("original"))
		})

		It("should be no-op for different types", func() {
			dst := SimpleStruct{StringField: "original"}
			src := NestedStruct{Name: "new"}

			utils.MergeStructFields(&dst, &src)

			Expect(dst.StringField).To(Equal("original"))
		})
	})

	Describe("bool field merging", func() {
		DescribeTable("merges bool fields correctly",
			func(dstBool, srcBool, expected bool) {
				dst := &SimpleStruct{BoolField: dstBool}
				src := &SimpleStruct{BoolField: srcBool}

				utils.MergeStructFields(dst, src)

				Expect(dst.BoolField).To(Equal(expected))
			},
			Entry("src true overwrites dst false", false, true, true),
			Entry("src false does not overwrite dst true", true, false, true),
			Entry("both false stays false", false, false, false),
			Entry("both true stays true", true, true, true),
		)
	})

	Describe("string field merging", func() {
		DescribeTable("merges string fields correctly",
			func(dstStr, srcStr, expected string) {
				dst := &SimpleStruct{StringField: dstStr}
				src := &SimpleStruct{StringField: srcStr}

				utils.MergeStructFields(dst, src)

				Expect(dst.StringField).To(Equal(expected))
			},
			Entry("src non-empty overwrites dst empty", "", "new", "new"),
			Entry("src non-empty overwrites dst non-empty", "original", "new", "new"),
			Entry("src empty does not overwrite dst non-empty", "original", "", "original"),
			Entry("both empty stays empty", "", "", ""),
		)
	})

	Describe("int field merging", func() {
		DescribeTable("merges int fields correctly",
			func(dstInt, srcInt, expectedInt int) {
				dst := &SimpleStruct{IntField: dstInt}
				src := &SimpleStruct{IntField: srcInt}

				utils.MergeStructFields(dst, src)

				Expect(dst.IntField).To(Equal(expectedInt))
			},
			Entry("src non-zero overwrites dst zero", 0, 42, 42),
			Entry("src non-zero overwrites dst non-zero", 10, 42, 42),
			Entry("src zero does not overwrite dst non-zero", 10, 0, 10),
			Entry("negative src overwrites", 0, -5, -5),
		)
	})

	Describe("int32 and int64 field merging", func() {
		It("should merge int32 and int64 fields correctly", func() {
			dst := &SimpleStruct{
				Int32Field: 0,
				Int64Field: 100,
			}
			src := &SimpleStruct{
				Int32Field: 32,
				Int64Field: 0,
			}

			utils.MergeStructFields(dst, src)

			Expect(dst.Int32Field).To(Equal(int32(32)))
			Expect(dst.Int64Field).To(Equal(int64(100))) // Not overwritten because src is 0
		})
	})

	Describe("uint field merging", func() {
		It("should merge uint fields correctly", func() {
			dst := &SimpleStruct{
				UintField:   0,
				Uint32Field: 100,
			}
			src := &SimpleStruct{
				UintField:   42,
				Uint32Field: 0,
			}

			utils.MergeStructFields(dst, src)

			Expect(dst.UintField).To(Equal(uint(42)))
			Expect(dst.Uint32Field).To(Equal(uint32(100))) // Not overwritten because src is 0
		})
	})

	Describe("float field merging", func() {
		It("should merge float fields correctly", func() {
			dst := &SimpleStruct{
				Float32: 0.0,
				Float64: 3.14,
			}
			src := &SimpleStruct{
				Float32: 1.5,
				Float64: 0.0,
			}

			utils.MergeStructFields(dst, src)

			Expect(dst.Float32).To(Equal(float32(1.5)))
			Expect(dst.Float64).To(Equal(float64(3.14))) // Not overwritten because src is 0
		})
	})

	Describe("pointer field merging", func() {
		It("should overwrite dst nil with src non-nil", func() {
			srcVal := 42
			dst := &NestedStruct{PtrInt: nil}
			src := &NestedStruct{PtrInt: &srcVal}

			utils.MergeStructFields(dst, src)

			Expect(dst.PtrInt).NotTo(BeNil())
			Expect(*dst.PtrInt).To(Equal(42))
		})

		It("should overwrite dst non-nil with src non-nil", func() {
			dstVal := 10
			srcVal := 42
			dst := &NestedStruct{PtrInt: &dstVal}
			src := &NestedStruct{PtrInt: &srcVal}

			utils.MergeStructFields(dst, src)

			Expect(dst.PtrInt).NotTo(BeNil())
			Expect(*dst.PtrInt).To(Equal(42))
		})

		It("should not overwrite dst non-nil with src nil", func() {
			dstVal := 10
			dst := &NestedStruct{PtrInt: &dstVal}
			src := &NestedStruct{PtrInt: nil}

			utils.MergeStructFields(dst, src)

			Expect(dst.PtrInt).NotTo(BeNil())
			Expect(*dst.PtrInt).To(Equal(10))
		})
	})

	Describe("slice field merging", func() {
		It("should overwrite dst nil with src non-nil slice", func() {
			dst := &StructWithSliceMap{Tags: nil}
			src := &StructWithSliceMap{Tags: []string{"tag1", "tag2"}}

			utils.MergeStructFields(dst, src)

			Expect(dst.Tags).To(Equal([]string{"tag1", "tag2"}))
		})

		It("should not overwrite dst non-nil with src nil slice", func() {
			dst := &StructWithSliceMap{Tags: []string{"original"}}
			src := &StructWithSliceMap{Tags: nil}

			utils.MergeStructFields(dst, src)

			Expect(dst.Tags).To(Equal([]string{"original"}))
		})

		It("should overwrite with empty slice (considered non-nil)", func() {
			dst := &StructWithSliceMap{Tags: []string{"original"}}
			src := &StructWithSliceMap{Tags: []string{}}

			utils.MergeStructFields(dst, src)

			Expect(dst.Tags).To(Equal([]string{}))
		})
	})

	Describe("map field merging", func() {
		It("should overwrite dst nil with src non-nil map", func() {
			dst := &StructWithSliceMap{Metadata: nil}
			src := &StructWithSliceMap{Metadata: map[string]string{"key": "value"}}

			utils.MergeStructFields(dst, src)

			Expect(dst.Metadata).To(Equal(map[string]string{"key": "value"}))
		})

		It("should not overwrite dst non-nil with src nil map", func() {
			dst := &StructWithSliceMap{Metadata: map[string]string{"original": "data"}}
			src := &StructWithSliceMap{Metadata: nil}

			utils.MergeStructFields(dst, src)

			Expect(dst.Metadata).To(Equal(map[string]string{"original": "data"}))
		})
	})

	Describe("nested struct merging", func() {
		It("should merge nested struct fields recursively", func() {
			dst := &NestedStruct{
				Name: "original",
				Inner: InnerStruct{
					Value:   "dst-value",
					Count:   10,
					Enabled: false,
				},
			}
			src := &NestedStruct{
				Name: "", // Empty, should not overwrite
				Inner: InnerStruct{
					Value:   "src-value",
					Count:   0, // Zero, should not overwrite
					Enabled: true,
				},
			}

			utils.MergeStructFields(dst, src)

			Expect(dst.Name).To(Equal("original"))         // Not overwritten (src empty)
			Expect(dst.Inner.Value).To(Equal("src-value")) // Overwritten
			Expect(dst.Inner.Count).To(Equal(10))          // Not overwritten (src zero)
			Expect(dst.Inner.Enabled).To(BeTrue())         // Overwritten (src true)
		})
	})

	Describe("complex scenario", func() {
		It("should handle complex merging scenario correctly", func() {
			srcVal := 100
			dst := &NestedStruct{
				Name: "original-name",
				Inner: InnerStruct{
					Value:   "",
					Count:   5,
					Enabled: true,
				},
				PtrInt: nil,
			}
			src := &NestedStruct{
				Name: "new-name",
				Inner: InnerStruct{
					Value:   "filled",
					Count:   0,
					Enabled: false,
				},
				PtrInt: &srcVal,
			}

			utils.MergeStructFields(dst, src)

			Expect(dst.Name).To(Equal("new-name"))
			Expect(dst.Inner.Value).To(Equal("filled"))
			Expect(dst.Inner.Count).To(Equal(5))   // Not overwritten (src zero)
			Expect(dst.Inner.Enabled).To(BeTrue()) // Not overwritten (src false for bool)
			Expect(dst.PtrInt).NotTo(BeNil())
			Expect(*dst.PtrInt).To(Equal(100))
		})
	})
})
