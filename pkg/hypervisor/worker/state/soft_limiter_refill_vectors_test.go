package worker

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// vectorFloat is a double in testdata/erl_refill_vectors.json: a number, or one
// of the strings "NaN", "Inf", "-Inf".
type vectorFloat float64

func (v *vectorFloat) UnmarshalJSON(data []byte) error {
	var number float64
	if err := json.Unmarshal(data, &number); err == nil {
		*v = vectorFloat(number)
		return nil
	}
	var text string
	if err := json.Unmarshal(data, &text); err != nil {
		return err
	}
	switch text {
	case "NaN":
		*v = vectorFloat(math.NaN())
	case "Inf":
		*v = vectorFloat(math.Inf(1))
	case "-Inf":
		*v = vectorFloat(math.Inf(-1))
	default:
		return fmt.Errorf("not a number: %q", text)
	}
	return nil
}

// The refill arithmetic is shared with the limiter inside the workload, which
// refills a bucket itself when it finds nobody doing so. Both sides must
// compute the same; the vectors are the common definition and are run here
// against the Go implementation.
func TestRefillVectors(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "erl_refill_vectors.json"))
	if err != nil {
		t.Fatal(err)
	}
	var file struct {
		MaxElapsed float64 `json:"max_elapsed"`
		Vectors    []struct {
			Name       string      `json:"name"`
			Last       vectorFloat `json:"last"`
			Now        vectorFloat `json:"now"`
			Rate       vectorFloat `json:"rate"`
			Capacity   vectorFloat `json:"capacity"`
			Tokens     vectorFloat `json:"tokens"`
			WantTokens vectorFloat `json:"want_tokens"`
			WantLast   vectorFloat `json:"want_last"`
			WantAdded  vectorFloat `json:"want_added"`
		} `json:"vectors"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		t.Fatal(err)
	}
	if len(file.Vectors) < 20 || file.MaxElapsed <= 0 {
		t.Fatalf("vector file is incomplete: %d vectors, max_elapsed %v", len(file.Vectors), file.MaxElapsed)
	}
	for _, vector := range file.Vectors {
		t.Run(vector.Name, func(t *testing.T) {
			var info SharedDeviceInfoV2
			info.SetERLLastTokenUpdate(float64(vector.Last))
			info.SetERLCurrentTokens(float64(vector.Tokens))

			added := info.RefillERLTokens(float64(vector.Now), float64(vector.Rate), float64(vector.Capacity), file.MaxElapsed)

			// Bit for bit: the two implementations must agree exactly.
			sameBits := func(a, b float64) bool { return math.Float64bits(a) == math.Float64bits(b) }
			if got, want := info.GetERLCurrentTokens(), float64(vector.WantTokens); !sameBits(got, want) {
				t.Errorf("tokens = %v, want %v", got, want)
			}
			if got, want := info.GetERLLastTokenUpdate(), float64(vector.WantLast); !sameBits(got, want) {
				t.Errorf("timestamp = %v, want %v", got, want)
			}
			if want := float64(vector.WantAdded); added != want {
				t.Errorf("added = %v, want %v", added, want)
			}
		})
	}
}
