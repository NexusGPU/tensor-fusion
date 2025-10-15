package component

import (
	"testing"
)

func TestCalculateDesiredUpdatedDelta(t *testing.T) {
	tests := []struct {
		name           string
		total          int
		updatedSize    int
		batchPercent   int32
		updateProgress int32
		wantDelta      int32
		wantProgress   int32
		wantBatchIdx   int32
	}{
		{
			name:           "initial update with 10% batch",
			total:          100,
			updatedSize:    0,
			batchPercent:   10,
			updateProgress: 0,
			wantDelta:      10,
			wantProgress:   0,
			wantBatchIdx:   0,
		},
		{
			name:           "second batch with 10%",
			total:          100,
			updatedSize:    10,
			batchPercent:   10,
			updateProgress: 10,
			wantDelta:      10,
			wantProgress:   10,
			wantBatchIdx:   1,
		},
		{
			name:           "handle non-divisible numbers",
			total:          95,
			updatedSize:    0,
			batchPercent:   30,
			updateProgress: 0,
			wantDelta:      29,
			wantProgress:   0,
			wantBatchIdx:   0,
		},
		{
			name:           "handle updated size larger than desired",
			total:          100,
			updatedSize:    35,
			batchPercent:   20,
			updateProgress: 20,
			wantDelta:      5,
			wantProgress:   20,
			wantBatchIdx:   1,
		},
		{
			name:           "final batch",
			total:          100,
			updatedSize:    90,
			batchPercent:   20,
			updateProgress: 80,
			wantDelta:      10,
			wantProgress:   80,
			wantBatchIdx:   4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotDelta, gotProgress, gotBatchIdx := calculateDesiredUpdatedDelta(
				tt.total,
				tt.updatedSize,
				tt.batchPercent,
				tt.updateProgress,
			)

			if gotDelta != tt.wantDelta {
				t.Errorf("calculateDesiredUpdatedDelta() delta = %v, want %v", gotDelta, tt.wantDelta)
			}
			if gotProgress != tt.wantProgress {
				t.Errorf("calculateDesiredUpdatedDelta() progress = %v, want %v", gotProgress, tt.wantProgress)
			}
			if gotBatchIdx != tt.wantBatchIdx {
				t.Errorf("calculateDesiredUpdatedDelta() batchIdx = %v, want %v", gotBatchIdx, tt.wantBatchIdx)
			}
		})
	}
}
