package device

import "testing"

func TestResolvePartitionResultHandlerDefaults(t *testing.T) {
	if _, ok := resolvePartitionResultHandler(PartitionTypeEnvironmentVariable); !ok {
		t.Fatalf("expected env partition handler to be available")
	}
	if _, ok := resolvePartitionResultHandler(PartitionTypeDeviceNode); !ok {
		t.Fatalf("expected device-node partition handler to be available")
	}
}

func TestResolvePartitionResultHandlerUnknownType(t *testing.T) {
	if _, ok := resolvePartitionResultHandler(PartitionResultType(100)); ok {
		t.Fatalf("expected unknown partition result type to be rejected")
	}
}
