package aws

import (
	"testing"
	"time"

	ec2Types "github.com/aws/aws-sdk-go-v2/service/ec2/types"
	"github.com/stretchr/testify/require"
)

func TestGPUNodeStatusFromInstanceAllowsMissingPublicIP(t *testing.T) {
	launchTime := time.Unix(123, 0)
	status, err := gpuNodeStatusFromInstance(ec2Types.Instance{
		InstanceId:       stringPtr("i-private"),
		LaunchTime:       &launchTime,
		PrivateIpAddress: stringPtr("10.0.0.4"),
	})
	require.NoError(t, err)
	require.Equal(t, "i-private", status.InstanceID)
	require.Equal(t, "10.0.0.4", status.PrivateIP)
	require.Empty(t, status.PublicIP)
}

func TestGPUNodeStatusFromInstanceRequiresIdentity(t *testing.T) {
	_, err := gpuNodeStatusFromInstance(ec2Types.Instance{})
	require.Error(t, err)
}

func stringPtr(value string) *string { return &value }
