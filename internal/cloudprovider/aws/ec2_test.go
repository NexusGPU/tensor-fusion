package aws

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/ec2"
	ec2Types "github.com/aws/aws-sdk-go-v2/service/ec2/types"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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

func TestCreateNodeUsesClaimIdentityForIdempotency(t *testing.T) {
	var tokens []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseForm(); err != nil {
			t.Error(err)
		}
		tokens = append(tokens, r.Form.Get("ClientToken"))
		w.Header().Set("Content-Type", "text/xml")
		_, _ = w.Write([]byte(`<RunInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2016-11-15/"><requestId>test</requestId><instancesSet><item><instanceId>i-test</instanceId></item></instancesSet></RunInstancesResponse>`))
	}))
	defer server.Close()
	provider := AWSGPUNodeProvider{
		ec2Client: ec2.NewFromConfig(aws.Config{Region: "us-west-2", Credentials: aws.AnonymousCredentials{}}, func(o *ec2.Options) { o.BaseEndpoint = aws.String(server.URL) }),
		nodeClass: &tfv1.GPUNodeClass{},
	}
	provider.nodeClass.Spec.OSImageSelectorTerms = []tfv1.NodeClassItemSelectorTerms{{ID: "ami-test"}}
	claim := &tfv1.GPUNodeClaim{ObjectMeta: metav1.ObjectMeta{Name: "claim", UID: "first-uid"}}
	for range 2 {
		_, err := provider.CreateNode(t.Context(), claim)
		require.NoError(t, err)
	}
	claim.UID = "second-uid"
	_, err := provider.CreateNode(t.Context(), claim)
	require.NoError(t, err)
	require.Len(t, tokens, 3)
	require.NotEmpty(t, tokens[0])
	require.Equal(t, tokens[0], tokens[1], "retries must not create another instance")
	require.NotEqual(t, tokens[0], tokens[2], "recreated claims need a new instance")
}
