package aws

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/ec2"
	"github.com/stretchr/testify/require"
)

func TestAWSProviderLoadsCredentialsForSignedRequests(t *testing.T) {
	for _, source := range []string{"environment", "web identity", "missing credentials"} {
		t.Run(source, func(t *testing.T) {
			for _, key := range []string{
				"AWS_ACCESS_KEY_ID", "AWS_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_SESSION_TOKEN",
				"AWS_ROLE_ARN", "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_STS",
				"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI",
			} {
				t.Setenv(key, "")
			}
			t.Setenv("AWS_EC2_METADATA_DISABLED", "true")
			t.Setenv("AWS_IGNORE_CONFIGURED_ENDPOINT_URLS", "false")
			t.Setenv("AWS_PROFILE", "default")
			t.Setenv("AWS_REGION", "us-east-1")
			configFile := filepath.Join(t.TempDir(), "config")
			require.NoError(t, os.WriteFile(configFile, []byte("[default]\n"), 0600))
			t.Setenv("AWS_CONFIG_FILE", configFile)
			t.Setenv("AWS_SHARED_CREDENTIALS_FILE", configFile)
			var ec2Requests, stsRequests atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := r.ParseForm(); err != nil {
					t.Error(err)
				}
				w.Header().Set("Content-Type", "text/xml")
				switch r.Form.Get("Action") {
				case "AssumeRoleWithWebIdentity":
					stsRequests.Add(1)
					if r.Form.Get("WebIdentityToken") != "test-web-identity-token" {
						t.Error("web identity token was not loaded from the configured file")
					}
					_, _ = fmt.Fprintf(w, `<AssumeRoleWithWebIdentityResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/">
<AssumeRoleWithWebIdentityResult><Credentials><AccessKeyId>test-key</AccessKeyId>
<SecretAccessKey>test-secret</SecretAccessKey>
<SessionToken>test-session</SessionToken><Expiration>%s</Expiration></Credentials></AssumeRoleWithWebIdentityResult>
</AssumeRoleWithWebIdentityResponse>`, time.Now().Add(time.Hour).UTC().Format(time.RFC3339))
				case "DescribeInstances":
					ec2Requests.Add(1)
					authorization := r.Header.Get("Authorization")
					if !strings.Contains(authorization, "Credential=test-key/") ||
						!strings.Contains(authorization, "/us-west-2/ec2/aws4_request") ||
						r.Header.Get("X-Amz-Security-Token") != "test-session" {
						t.Error("EC2 request was not signed using the resolved credentials")
					}
					_, _ = w.Write([]byte(`<DescribeInstancesResponse xmlns="http://ec2.amazonaws.com/doc/2016-11-15/">
<requestId>test</requestId><reservationSet/></DescribeInstancesResponse>`))
				default:
					t.Errorf("unexpected AWS request: %s", r.Form.Get("Action"))
					http.Error(w, "unexpected request", http.StatusBadRequest)
				}
			}))
			defer server.Close()
			switch source {
			case "environment":
				t.Setenv("AWS_ACCESS_KEY_ID", "test-key")
				t.Setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
				t.Setenv("AWS_SESSION_TOKEN", "test-session")
			case "web identity":
				tokenFile := filepath.Join(t.TempDir(), "token")
				require.NoError(t, os.WriteFile(tokenFile, []byte("test-web-identity-token"), 0600))
				t.Setenv("AWS_WEB_IDENTITY_TOKEN_FILE", tokenFile)
				t.Setenv("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/test-role")
				t.Setenv("AWS_ENDPOINT_URL_STS", server.URL)
			}
			provider, err := NewAWSGPUNodeProvider(t.Context(), tfv1.ComputingVendorConfig{
				Params: tfv1.ComputingVendorParams{DefaultRegion: "us-west-2"},
			}, &tfv1.GPUNodeClass{})
			require.NoError(t, err)
			require.Equal(t, "us-west-2", provider.ec2Client.Options().Region)
			// Exercise the real constructor's credentials against local endpoints.
			provider.ec2Client = ec2.New(provider.ec2Client.Options(), func(o *ec2.Options) {
				o.BaseEndpoint = aws.String(server.URL)
			})
			if source == "missing credentials" {
				require.Error(t, provider.TestConnection())
				require.Zero(t, ec2Requests.Load(), "missing credentials must not send an unsigned request")
				return
			}
			require.NoError(t, provider.TestConnection())
			require.EqualValues(t, 1, ec2Requests.Load())
			if source == "web identity" {
				require.EqualValues(t, 1, stsRequests.Load())
			}
		})
	}
}
