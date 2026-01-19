package alert

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/NexusGPU/tensor-fusion/internal/config"
)

var _ = Describe("Alert API", func() {
	Describe("SendAlert", func() {
		It("should return nil for empty alerts", func() {
			err := SendAlert(context.Background(), "http://localhost:9093", []config.PostableAlert{})
			Expect(err).NotTo(HaveOccurred())

			err = SendAlert(context.Background(), "http://localhost:9093", nil)
			Expect(err).NotTo(HaveOccurred())
		})

		Describe("URL Normalization", func() {
			DescribeTable("normalizes URL correctly",
				func(inputURL, expectedURL string) {
					var receivedPath string

					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						receivedPath = r.URL.Path
						w.WriteHeader(http.StatusOK)
					}))
					defer server.Close()

					// Replace the base URL with our test server
					testURL := server.URL
					if inputURL != "http://localhost:9093" && inputURL != "http://localhost:9093/" {
						if inputURL == "http://localhost:9093/alertmanager" {
							testURL = server.URL + "/alertmanager"
						} else if inputURL == "http://localhost:9093/alertmanager/" {
							testURL = server.URL + "/alertmanager/"
						}
					} else if inputURL == "http://localhost:9093/" {
						testURL = server.URL + "/"
					}

					alerts := []config.PostableAlert{
						{
							Alert: config.Alert{
								Labels: config.LabelSet{"alertname": "test"},
							},
							StartsAt: time.Now(),
						},
					}

					err := SendAlert(context.Background(), testURL, alerts)
					Expect(err).NotTo(HaveOccurred())
					Expect(receivedPath).To(Equal(expectedURL))
				},
				Entry("URL without trailing slash", "http://localhost:9093", "/api/v2/alerts"),
				Entry("URL with trailing slash", "http://localhost:9093/", "/api/v2/alerts"),
				Entry("URL with path without trailing slash", "http://localhost:9093/alertmanager", "/alertmanager/api/v2/alerts"),
				Entry("URL with path with trailing slash", "http://localhost:9093/alertmanager/", "/alertmanager/api/v2/alerts"),
			)
		})

		It("should send valid JSON request body", func() {
			var receivedBody []byte
			var contentType string

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				contentType = r.Header.Get("Content-Type")
				body, err := io.ReadAll(r.Body)
				Expect(err).NotTo(HaveOccurred())
				receivedBody = body
				w.WriteHeader(http.StatusOK)
			}))
			defer server.Close()

			alerts := []config.PostableAlert{
				{
					Alert: config.Alert{
						Labels: config.LabelSet{
							"alertname": "HighCPUUsage",
							"severity":  "critical",
							"instance":  "server-01",
						},
						GeneratorURL: "http://example.com/alerts",
					},
					StartsAt: time.Date(2024, 1, 15, 10, 0, 0, 0, time.UTC),
					Annotations: config.LabelSet{
						"summary":     "High CPU usage detected",
						"description": "CPU usage is above 90%",
					},
				},
				{
					Alert: config.Alert{
						Labels: config.LabelSet{
							"alertname": "LowMemory",
							"severity":  "warning",
						},
					},
					StartsAt: time.Date(2024, 1, 15, 10, 5, 0, 0, time.UTC),
				},
			}

			err := SendAlert(context.Background(), server.URL, alerts)
			Expect(err).NotTo(HaveOccurred())

			// Verify Content-Type header
			Expect(contentType).To(Equal("application/json"))

			// Verify body is valid JSON
			var parsedAlerts []config.PostableAlert
			err = json.Unmarshal(receivedBody, &parsedAlerts)
			Expect(err).NotTo(HaveOccurred())

			Expect(parsedAlerts).To(HaveLen(2))
			Expect(parsedAlerts[0].Labels["alertname"]).To(Equal("HighCPUUsage"))
			Expect(parsedAlerts[0].Labels["severity"]).To(Equal("critical"))
			Expect(parsedAlerts[1].Labels["alertname"]).To(Equal("LowMemory"))
		})

		Describe("HTTP Response Handling", func() {
			DescribeTable("handles HTTP responses correctly",
				func(statusCode int, expectError bool, errorContains string) {
					server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						w.WriteHeader(statusCode)
					}))
					defer server.Close()

					alerts := []config.PostableAlert{
						{
							Alert: config.Alert{
								Labels: config.LabelSet{"alertname": "test"},
							},
						},
					}

					err := SendAlert(context.Background(), server.URL, alerts)

					if expectError {
						Expect(err).To(HaveOccurred())
						Expect(err.Error()).To(ContainSubstring(errorContains))
					} else {
						Expect(err).NotTo(HaveOccurred())
					}
				},
				Entry("200 OK - success", http.StatusOK, false, ""),
				Entry("400 Bad Request", http.StatusBadRequest, true, "400"),
				Entry("401 Unauthorized", http.StatusUnauthorized, true, "401"),
				Entry("500 Internal Server Error", http.StatusInternalServerError, true, "500"),
				Entry("503 Service Unavailable", http.StatusServiceUnavailable, true, "503"),
			)
		})

		It("should return error on network failure", func() {
			alerts := []config.PostableAlert{
				{
					Alert: config.Alert{
						Labels: config.LabelSet{"alertname": "test"},
					},
				},
			}

			err := SendAlert(context.Background(), "http://127.0.0.1:1", alerts)
			Expect(err).To(HaveOccurred())
		})

		It("should use POST method", func() {
			var receivedMethod string

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				receivedMethod = r.Method
				w.WriteHeader(http.StatusOK)
			}))
			defer server.Close()

			alerts := []config.PostableAlert{
				{
					Alert: config.Alert{
						Labels: config.LabelSet{"alertname": "test"},
					},
				},
			}

			err := SendAlert(context.Background(), server.URL, alerts)
			Expect(err).NotTo(HaveOccurred())

			Expect(receivedMethod).To(Equal(http.MethodPost))
		})
	})

})
