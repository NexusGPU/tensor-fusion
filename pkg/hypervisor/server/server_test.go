package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// The limiter events route is registered, and as the server wires it there is
// no way to reach it without a Pod token the backend verified: a server whose
// backend cannot review tokens answers 401 to everything.
func TestLimiterEventsRouteRequiresAVerifiedPodToken(t *testing.T) {
	s := NewServer(t.Context(), nil, nil, nil, nil, nil, 0)
	const url = "/api/v1/limiter/events?container_pid=1&container_name=worker"
	const report = `{"version":1,"pid":1,"container":"worker","events":[{"kind":"stalled_bucket","count":1}]}`

	for name, authorization := range map[string]string{
		"no token":           "",
		"unverifiable token": "Bearer header.payload.signature",
	} {
		request := httptest.NewRequest(http.MethodPost, url, strings.NewReader(report))
		if authorization != "" {
			request.Header.Set("Authorization", authorization)
		}
		response := httptest.NewRecorder()
		s.router.ServeHTTP(response, request)
		if response.Code != http.StatusUnauthorized {
			t.Errorf("%s: status = %d, want 401 (%s)", name, response.Code, response.Body.String())
		}
	}
	// Only POST is served.
	response := httptest.NewRecorder()
	s.router.ServeHTTP(response, httptest.NewRequest(http.MethodGet, url, nil))
	if response.Code != http.StatusNotFound && response.Code != http.StatusMethodNotAllowed {
		t.Errorf("GET: status = %d", response.Code)
	}
	if stats := s.LimiterEventStats(); len(stats.Pods) != 0 || stats.Totals["stalled_bucket"] != 0 {
		t.Errorf("unauthenticated reports were counted: %+v", stats)
	}
}
