package utils

import (
	"strings"

	"github.com/NexusGPU/tensor-fusion/pkg/constants"
	"github.com/gin-gonic/gin"
)

const BearerPrefix = "Bearer "

// ExtractBearerToken extracts the authorization token from the gin context.
// It handles both cases: token with "Bearer " prefix and token without prefix.
// Returns the token string (with Bearer prefix stripped if present) and true if token exists.
// Returns empty string and false if token is missing.
func ExtractBearerToken(ctx *gin.Context) (string, bool) {
	token := ctx.Request.Header.Get(constants.AuthorizationHeader)
	if token == "" {
		return "", false
	}

	// Strip Bearer prefix if present
	token = strings.TrimPrefix(token, BearerPrefix)

	return token, true
}
