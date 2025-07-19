package recommender_test

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestRecommender(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Recommender Suite")
}
