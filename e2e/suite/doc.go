// Package suite holds TensorFusion's cluster-level end-to-end suite.
//
// Every test file in this directory is behind the build tag `e2e`, so ordinary
// builds never compile the suite. This untagged file exists so that the
// directory is still a well-formed (empty) Go package: `go build ./...`,
// `go vet ./...`, `go test ./...` and `ginkgo -r` (which discovers suites by
// file name, not by build tag) see a package with no tests instead of
// "build constraints exclude all Go files".
//
// Run it through e2e/Makefile, or directly:
//
//	go test -tags e2e -mod=vendor -p 2 -c -o suite.test ./e2e/suite
//
// See e2e/README.md.
package suite
