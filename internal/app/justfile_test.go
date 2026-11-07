package app

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestArtifactsRecipeSaysWhenThereIsNothing checks `just artifacts` reports an
// empty build directory instead of exiting silently.
//
// The recipe used to be `du -sh build/*/* 2>/dev/null | sort -k2 || echo "..."`.
// With no build/ the stderr goes to /dev/null, du fails, sort succeeds on empty
// input, and the pipeline exits 0 — so the fallback never ran and the command
// printed nothing. A reader could not tell "nothing built yet" from "the recipe is
// broken".
func TestArtifactsRecipeSaysWhenThereIsNothing(t *testing.T) {
	if _, err := exec.LookPath("just"); err != nil {
		t.Skip("just is not installed")
	}
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}

	out, err := exec.Command("just", "artifacts").Output()
	if err != nil {
		t.Fatalf("just artifacts failed: %v", err)
	}
	got := strings.TrimSpace(string(out))

	_, statErr := os.Stat(filepath.Join(root, "build"))
	if os.IsNotExist(statErr) {
		if got == "" {
			t.Fatal("just artifacts prints nothing when build/ is absent; it should " +
				"say so")
		}
		if !strings.Contains(got, "build") {
			t.Fatalf("just artifacts says %q, which does not mention build/", got)
		}
		return
	}
	// A build/ tree exists in this workspace: either it lists files or it says
	// the directory is empty, but it must say something.
	if got == "" {
		t.Fatal("just artifacts prints nothing")
	}
}

// TestCIConfigIsRunnable keeps the workflow's test commands within reach of the
// timeout they run under.
//
// The race step used to be `go test -race ./...`. Only internal/model has
// goroutines; internal/app is single-threaded UI code, and under the race detector
// one rendered frame costs 24ms instead of 0.8ms. The app package alone needs about
// 11 minutes that way, past Go's default 10-minute test timeout — so the step could
// not pass on a clean checkout. Racing a package with no concurrency bought
// nothing and cost the whole job.
func TestCIConfigIsRunnable(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", ".github", "workflows", "ci.yml"))
	if err != nil {
		t.Fatalf("read ci.yml: %v", err)
	}
	ci := string(b)

	// The race step must be scoped to the packages that actually have goroutines,
	// and must state its timeout rather than inheriting Go's default.
	if strings.Contains(ci, "go test -race ./...") {
		t.Fatal("ci.yml still races every package; internal/app has no concurrency " +
			"and takes ~11 minutes under -race")
	}
	if !strings.Contains(ci, "go test -race -timeout") {
		t.Fatal("the race step does not set an explicit -timeout")
	}
	if !strings.Contains(ci, "./internal/model/") {
		t.Fatal("the race step no longer covers internal/model, which is the only " +
			"package with goroutines")
	}

	// Every `go test` step should carry a timeout so a slow machine fails with a
	// clear message instead of hanging into the job timeout.
	for _, line := range strings.Split(ci, "\n") {
		trimmed := strings.TrimSpace(line)
		// Only the command line, not the step's name or comments.
		if !strings.HasPrefix(trimmed, "run:") || !strings.Contains(trimmed, "go test") {
			continue
		}
		if !strings.Contains(trimmed, "-timeout") {
			t.Fatalf("a go test step inherits the default timeout: %q", trimmed)
		}
	}
}

// TestConcurrencyLivesOnlyInModel pins the reason the race scope is what it is.
func TestConcurrencyLivesOnlyInModel(t *testing.T) {
	hasConcurrency := func(pattern string) bool {
		files, err := filepath.Glob(pattern)
		if err != nil {
			return false
		}
		for _, f := range files {
			if strings.HasSuffix(f, "_test.go") {
				continue
			}
			b, err := os.ReadFile(f)
			if err != nil {
				continue
			}
			src := string(b)
			if strings.Contains(src, "go func(") || strings.Contains(src, "sync.WaitGroup") {
				return true
			}
		}
		return false
	}
	if !hasConcurrency("../model/*.go") {
		t.Fatal("internal/model no longer has goroutines; the -race step could be " +
			"dropped or the comment explaining it is stale")
	}
	if hasConcurrency("*.go") {
		t.Fatal("internal/app has goroutines now, so the CI race step should cover it")
	}
}

// TestCheckCIRecipeMatchesTheWorkflow keeps the local full gate in step with CI.
//
// `just check` deliberately stops at gofmt, vet and test — about 35 seconds — while CI also
// verifies go.mod is tidy, races the concurrent package and re-runs everything under
// GOEXPERIMENT=simd. That split was already documented as a known difference, but the
// alignment had to be done by hand from a list in CONTRIBUTING.md, which is the kind of list
// that goes stale the first time CI gains a step.
//
// check-ci runs all of it. This test requires the recipe to exist and to run exactly the
// commands the workflow runs, so adding a CI step without teaching the recipe fails here
// rather than surprising a contributor with a red build.
func TestCheckCIRecipeMatchesTheWorkflow(t *testing.T) {
	read := func(name string) string {
		b, err := os.ReadFile(filepath.Join("..", "..", name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		return string(b)
	}
	recipe := read("justfile")
	workflow := read(".github/workflows/ci.yml")

	const target = "check-ci: check\n"
	i := strings.Index(recipe, target)
	if i < 0 {
		t.Fatal("the justfile has no check-ci recipe")
	}
	// The recipe body runs to the next blank line.
	body := recipe[i+len(target):]
	if j := strings.Index(body, "\n\n"); j >= 0 {
		body = body[:j]
	}

	for _, want := range []string{
		"go mod tidy",
		"git diff --exit-code -- go.mod go.sum",
		"go test -race -timeout 300s ./internal/model/",
		"GOEXPERIMENT=simd go test -timeout 300s ./...",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("check-ci does not run %q", want)
		}
		if !strings.Contains(workflow, want) {
			t.Errorf("the workflow does not run %q either; this list is stale", want)
		}
	}
}
