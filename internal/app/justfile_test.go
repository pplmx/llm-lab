package app

import (
	"fmt"
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

// TestSnapshotRecipesExist keeps the visual gate runnable without hand-written shell.
//
// Comparing 39 PNGs was done by hand every time the UI changed, and the hand version failed
// twice in ways worth encoding: once against an empty baseline directory, which reported
// every file as changed, and once against a stale directory from a round that had rendered
// nothing. A recipe that refuses an empty baseline turns both into a clear error.
//
// The test checks the recipes exist and that the diff one guards the empty case, since that
// guard is the part a future edit is most likely to drop.
func TestSnapshotRecipesExist(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "justfile"))
	if err != nil {
		t.Fatal(err)
	}
	src := string(b)

	for _, want := range []string{"\nsnapshots dir=", "\nsnapshots-diff base "} {
		if !strings.Contains(src, want) {
			t.Fatalf("the justfile has no %q recipe", strings.TrimSpace(want))
		}
	}

	// The render recipe must go through RENDER_DIR, or it would write to /tmp and the
	// diff would compare against whatever happened to be there.
	if !strings.Contains(src, `RENDER=1 RENDER_DIR="{{dir}}" go test`) {
		t.Fatal("the snapshots recipe does not render into its argument")
	}

	// And the diff must refuse an empty baseline.
	if !strings.Contains(src, "基线目录是空的") {
		t.Fatal("snapshots-diff no longer rejects an empty baseline; an empty directory " +
			"would report every file as changed")
	}
}

// TestContributingSnapshotCountMatchesTheSet keeps the guide's "43 张" honest.
//
// The count is a claim of the same shape as the tooltip number, and of the same shape as the
// Tab-stop heading that was once "五个" over a six-row table: a number written beside a list,
// which nothing re-derives when an entry is added. It is unusually easy to get wrong here
// because the set is produced by *two* test functions — `just snapshots` runs TestRenderSteps
// and TestRenderSizes together — so counting either alone gives 8 or 35 and neither is the
// number the guide means.
//
// Counting writeSnapshot call sites in the source does not work: several of them sit inside
// loops (three window sizes, seven stages, two widths), so 13 call sites produce 43 files. The
// count has to come from the render itself. That is affordable — about 3.5 seconds, once — and
// it is the only number that cannot drift from the set.
func TestContributingSnapshotCountMatchesTheSet(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "CONTRIBUTING.md"))
	if err != nil {
		t.Fatalf("read CONTRIBUTING.md: %v", err)
	}
	doc := string(b)

	dir := t.TempDir()
	// The two functions `just snapshots` runs, together, into a scratch directory. This is
	// the same command the recipe issues, so the number counted here is the number a reader
	// gets from `just snapshots`.
	cmd := exec.Command("go", "test", ".", "-run", "TestRenderSteps|TestRenderSizes", "-count=1")
	cmd.Env = append(os.Environ(), "RENDER=1", "RENDER_DIR="+dir)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("rendering the snapshot set failed: %v\n%s", err, out)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read the rendered set: %v", err)
	}
	n := 0
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".png") {
			n++
		}
	}
	if n == 0 {
		t.Fatal("the snapshot set is empty; the guide's count is stale")
	}

	// Both mentions have to agree with the set, not just one. The guide states the number
	// twice (the recipe paragraph and the "screenshots cannot judge taste" note), and a check
	// that only looks for *a* match passes while the other copy says something else — verified
	// by mutation: changing one of the two to 44 left this test green until it counted them.
	want := fmt.Sprintf("%d 张", n)
	if got := strings.Count(doc, want); got != 2 {
		t.Fatalf("`just snapshots` writes %d PNGs, so CONTRIBUTING.md should say %q twice — "+
			"once in the recipe paragraph and once in the pixel-diff note — but the string "+
			"appears %d time(s); update both when the set changes on purpose", n, want, got)
	}
}

// TestEveryRecipeDescriptionIsOneLine keeps `just --list` readable.
//
// just takes the last comment line directly above a recipe as that recipe's description, so a
// multi-line comment block above a recipe silently promotes its *final* line into the listing.
// Four recipes had that: check showed "想一次对齐跑 just check-ci。", check-ci showed "两者分开是
// 有意的——…", docs showed "改完界面后跑一次这个。", and snapshots-diff showed "一张、或者把空目录
// 当基线（会误报「全部变化」）都不会有人发现。" — each a fragment of the sentence above it,
// listed as if it were the recipe's name.
//
// The rule this enforces: whatever comment sits directly above a recipe must be a complete,
// standalone description. Multi-line explanation is welcome, but it goes above a blank line.
func TestEveryRecipeDescriptionIsOneLine(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "..", "justfile"))
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(string(src), "\n")
	var bad []string
	for i, line := range lines {
		// A recipe line: `name:` or `name arg=default:` at column 0. Recipe bodies are
		// indented, so this cannot match one.
		if line == "" || line[0] == '#' || line[0] == ' ' || line[0] == '\t' {
			continue
		}
		head := line
		if j := strings.Index(head, ":"); j >= 0 {
			head = head[:j]
		} else {
			continue
		}
		// Recipes may carry parameters; the name is the first token.
		name := strings.Fields(head)[0]
		if name == "" || strings.HasPrefix(name, "_") {
			continue // private helpers are not listed
		}
		// `just --list` shows recipes only. These are the other things a justfile holds, and
		// they are not listed, so they need no description: settings and exports take `:=`,
		// variables are `name := value`, aliases are `alias x := y`.
		if name == "set" || name == "export" || name == "alias" {
			continue
		}
		if strings.Contains(line, ":=") {
			continue // a variable assignment such as windows_platforms := "windows/amd64"
		}
		// Walk up to the comment block.
		j := i - 1
		var block []string
		for j >= 0 && strings.HasPrefix(lines[j], "#") {
			block = append([]string{lines[j]}, block...)
			j--
		}
		if len(block) == 0 {
			bad = append(bad, fmt.Sprintf("%s has no description above it", name))
			continue
		}
		// The description is the last line. A description ending in a clause connector, or
		// consisting only of punctuation after a dash, is a fragment of the line above.
		desc := strings.TrimPrefix(block[len(block)-1], "#")
		desc = strings.TrimSpace(desc)
		if desc == "" {
			bad = append(bad, fmt.Sprintf("%s has an empty description", name))
			continue
		}
		for _, frag := range []string{"。", "；", "，"} {
			_ = frag
		}
		// A fragment is a description whose previous comment line does not end a sentence.
		if len(block) > 1 {
			prev := strings.TrimSpace(strings.TrimPrefix(block[len(block)-2], "#"))
			if prev != "" && !strings.HasSuffix(prev, "。") {
				bad = append(bad, fmt.Sprintf("%s: %q continues a sentence begun on the line "+
					"above, so it is a fragment of the explanation rather than a description",
					name, desc))
			}
		}
	}
	if len(bad) > 0 {
		t.Fatalf("just takes the last comment line above a recipe as its description, so "+
			"multi-line explanation has to sit above a blank line:\n  %s",
			strings.Join(bad, "\n  "))
	}
}
