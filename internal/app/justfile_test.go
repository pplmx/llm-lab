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
