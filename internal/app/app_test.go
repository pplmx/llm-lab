package app

import (
	"fmt"
	"image/png"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"

	"llmlab/internal/model"
)

func newTestApp(t testing.TB) *App {
	t.Helper()
	w, err := model.Bundled()
	if err != nil {
		t.Fatalf("bundled weights: %v", err)
	}
	return New(w)
}

// TestRenderSteps draws every stage to /tmp so the layout can be inspected
// without a window. Run with -run TestRenderSteps and look at the PNGs.
func TestRenderSteps(t *testing.T) {
	if os.Getenv("RENDER") == "" {
		t.Skip("set RENDER=1 to write snapshots")
	}
	a := newTestApp(t)
	dir := os.Getenv("RENDER_DIR")
	if dir == "" {
		dir = "/tmp"
	}
	for step := range lessons {
		a.Step = step
		img := ui.Render(a.View, 1500, 940, 1)
		path := fmt.Sprintf("%s/llmlab-step%d.png", dir, step)
		f, err := os.Create(path)
		if err != nil {
			t.Fatal(err)
		}
		if err := png.Encode(f, img); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s", path)
	}

	// A CJK prompt exercises three-byte tokens and wider chips.
	a.Prompt = "注意力就是加权求和"
	a.Step = 0
	a.recompute()
	img := ui.Render(a.View, 1500, 940, 1)
	path := dir + "/llmlab-step0-cjk.png"
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %s", path)
}

func TestRecomputeKeepsSelectionValid(t *testing.T) {
	a := newTestApp(t)
	a.SelToken = 999
	a.recompute()
	if a.SelToken >= len(a.Trace.Tokens) {
		t.Fatalf("selection %d out of range for %d tokens", a.SelToken, len(a.Trace.Tokens))
	}
	if len(a.Trace.Tokens) == 0 {
		t.Fatal("expected tokens")
	}
}

func TestSamplingAppendsToken(t *testing.T) {
	a := newTestApp(t)
	before := len(a.Prompt)
	a.sampleNext()
	if len(a.Prompt) != before+1 {
		t.Fatalf("prompt grew by %d bytes, want 1", len(a.Prompt)-before)
	}
	if a.LastSample == "" {
		t.Fatal("expected a sample description")
	}
}

// TestLongPromptTruncates checks that a prompt longer than the context window
// keeps the most recent tokens instead of panicking.
func TestLongPromptTruncates(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = strings.Repeat("transformer ", 20)
	a.recompute()
	if got := len(a.Trace.Tokens); got != a.W.Config.Ctx {
		t.Fatalf("%d tokens, want %d", got, a.W.Config.Ctx)
	}
	if !strings.Contains(a.Status, "截断") {
		t.Fatalf("status = %q", a.Status)
	}
}

// TestTesterClicks steps through the lesson with real pointer events, without
// a window.
func TestTesterClicks(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	if err := tr.Click("Softmax"); err != nil {
		t.Fatalf("click Softmax: %v", err)
	}
	if a.Step != 3 {
		t.Fatalf("step = %d, want 3", a.Step)
	}
	if err := tr.Click("FFN"); err != nil {
		t.Fatalf("click FFN: %v", err)
	}
	if a.Step != 5 {
		t.Fatalf("step = %d, want 5", a.Step)
	}
}

func TestSelectStepResetsScrollAndStopsPlaying(t *testing.T) {
	a := newTestApp(t)
	a.Scroll.Y = 200
	a.Playing = true
	a.selectStep(3)
	if a.Step != 3 || a.Scroll.Y != 0 || a.Playing {
		t.Fatalf("selectStep = step %d, scroll %v, playing %v", a.Step, a.Scroll.Y, a.Playing)
	}
}

func TestAutoplayUsesFrameClock(t *testing.T) {
	a := newTestApp(t)
	a.Playing = true
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	tr.Advance(1400 * time.Millisecond)
	if a.Step != 1 {
		t.Fatalf("step after 1.4s = %d, want 1", a.Step)
	}
}

func TestRecomputePreservesSelectionWhenPromptIsExtended(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = strings.Repeat("0123456789abcdef", 5) // 80 bytes, truncated at 64
	a.recompute()
	a.SelToken = 5
	old := a.Trace.Tokens[a.SelToken]
	a.Prompt += "XYZ"
	a.recompute()
	if got := a.Trace.Tokens[a.SelToken]; got.ID != old.ID {
		t.Fatalf("selection moved from id %d to id %d", old.ID, got.ID)
	}
}

func TestEmptyPromptDoesNotSample(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = ""
	a.recompute()
	a.sampleNext()
	if a.Prompt != "" {
		t.Fatalf("empty prompt grew to %q", a.Prompt)
	}
}

func BenchmarkView(b *testing.B) {
	a := newTestApp(b)
	a.Step = 3
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ui.Render(a.View, 1500, 940, 1)
	}
}

func BenchmarkRecomputeLong(b *testing.B) {
	a := newTestApp(b)
	a.Prompt = strings.Repeat("attention ", 8)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.recompute()
	}
}
