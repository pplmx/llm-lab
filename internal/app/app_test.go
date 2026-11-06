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

// TestKeyboardNavigation drives the lesson with the shortcuts the footer
// advertises, without touching the pointer.
func TestKeyboardNavigation(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	tr.Key(0, ui.KeyRight)
	tr.Frame()
	if a.Step != 1 {
		t.Fatalf("step after right = %d, want 1", a.Step)
	}
	tr.Key(0, ui.Key3)
	tr.Frame()
	if a.Step != 2 {
		t.Fatalf("step after 3 = %d, want 2", a.Step)
	}
	tr.Key(0, ui.KeyLeft)
	tr.Frame()
	if a.Step != 1 {
		t.Fatalf("step after left = %d, want 1", a.Step)
	}
	tr.Key(0, ui.KeySpace)
	tr.Frame()
	if !a.Playing {
		t.Fatal("space did not start autoplay")
	}
	tr.Key(0, ui.KeyEscape)
	tr.Frame()
	if a.Playing {
		t.Fatal("escape did not stop autoplay")
	}
}

// TestTypingDoesNotTriggerShortcuts checks that the global shortcuts yield to
// a focused text field: typing a digit or an arrow key edits the prompt
// instead of jumping between lessons.
func TestTypingDoesNotTriggerShortcuts(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	if err := tr.Click("输入"); err != nil {
		t.Fatalf("click input: %v", err)
	}
	tr.Frame()
	before := a.Prompt
	tr.TypeKey(0, ui.Key3, "3")
	tr.Frame()
	if a.Step != 0 {
		t.Fatalf("typing a digit changed the step to %d", a.Step)
	}
	if a.Prompt == before {
		t.Fatalf("typing a digit did not reach the input: %q", a.Prompt)
	}
	tr.Key(0, ui.KeyLeft)
	tr.Frame()
	if a.Step != 0 {
		t.Fatalf("left arrow in the input changed the step to %d", a.Step)
	}
}

// TestPresetPrompt clicks the long-prompt preset, which should switch the
// prompt and truncate it to the context window.
func TestPresetPrompt(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	if err := tr.Click("长提示词"); err != nil {
		t.Fatalf("click preset: %v", err)
	}
	want := strings.Repeat("attention ", 12)
	if a.Prompt != want {
		t.Fatalf("prompt = %q, want %q", a.Prompt, want)
	}
	if got := len(a.Trace.Tokens); got != a.W.Config.Ctx {
		t.Fatalf("%d tokens, want %d", got, a.W.Config.Ctx)
	}
	if !strings.Contains(a.Status, "截断") {
		t.Fatalf("status = %q", a.Status)
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

// TestTokenRailSelectsToken clicks a chip in the strip that is visible on every
// stage, and checks that the inspector follows the new selection.
func TestTokenRailSelectsToken(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	// The strip opens scrolled to the selected token, which is the last one;
	// a reader scrolled back to the start would see the earlier chips. Put it
	// there before clicking one, so the click lands on a chip in view.
	a.RailScroll.X = 0
	a.RevealRail = false
	tr.Frame()
	if err := tr.Click("token 2"); err != nil {
		t.Fatalf("click rail chip: %v", err)
	}
	tr.Frame()
	if a.SelToken != 2 {
		t.Fatalf("selection = %d, want 2", a.SelToken)
	}
}

// TestTokenRailStaysAcrossStages checks that the strip is drawn on every stage,
// not only while tokenising: the token sequence is the input of all seven.
func TestTokenRailStaysAcrossStages(t *testing.T) {
	a := newTestApp(t)
	for step := range lessons {
		a.Step = step
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		if _, ok := tr.Find("token 0"); !ok {
			t.Fatalf("stage %d: the token strip is missing", step)
		}
	}
}

// TestShiftArrowMovesTokenSelection drives the keyboard shortcut the strip
// advertises, including its clamp at both ends of the sequence.
func TestShiftArrowMovesTokenSelection(t *testing.T) {
	a := newTestApp(t)
	a.SelToken = 1
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	tr.Key(ui.Shift, ui.KeyRight)
	tr.Frame()
	if a.SelToken != 2 {
		t.Fatalf("after Shift+Right: selection = %d, want 2", a.SelToken)
	}
	tr.Key(ui.Shift, ui.KeyLeft)
	tr.Frame()
	if a.SelToken != 1 {
		t.Fatalf("after Shift+Left: selection = %d, want 1", a.SelToken)
	}
	a.SelToken = 0
	tr.Frame()
	tr.Key(ui.Shift, ui.KeyLeft)
	tr.Frame()
	if a.SelToken != 0 {
		t.Fatalf("Shift+Left at the first token moved to %d", a.SelToken)
	}
}

// TestHoverCellMarksRailTokens checks the cross-view link: pointing at an
// attention cell marks its query and key on the strip. The heatmap draws no
// text, so the pointer finds it the way a reader does — by moving over the
// matrix until a cell reports in — which keeps the assertion on behaviour
// rather than on exact pixel coordinates.
func TestHoverCellMarksRailTokens(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	// Scan from the matrix's bottom-left, where the causal mask hides nothing:
	// every cell there has a key far below its query, which is the pair the
	// strip has to tell apart. The diagonal would mark one chip twice.
	found := false
	for y := float32(700); y > 240 && !found; y -= 6 {
		for x := float32(240); x < 420 && !found; x += 6 {
			tr.Move(x, y)
			tr.Frame()
			found = a.HoverQ >= 0 && a.HoverK >= 0 && a.HoverQ != a.HoverK
		}
	}
	if !found {
		t.Fatal("moving over the attention matrix never set an off-diagonal cell")
	}
	q, k := a.HoverQ, a.HoverK
	if got := a.railMarks(q); got != "q" {
		t.Fatalf("railMarks(query %d) = %q, want %q", q, got, "q")
	}
	if got := a.railMarks(k); got != "k" {
		t.Fatalf("railMarks(key %d) = %q, want %q", k, got, "k")
	}
	// Moving off the matrix clears the marks.
	tr.Move(1250, 500)
	tr.Frame()
	if a.HoverQ >= 0 || a.HoverK >= 0 {
		t.Fatalf("hover persisted after leaving the matrix: q=%d k=%d", a.HoverQ, a.HoverK)
	}
	// And the strip marks cells only on the two attention stages, so a
	// crosshair cannot leak into a stage that draws no matrix.
	a.HoverQ, a.HoverK = q, k
	a.selectStep(5)
	if got := a.railMarks(q); got != "" {
		t.Fatalf("railMarks on the FFN stage = %q, want no mark", got)
	}
}
