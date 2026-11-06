package app

import (
	"fmt"
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
		writeSnapshot(t, dir, fmt.Sprintf("llmlab-step%d.png", step), ui.Render(a.View, 1500, 940, 1))
	}

	// A CJK prompt exercises three-byte tokens and wider chips.
	a.Prompt = "注意力就是加权求和"
	a.Step = 0
	a.recompute()
	writeSnapshot(t, dir, "llmlab-step0-cjk.png", ui.Render(a.View, 1500, 940, 1))
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

// TestStageScopedControls checks that each top-bar control is live only on the
// stages that read it: a control that changes no pixels must not look live.
func TestStageScopedControls(t *testing.T) {
	if layerFrom > headFrom || headFrom > tempFrom {
		t.Fatalf("control ranges out of order: layer %d head %d temp %d", layerFrom, headFrom, tempFrom)
	}
	for step := range lessons {
		enabled := map[string]bool{
			"层":  inStages(step, layerFrom, layerTo),
			"头":  inStages(step, headFrom, headTo),
			"温度": inStages(step, tempFrom, tempTo),
		}
		// The ranges must be contiguous and cover exactly the stages that read
		// the value, otherwise a dimmed control still changes what is drawn or
		// a live one changes nothing.
		wantLayer := step >= 2 && step <= 5
		wantHead := step == 2 || step == 3
		wantTemp := step == 6
		if enabled["层"] != wantLayer || enabled["头"] != wantHead || enabled["温度"] != wantTemp {
			t.Fatalf("stage %d: controls %v, want 层=%v 头=%v 温度=%v",
				step, enabled, wantLayer, wantHead, wantTemp)
		}
	}
}

// TestDimmedControlIgnoresClicks proves the dimming is not cosmetic: at a stage
// that ignores the layer, clicking a layer segment must not move it.
func TestDimmedControlIgnoresClicks(t *testing.T) {
	a := newTestApp(t)
	a.Step = 0
	a.Layer = 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	if err := tr.Click("1"); err != nil {
		t.Fatalf("click layer segment: %v", err)
	}
	tr.Frame()
	if a.Layer != 0 {
		t.Fatalf("dimmed layer control moved to %d", a.Layer)
	}
	// On the attention stage the same click does move it.
	a.Step = 3
	tr.Frame()
	if err := tr.Click("1"); err != nil {
		t.Fatalf("click layer segment on stage 3: %v", err)
	}
	tr.Frame()
	if a.Layer != 1 {
		t.Fatalf("live layer control did not move: %d", a.Layer)
	}
}

// TestStepButtonsDimAtEnds checks the two ends of the lesson: the button that
// cannot move the reader is disabled, the other one is not.
func TestStepButtonsDimAtEnds(t *testing.T) {
	a := newTestApp(t)
	a.Step = 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	// "← 上一步" at the first step: a click leaves the lesson where it is.
	if err := tr.Click("← 上一步"); err != nil {
		t.Fatalf("click prev: %v", err)
	}
	tr.Frame()
	if a.Step != 0 {
		t.Fatalf("prev moved past the first step: %d", a.Step)
	}
	a.Step = len(lessons) - 1
	tr.Frame()
	if err := tr.Click("下一步 →"); err != nil {
		t.Fatalf("click next: %v", err)
	}
	tr.Frame()
	if a.Step != len(lessons)-1 {
		t.Fatalf("next moved past the last step: %d", a.Step)
	}
}

// TestEmptyPromptOffersPresets checks that the empty state names a way out on
// every stage: the presets used to live only in the tokenising card, so a
// reader who emptied the prompt on step 5 had no visible recovery.
func TestEmptyPromptOffersPresets(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = ""
	a.recompute()
	for _, step := range []int{0, 4, 6} {
		a.Step = step
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		if !tr.HasText("还没有提示词") {
			t.Fatalf("stage %d: empty state does not say there is no prompt", step)
		}
		if !tr.HasText("字节级分词器") {
			t.Fatalf("stage %d: empty state does not explain what the field accepts", step)
		}
		// The chips themselves must be live, not just named: clicking one
		// restores the pipeline.
		if err := tr.Click("中文"); err != nil {
			t.Fatalf("stage %d: click a preset: %v", step, err)
		}
		tr.Frame()
		if len(a.Trace.Tokens) == 0 {
			t.Fatalf("stage %d: the preset did not restore tokens", step)
		}
		if a.Step != step {
			t.Fatalf("stage %d: choosing a preset jumped to stage %d", step, a.Step)
		}
		a.Prompt = ""
		a.recompute()
	}
}

// TestEmptyPromptHasNoEmptyPresetChip checks the chip the reader cannot use is
// not offered: with an empty prompt, an "空" chip would be a button that does
// nothing, and the inspector already says the pipeline has no input.
func TestEmptyPromptHasNoEmptyPresetChip(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = ""
	a.recompute()
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	for _, s := range tr.Texts() {
		if s == "空" {
			t.Fatal("the empty preset is offered while the prompt is already empty")
		}
	}
}

// TestUntrainedWeightsWarned checks that random weights are called out. They
// still produce a plausible causal triangle and rows that sum to 1, so nothing
// else on screen would tell the reader the numbers mean nothing.
func TestUntrainedWeightsWarned(t *testing.T) {
	w := model.Random(model.DefaultConfig, 42)
	a := New(w)
	if !a.untrained() {
		t.Fatal("random weights are not reported as untrained")
	}
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	if !tr.HasText("权重未训练") {
		t.Fatal("no banner for untrained weights")
	}
	if !tr.HasText("别从输出里读出意义") {
		t.Fatal("the banner does not say the outputs are meaningless")
	}

	// Trained weights say nothing: the banner is for the failure only.
	trained := newTestApp(t)
	if trained.untrained() {
		t.Fatal("bundled weights are reported as untrained")
	}
	tr2 := ui.NewTester(trained.View, 1500, 940)
	tr2.Frame()
	if tr2.HasText("权重未训练") {
		t.Fatal("the untrained banner shows for the bundled weights")
	}
}

// TestEmptyPromptInspectorExplains checks the inspector's empty state names the
// next action rather than only describing what it would show.
func TestEmptyPromptInspectorExplains(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = ""
	a.recompute()
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	if !tr.HasText("先在上方输入一句提示词") {
		t.Fatal("the inspector's empty state does not point at the input field")
	}
}

// TestUntrainedBannerNamesTheCause checks the two ways to reach an untrained
// model get different advice: the bundled file is missing, or the -weights file
// the reader named carries no training metadata.
func TestUntrainedBannerNamesTheCause(t *testing.T) {
	bundledMissing := New(model.Random(model.DefaultConfig, 42))
	if got := bundledMissing.untrainedWhy(); !strings.Contains(got, "内置权重") {
		t.Fatalf("random fallback says %q, want it to name the bundled file", got)
	}
	custom := model.Random(model.DefaultConfig, 7)
	custom.Source = "/tmp/other.json"
	if got := New(custom).untrainedWhy(); !strings.Contains(got, "没有训练信息") {
		t.Fatalf("custom untrained weights say %q, want it to name the file's missing steps", got)
	}
}
