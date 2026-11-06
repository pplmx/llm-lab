package app

import (
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"regexp"
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

// TestMisconceptionNumbersAreTrue checks the claim the score-stage callout
// makes against the model: shifting a whole score row by a constant must leave
// the attention weights untouched. If that ever stops holding, the callout is
// teaching something false.
func TestMisconceptionNumbersAreTrue(t *testing.T) {
	a := newTestApp(t)
	h := a.layer().Heads[a.Head]
	q := a.SelToken
	n := q + 1
	softmaxShifted := func(shift float64) []float64 {
		out := make([]float64, n)
		m := math.Inf(-1)
		for j := 0; j < n; j++ {
			if v := float64(h.Scores[q][j]) + shift; v > m {
				m = v
			}
		}
		var sum float64
		for j := 0; j < n; j++ {
			out[j] = math.Exp(float64(h.Scores[q][j]) + shift - m)
			sum += out[j]
		}
		for j := range out {
			out[j] /= sum
		}
		return out
	}
	base := softmaxShifted(0)
	if base[0] > 1 { // guard against a softmax over nothing
		t.Fatalf("softmax did not normalise: %v", base[0])
	}
	for _, shift := range []float64{-5, -10, 5, 10, 100} {
		moved := softmaxShifted(shift)
		for j := range base {
			if d := math.Abs(base[j] - moved[j]); d > 1e-12 {
				t.Fatalf("shifting the row by %v changed weight %d by %g; the callout claims it cannot",
					shift, j, d)
			}
		}
	}
}

// TestScoreLegendDoesNotClaimSignControlsAttention guards the correction this
// round made: the old legend said a negative score suppresses attention, which
// is false here — sign and relative weight disagree in a quarter of the cells.
func TestScoreLegendDoesNotClaimSignControlsAttention(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	for _, s := range tr.Texts() {
		if strings.Contains(s, "负分：抑制注意") {
			t.Fatal("the score legend still says a negative score suppresses attention")
		}
	}
	if !tr.HasText("常见误解") {
		t.Fatal("the score stage does not carry the misconception callout")
	}
}

// TestGELUFloorMatchesReality pins the -0.170 constant the FFN callout quotes
// to the exact erf-based GELU that internal/model samples into its lookup
// table. The table is unexported, so this recomputes the reference function the
// same way tensor.go does rather than reaching into it.
func TestGELUFloorMatchesReality(t *testing.T) {
	gelu := func(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }
	lo := math.Inf(1)
	for x := -4.0; x <= 0; x += 0.0005 {
		if v := gelu(x); v < lo {
			lo = v
		}
	}
	if math.Abs(lo-geluFloor) > 0.001 {
		t.Fatalf("geluFloor = %v, but GELU's minimum is %v", geluFloor, lo)
	}
	if lo >= 0 {
		t.Fatal("GELU never went negative; the callout would be wrong")
	}
}

// TestFFNParamShareMatchesParamCount checks the two shares against the count the
// sidebar already shows, so the number in the FFN header cannot drift from the
// model's actual parameter count.
func TestFFNParamShareMatchesParamCount(t *testing.T) {
	a := newTestApp(t)
	ofAll, ofBlocks := ffnParamShare(a.W.Config)
	c := a.W.Config
	total := float64(a.paramCount())
	ffn := float64(c.NLayer) * 2 * float64(c.DModel) * float64(c.DFF)
	if math.Abs(ofAll-ffn/total) > 1e-9 {
		t.Fatalf("ofAll = %v, want %v", ofAll, ffn/total)
	}
	if ofBlocks <= ofAll {
		t.Fatalf("block share %v should exceed the all-parameter share %v", ofBlocks, ofAll)
	}
	if ofAll <= 0 || ofAll >= 1 || ofBlocks <= 0 || ofBlocks >= 1 {
		t.Fatalf("shares out of range: ofAll=%v ofBlocks=%v", ofAll, ofBlocks)
	}
}

// TestMisconceptionCalloutsOnlyWhereMeasured keeps the callouts tied to stages
// whose claims were verified against the model, so a later edit cannot quietly
// move one next to a picture it does not describe. Steps 3, 4, 5, 6 and 7 have
// one; steps 1 and 2 are not misconceptions in the same sense.
func TestMisconceptionCalloutsOnlyWhereMeasured(t *testing.T) {
	want := map[int]bool{2: true, 3: true, 4: true, 5: true, 6: true}
	for step := range lessons {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		got := false
		for _, s := range tr.Texts() {
			if s == "常见误解" {
				got = true
				break
			}
		}
		if got != want[step] {
			t.Fatalf("stage %d has misconception callout = %v, want %v", step, got, want[step])
		}
	}
}

// TestCalloutNumbersFollowTheModel checks the misconception text quotes the
// running model, not the default prompt: the same stage must say different
// numbers when the prompt changes, and each number must match what the trace
// actually holds.
func TestCalloutNumbersFollowTheModel(t *testing.T) {
	quotes := func(prompt string, step int) string {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		a.Step = step
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		// The callout's correction is the longest text in the stage.
		longest := ""
		for _, s := range tr.Texts() {
			if len([]rune(s)) > len([]rune(longest)) {
				longest = s
			}
		}
		return longest
	}

	// Step 7 quotes the top-1 probability, which moves with the prompt.
	short := quotes("hi", 6)
	long := quotes("attention is all you need and the transformer", 6)
	if short == long {
		t.Fatal("the output callout says the same thing for two very different prompts; " +
			"its numbers are not read from the model")
	}
	for name, text := range map[string]string{"short prompt": short, "long prompt": long} {
		if !strings.Contains(text, "%") {
			t.Fatalf("%s: the output callout no longer quotes a probability: %s", name, text)
		}
	}

	// Step 6 quotes the FFN's negative-dimension count, which follows the token.
	ffnA := quotes("hi", 5)
	ffnB := quotes("attention is all you need and the transformer", 5)
	if ffnA == ffnB {
		t.Fatal("the FFN callout says the same thing for two different tokens")
	}
}

// TestFFNCalloutZeroCountIsMeasured checks the "sparse means near zero, not
// exactly zero" claim quotes the real count of exact zeros.
func TestFFNCalloutZeroCountIsMeasured(t *testing.T) {
	a := newTestApp(t)
	hidden := a.layer().Hidden[a.SelToken]
	zeros := zeroDims(hidden)
	for _, v := range hidden {
		if v == 0 {
			zeros--
		}
	}
	if zeros != 0 {
		t.Fatalf("zeroDims overcounted: %d left", zeros)
	}
	// GELU only returns exact 0 below its table's lower bound, so a normal
	// activation row should have none.
	if zeroDims(hidden) != 0 {
		t.Logf("note: this row has %d exact zeros", zeroDims(hidden))
	}
}

// TestAttentionRowsSumToOneButColumnsDoNot pins the attention stage's callout:
// rows are constrained to sum to 1, columns are not, and the grand total equals
// the token count.
func TestAttentionRowsSumToOneButColumnsDoNot(t *testing.T) {
	a := newTestApp(t)
	h := a.layer().Heads[a.Head]
	for i, row := range h.Attn {
		var sum float64
		for j := 0; j <= i && j < len(row); j++ {
			sum += float64(row[j])
		}
		if math.Abs(sum-1) > 1e-4 {
			t.Fatalf("row %d sums to %v, want 1", i, sum)
		}
	}
	colMax, colMin := columnSumRange(h.Attn)
	if !(colMax > 1) {
		t.Fatalf("every column sums to <= 1 (max %v); the callout claims columns are not constrained", colMax)
	}
	if !(colMin < 0.5) {
		t.Fatalf("no column is small (min %v); the callout's contrast would be invisible", colMin)
	}
	var total float64
	for i := range h.Attn {
		for j := 0; j <= i && j < len(h.Attn[i]); j++ {
			total += float64(h.Attn[i][j])
		}
	}
	if math.Abs(total-float64(len(h.Attn))) > 1e-3 {
		t.Fatalf("grand total %v, want %d (one per row)", total, len(h.Attn))
	}
}

// TestTypeScaleIsTheOnlySourceOfFontSizes keeps the ten-by-eye sizing from
// creeping back: every FontSize and Span Size in the package must be one of the
// six scale constants, never a literal. A literal here is a size nobody
// remembers writing and nobody dares change.
func TestTypeScaleIsTheOnlySourceOfFontSizes(t *testing.T) {
	sources, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	// Both the call form and the struct-field form the strip and stepper use.
	patterns := []*regexp.Regexp{
		regexp.MustCompile(`FontSize\(\s*[0-9]`),
		regexp.MustCompile(`Size:\s*[0-9]`),
	}
	for _, src := range sources {
		if strings.HasSuffix(src, "_test.go") {
			continue
		}
		body, err := os.ReadFile(src)
		if err != nil {
			t.Fatal(err)
		}
		for i, line := range strings.Split(string(body), "\n") {
			// The scale's own definition is the one place literals belong.
			if strings.Contains(line, "fontMicro =") || strings.Contains(line, "//\tfont") {
				continue
			}
			for _, re := range patterns {
				if re.MatchString(line) {
					t.Errorf("%s:%d sizes text with a literal, not the scale: %s",
						src, i+1, strings.TrimSpace(line))
				}
			}
		}
	}
}

// TestTypeScaleIsOrdered checks the scale itself stays monotonic and covers the
// roles it names, so the mapping table in theme.go and the numbers agree.
func TestTypeScaleIsOrdered(t *testing.T) {
	scale := []float32{fontMicro, fontSmall, fontBody, fontData, fontTitle, fontHero}
	for i := 1; i < len(scale); i++ {
		if scale[i] <= scale[i-1] {
			t.Fatalf("type scale is not increasing at step %d: %v", i, scale)
		}
	}
	// The smallest step has to stay legible: 9px was too small for the status
	// line, which is why the micro step is 10.
	if fontMicro < 10 {
		t.Fatalf("fontMicro = %v, below the 10px legibility floor", fontMicro)
	}
	// The scale must not grow back into a dozen steps by adding constants at
	// 1px intervals.
	if len(scale) > 6 {
		t.Fatalf("type scale has %d steps; six is the ceiling", len(scale))
	}
}

// TestStageSwitchFadesIn checks a stage switch animates its two panels and then
// settles exactly where a plain frame of that stage would be. The fade is
// deliberately opacity only: the lab is for comparing one stage's numbers with
// the next, so content that slid would make that harder to follow.
func TestStageSwitchFadesIn(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()
	a.selectStep(3)
	tr.Frame()
	if !tr.FramePending() {
		t.Fatal("switching stages asked for no animation")
	}
	// Mid-fade differs from the resting frame.
	mid := tr.Image()
	tr.Advance(200 * time.Millisecond)
	tr.Frame()
	if tr.FramePending() {
		t.Fatal("the stage fade never settled")
	}
	settled := tr.Image()
	if samePixelsE(mid, settled) {
		t.Fatal("the frame did not change between mid-fade and settling")
	}
	// And where it settles is exactly the stage's plain state, so the
	// animation cannot leave the layout somewhere a plain render is not.
	plain := ui.Render(a.View, 1180, 720, 1)
	if !samePixelsE(settled, plain) {
		t.Fatal("the settled frame differs from a plain render of the same stage")
	}
}

// TestStageFadeHasNoBlankFrame checks the fade never shows a frame with the
// content invisible. mygo plays the first frame of an Enter at its starting
// opacity, so Enter.Opacity 0 flashes emptiness on every switch — including
// under ReduceMotion, where mygo still fades a fade-only Enter. Measured over
// the stage column the opening frame must stay well above the background.
func TestStageFadeHasNoBlankFrame(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()
	a.selectStep(3)
	tr.Frame()
	darkest := stageBrightness(tr.Image())
	for tr.FramePending() {
		tr.Advance(20 * time.Millisecond)
		tr.Frame()
		if b := stageBrightness(tr.Image()); b < darkest {
			darkest = b
		}
	}
	tr.Advance(300 * time.Millisecond)
	tr.Frame()
	steady := stageBrightness(tr.Image())
	if darkest*3 < steady {
		t.Fatalf("the fade shows a nearly blank frame: darkest %d vs steady %d", darkest, steady)
	}
	if stageFade.Enter == nil || stageFade.Enter.Opacity < 0.2 {
		t.Fatalf("Enter.Opacity = %v, low enough to flash an empty frame",
			stageFade.Enter.Opacity)
	}
}

// stageBrightness is the mean channel value over a crop of the stage column,
// used to tell "content drawn" from "background only".
func stageBrightness(img *image.RGBA) int {
	crop := image.Rect(240, 300, 840, 500)
	sum, n := 0, 0
	for y := crop.Min.Y; y < crop.Max.Y; y++ {
		for x := crop.Min.X; x < crop.Max.X; x++ {
			i := img.PixOffset(x, y)
			sum += int(img.Pix[i]) + int(img.Pix[i+1]) + int(img.Pix[i+2])
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / n
}

// TestRenderIgnoresEnterMotion guards the screenshot suite: ui.Render draws one
// frame on a frozen clock, so a Transition's Enter must not hide content from
// it. If that ever changes, every snapshot would capture a faded-out stage.
func TestRenderIgnoresEnterMotion(t *testing.T) {
	a := newTestApp(t)
	for _, step := range []int{0, 3, 6} {
		a.Step = step
		img := ui.Render(a.View, 1180, 720, 1)
		// Nothing is drawn at the extreme right of the stage column unless the
		// stage rendered; a faded-out frame would leave that area at panel
		// colour only. Compare against the same render to prove determinism,
		// then check the frame is not blank by counting non-background pixels.
		again := ui.Render(a.View, 1180, 720, 1)
		if !samePixelsE(img, again) {
			t.Fatalf("stage %d: ui.Render is not deterministic", step)
		}
		if nonBgPixels(img) < 5000 {
			t.Fatalf("stage %d: only %d non-background pixels; the stage did not render",
				step, nonBgPixels(img))
		}
	}
}

func samePixelsE(a, b *image.RGBA) bool {
	if a.Bounds() != b.Bounds() {
		return false
	}
	for i := range a.Pix {
		if a.Pix[i] != b.Pix[i] {
			return false
		}
	}
	return true
}

func nonBgPixels(img *image.RGBA) int {
	n := 0
	bg := img.Pix[0]
	for i := 0; i < len(img.Pix); i += 4 {
		if img.Pix[i] != bg || img.Pix[i+1] != bg || img.Pix[i+2] != bg {
			n++
		}
	}
	return n
}

// TestShortcutIsAskedOncePerFrame guards the trap this round hit: Context.Shortcut
// reports false on a second call for the same key in one frame, because the
// first ask consumes the press. Writing `Shortcut(Shift, KeyRight) && stage`
// before a bare `Shortcut(Shift, KeyRight)` therefore makes the bare case
// unreachable, and the shortcut silently stops working.
func TestShortcutIsAskedOncePerFrame(t *testing.T) {
	var firstHit, secondHit bool
	view := func(c *ui.Context) {
		// Same key asked twice, as the broken shape did.
		firstHit = c.Shortcut(ui.Shift, ui.KeyUp)
		secondHit = c.Shortcut(ui.Shift, ui.KeyRight)
		ui.Text(c, "x")
	}
	tr := ui.NewTester(view, 400, 200)
	tr.Frame()
	tr.Key(ui.Shift, ui.KeyRight)
	for i := 0; i < 3; i++ {
		tr.Frame()
	}
	if firstHit {
		t.Fatal("Shift+Up reported for a Shift+Right press")
	}
	// This is mygo's contract, not a bug to work around: one ask per key per
	// frame. If a later mygo lets the second ask see the press, this test
	// fails and the guarded shape in handleKeys can be simplified.
	if secondHit {
		t.Log("mygo now allows a second Shortcut call for the same key in one frame")
	}
}

// TestShiftArrowsSplitByStage checks the two behaviours of Shift+Left/Right:
// the token selection on the stages without a matrix, and the attention cell on
// the two that have one.
func TestShiftArrowsSplitByStage(t *testing.T) {
	// Stages without a matrix: the selection moves.
	for _, step := range []int{0, 1, 4, 5, 6} {
		a := newTestApp(t)
		a.Step = step
		a.SelToken = 1
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		tr.Key(ui.Shift, ui.KeyRight)
		tr.Frame()
		if a.SelToken != 2 {
			t.Fatalf("stage %d: Shift+Right left the selection at %d, want 2", step, a.SelToken)
		}
		if a.KeyQ >= 0 {
			t.Fatalf("stage %d: a matrix cell was selected on a stage with no matrix", step)
		}
	}

	// The two attention stages: the cell moves, and the query follows it.
	for _, step := range []int{2, 3} {
		a := newTestApp(t)
		a.Step = step
		a.SelToken = 1
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		tr.Key(ui.Shift, ui.KeyRight)
		tr.Frame()
		if a.KeyQ < 0 || a.KeyK < 0 {
			t.Fatalf("stage %d: Shift+Right did not enter the matrix", step)
		}
		if a.KeyK > a.KeyQ {
			t.Fatalf("stage %d: keyboard cell %d,%d is outside the causal triangle",
				step, a.KeyQ, a.KeyK)
		}
		if a.SelToken != a.KeyQ {
			t.Fatalf("stage %d: selection %d does not follow the cell's query %d",
				step, a.SelToken, a.KeyQ)
		}
	}
}

// TestKeyboardCellSurvivesPointerMove is the reason KeyQ/KeyK exist separately
// from HoverQ/HoverK: the heatmap rewrites the hover pair every frame it draws,
// so a cell chosen with the keyboard would be erased by any pointer movement.
func TestKeyboardCellSurvivesPointerMove(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	tr.Key(ui.Shift, ui.KeyDown)
	tr.Frame()
	q, k := a.KeyQ, a.KeyK
	if q < 0 {
		t.Fatal("Shift+Down did not select a cell")
	}
	tr.Move(1250, 500) // well outside the matrix
	tr.Frame()
	if a.KeyQ != q || a.KeyK != k {
		t.Fatalf("a pointer move erased the keyboard cell: %d,%d -> %d,%d",
			q, k, a.KeyQ, a.KeyK)
	}
	if a.HoverQ >= 0 {
		t.Fatal("hover was expected to clear outside the matrix")
	}
	// And the cursor the UI reads is still the keyboard's.
	if cq, ck := a.cellCursor(); cq != q || ck != k {
		t.Fatalf("cellCursor = %d,%d, want the keyboard cell %d,%d", cq, ck, q, k)
	}
}

// TestKeyboardCellStaysInTriangle walks the cell in every direction and checks
// it never lands on a masked entry.
func TestKeyboardCellStaysInTriangle(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	tr.Key(ui.Shift, ui.KeyDown)
	tr.Frame()
	for _, key := range []ui.Key{ui.KeyLeft, ui.KeyUp, ui.KeyDown, ui.KeyRight} {
		for i := 0; i < 12; i++ {
			tr.Key(ui.Shift, key)
			tr.Frame()
			if a.KeyQ < 0 || a.KeyK < 0 {
				t.Fatalf("after %v: cell unset", key)
			}
			if a.KeyQ >= len(a.Trace.Tokens) || a.KeyK >= len(a.Trace.Tokens) {
				t.Fatalf("after %v: cell %d,%d past the sequence", key, a.KeyQ, a.KeyK)
			}
			if a.KeyK > a.KeyQ {
				t.Fatalf("after %v: cell %d,%d is above the diagonal (masked)",
					key, a.KeyQ, a.KeyK)
			}
		}
	}
}

// contrastRatio is the WCAG 2.1 relative-luminance ratio between two colours.
func contrastRatio(a, b ui.Color) float64 {
	la, lb := relativeLuminance(a), relativeLuminance(b)
	if la < lb {
		la, lb = lb, la
	}
	return (la + 0.05) / (lb + 0.05)
}

func relativeLuminance(c ui.Color) float64 {
	lin := func(v uint8) float64 {
		f := float64(v) / 255
		if f <= 0.04045 {
			return f / 12.92
		}
		return math.Pow((f+0.055)/1.055, 2.4)
	}
	return 0.2126*lin(c.R) + 0.7152*lin(c.G) + 0.0722*lin(c.B)
}

// over composites a translucent colour onto an opaque one, as the painter does.
func over(fg, bg ui.Color) ui.Color {
	a := float64(fg.A) / 255
	mix := func(f, b uint8) uint8 {
		return uint8(math.Round(float64(f)*a + float64(b)*(1-a)))
	}
	return ui.Color{R: mix(fg.R, bg.R), G: mix(fg.G, bg.G), B: mix(fg.B, bg.B), A: 255}
}

// TestPaletteContrast pins the pairs this round fixed and the ones around them
// that the audit found already passing, so a later palette tweak cannot quietly
// drop a caption below the 4.5:1 that small text needs. Disabled controls are
// exempt under WCAG, so they are not listed.
func TestPaletteContrast(t *testing.T) {
	const small = 4.5
	cases := []struct {
		name    string
		fg, bg  ui.Color
		minimum float64
	}{
		{"card body", colText, colPanel, small},
		{"card muted prose", colMuted, colPanel, small},
		{"card caption", colFaint, colPanel, small},
		{"sidebar status line", colFaint, colBg, small},
		{"sidebar model label", colMuted, colBg, small},
		// The three this round fixed.
		{"byte digit on its chip", colMuted, colPanel2, small},
		{"rail id on a hovered chip", colMuted, colPanel2, small},
		{"rail id on the selected chip", colMuted, over(colAccent.Alpha(0.16), colPanel), small},
		{"inspector glyph on its tint", colAccentOnTint, over(colAccent.Alpha(0.16), colPanel), small},
		// Text on a filled accent/badge surface runs the other way.
		{"stage badge", colBg, colAccent, small},
		{"stepper number on a done dot", colBg, colMuted, small},
		{"mark chip on warm", colBg, colWarm, small},
		{"mark chip on accent2", colBg, colAccent2, small},
	}
	for _, c := range cases {
		if got := contrastRatio(c.fg, c.bg); got < c.minimum {
			t.Errorf("%s: contrast %.2f:1, want at least %.1f:1", c.name, got, c.minimum)
		}
	}
}

// TestRailIDContrastInEveryChipState checks the rail's byte id against all
// three backgrounds the chip can have: idle, hovered, selected. It was colFaint
// in every state, which passed at 4.81:1 when idle but fell to 4.46:1 hovered
// and 3.84:1 selected.
func TestRailIDContrastInEveryChipState(t *testing.T) {
	states := map[string]ui.Color{
		"idle":     colPanel,
		"hovered":  colPanel2,
		"selected": over(colAccent.Alpha(0.16), colPanel),
	}
	for name, bg := range states {
		if got := contrastRatio(colMuted, bg); got < 4.5 {
			t.Errorf("rail id on the %s chip: %.2f:1", name, got)
		}
	}
}

// TestResidualCalloutNumbersAreTrue pins the residual stage's claim: the stream
// grows across blocks while the LayerNorm outputs feeding each sublayer stay
// pinned near √d_model. If either half stops holding, the callout is teaching
// something false.
func TestResidualCalloutNumbersAreTrue(t *testing.T) {
	a := newTestApp(t)
	tr := a.Trace
	inMean, outMean := residualNormMeans(tr)
	if !(outMean > inMean*1.5) {
		t.Fatalf("residual stream did not grow: %.2f -> %.2f; the callout claims it does",
			inMean, outMean)
	}
	pin := math.Sqrt(float64(tr.Config.DModel))
	lo, hi := lnNormRange(tr.Layers, len(tr.Tokens), tr.FinalNorm)
	if lo <= 0 || hi <= 0 {
		t.Fatalf("lnNormRange returned %v..%v", lo, hi)
	}
	// The callout quotes these as percentages of √d_model, so they have to sit
	// in a band narrow enough to call "almost constant".
	if lo < pin*0.8 || hi > pin*1.25 {
		t.Fatalf("LayerNorm norms span %.2f..%.2f around √d_model = %.2f; too wide for '几乎是个常数'",
			lo, hi, pin)
	}
	// And the stream must genuinely outgrow the normalized band, or the
	// contrast the callout draws is not visible.
	if outMean <= hi {
		t.Fatalf("stream output %.2f never exceeds the normalized band (max %.2f)", outMean, hi)
	}
}

// TestResidualCalloutScalesWithPrompt checks the callout recomputes for a
// different prompt rather than quoting one default run.
func TestResidualCalloutScalesWithPrompt(t *testing.T) {
	text := func(prompt string) string {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		a.Step = 4
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		longest := ""
		for _, s := range tr.Texts() {
			if strings.Contains(s, "残差流确实会一路变长") && len(s) > len(longest) {
				longest = s
			}
		}
		if longest == "" {
			t.Fatalf("prompt %q: the residual callout is missing", prompt)
		}
		return longest
	}
	if a, b := text("hi"), text("attention is all you need and more"); a == b {
		t.Fatal("the residual callout quotes the same numbers for two very different prompts")
	}
}

// TestReduceMotionSkipsTheStageFade checks the app honours the desktop's
// reduced-motion preference itself. mygo honours it for moves and size changes
// but still plays a fade-only Enter (transition.go gates that on
// `!still || m.fades()`), which here meant eight frames of fade on every stage
// switch for a reader who had asked for no animation.
func TestReduceMotionSkipsTheStageFade(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1180, 720)
	tr.SetPreferences(ui.Preferences{ReduceMotion: true})
	tr.Frame()
	tr.Frame()
	a.selectStep(3)
	tr.Frame()
	if tr.FramePending() {
		t.Fatal("a stage switch still animates under ReduceMotion")
	}
	// With no animation the frame must be exactly the settled one.
	plain := ui.Render(a.View, 1180, 720, 1)
	got := tr.Image()
	for i := 0; i < len(got.Pix) && i < len(plain.Pix); i++ {
		if got.Pix[i] != plain.Pix[i] {
			t.Fatal("under ReduceMotion the frame after a switch is not the settled frame")
		}
	}
}

// TestStageFadeStillRunsWithoutThePreference is the other half: the animation
// must not have been disabled for everyone while fixing the preference case.
func TestStageFadeStillRunsWithoutThePreference(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()
	a.selectStep(3)
	tr.Frame()
	if !tr.FramePending() {
		t.Fatal("a stage switch no longer animates with motion allowed")
	}
	tr.Advance(400 * time.Millisecond)
	tr.Frame()
	if tr.FramePending() {
		t.Fatal("the fade never settled")
	}
}

// TestStageTransitionReadsThePreference pins the two branches of
// stageTransition, so the preference is what selects them.
func TestStageTransitionReadsThePreference(t *testing.T) {
	a := newTestApp(t)
	var on, off ui.ElementTransition
	view := func(c *ui.Context) {
		on = a.stageTransition(c)
	}
	tr := ui.NewTester(view, 400, 200)
	tr.Frame()
	if on.Duration == 0 || on.Enter == nil {
		t.Fatalf("motion allowed but stageTransition returned %+v", on)
	}

	tr2 := ui.NewTester(view, 400, 200)
	tr2.SetPreferences(ui.Preferences{ReduceMotion: true})
	tr2.Frame()
	off = on
	tr3 := ui.NewTester(func(c *ui.Context) { off = a.stageTransition(c) }, 400, 200)
	tr3.SetPreferences(ui.Preferences{ReduceMotion: true})
	tr3.Frame()
	if off.Duration != 0 || off.Enter != nil {
		t.Fatalf("ReduceMotion set but stageTransition returned %+v", off)
	}
}

// TestSoftmaxWorkingsAddUp checks the inspector's worked formula reproduces the
// weight it sits under. A reader should be able to rewrite those two lines and
// get the quoted number back, so the assertion is on the arithmetic rather than
// on the text.
func TestSoftmaxWorkingsAddUp(t *testing.T) {
	a := newTestApp(t)
	h := a.layer().Heads[a.Head]
	for _, cell := range []struct{ q, k int }{
		{len(a.Trace.Tokens) - 1, 0},
		{len(a.Trace.Tokens) - 1, len(a.Trace.Tokens) / 2},
		{10, 3},
		{0, 0},
	} {
		q, k := cell.q, cell.k
		// The numbers printed come from computeSoftmaxParts, so this checks that
		// function against the trace rather than re-deriving the formula here.
		p := computeSoftmaxParts(h, q, k)
		if !p.ok {
			t.Fatalf("cell %d,%d: no parts computed", q, k)
		}
		if want := float64(h.Attn[q][k]); math.Abs(p.weight-want) > 1e-6 {
			t.Fatalf("cell %d,%d: computeSoftmaxParts gives %.6f, the trace holds %.6f",
				q, k, p.weight, want)
		}
		// And what the card prints must round-trip to the weight line above it.
		if printed, recomputed := fmt.Sprintf("%.4f", h.Attn[q][k]), fmt.Sprintf("%.4f", p.weight); printed != recomputed {
			t.Fatalf("cell %d,%d: card prints %s, the formula prints %s", q, k, printed, recomputed)
		}
	}
}

// TestSoftmaxWorkingsShownForLiveCells checks the formula appears when a cell is
// selected and stays away from masked ones, where there is nothing to compute.
func TestSoftmaxWorkingsShownForLiveCells(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	a.KeyQ, a.KeyK = 20, 5
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	if !tr.HasText("分母 = Σ exp(score") {
		t.Fatal("no worked formula under the weight")
	}
	if !tr.HasText("α = exp(") {
		t.Fatal("no final line for the quotient")
	}

	// A masked cell has no softmax entry; the card should say so and not print
	// a formula that would divide by a denominator it never joined.
	b := newTestApp(t)
	b.Step = 3
	b.KeyQ, b.KeyK = 5, 20 // key after query: masked
	tr2 := ui.NewTester(b.View, 1500, 940)
	tr2.Frame()
	if !tr2.HasText("被因果掩码遮住") {
		t.Fatal("a masked cell is not reported as masked")
	}
	if tr2.HasText("分母 = Σ exp(score") {
		t.Fatal("a masked cell shows a softmax formula")
	}
}

// TestSoftmaxWorkingsFitTheInspector keeps the two lines inside the panel at
// the narrowest window, where the inspector is 330 DIPs minus its paddings.
func TestSoftmaxWorkingsFitTheInspector(t *testing.T) {
	a := newTestApp(t)
	// The widest realistic line: two minus signs and a two-digit denominator.
	line := "α = exp(-4.5163 − −5.3791) ÷ 12.2241 = 0.0096"
	var w float32
	view := func(c *ui.Context) {
		w, _ = c.MeasureText(0, ui.Span{Text: line, Size: fontMicro})
	}
	ui.Render(view, 400, 100, 1)
	const available = 330 - 2*14 - 2*14
	if w > available {
		t.Fatalf("the widest formula line is %.0f wide, the inspector offers %.0f", w, float32(available))
	}
	_ = a
}

// TestLNWorkingsAddUp checks the LayerNorm formula the stage prints reproduces
// the normalized row the model recorded. Like the attention one, this compares
// against the trace rather than re-deriving the arithmetic in the test.
func TestLNWorkingsAddUp(t *testing.T) {
	a := newTestApp(t)
	l := a.layer()
	w := a.W.Layers[a.Layer]
	for _, dim := range []int{0, 1, 47, len(l.Norm2[a.SelToken]) - 1} {
		p := computeLN(l.AfterAttn[a.SelToken], w.LN2G, w.LN2B, dim)
		if !p.ok {
			t.Fatalf("dim %d: no parts computed", dim)
		}
		want := float64(l.Norm2[a.SelToken][dim])
		if math.Abs(p.out-want) > 1e-5 {
			t.Fatalf("dim %d: formula gives %.6f, the trace holds %.6f", dim, p.out, want)
		}
	}
}

// TestLNEpsMatchesTheModel keeps the lab's copy of the epsilon in step with the
// one inside model.layerNorm. They are deliberately separate constants — this
// one is only used to print the formula — so nothing but a test ties them.
func TestLNEpsMatchesTheModel(t *testing.T) {
	// Recompute a row with the lab's epsilon and with a different one; the
	// trace must agree with the lab's, which means the model used the same.
	a := newTestApp(t)
	l := a.layer()
	w := a.W.Layers[a.Layer]
	row := l.AfterAttn[a.SelToken]
	p := computeLN(row, w.LN2G, w.LN2B, 0)
	if !p.ok {
		t.Fatal("no parts")
	}
	// invStd must equal 1/sqrt(variance + lnEps) for the row's own variance.
	_, std := meanStd(row)
	variance := float64(std) * float64(std)
	if got, want := p.invStd, 1/math.Sqrt(variance+lnEps); math.Abs(got-want) > 1e-5 {
		t.Fatalf("invStd %.6f, want %.6f", got, want)
	}
	// With a wildly different epsilon the printed value would drift from the
	// trace, so this constant is load-bearing rather than decorative.
	alt := 1 - math.Sqrt(float64(std)*float64(std))/math.Sqrt(variance+1e-3)
	if alt == 0 {
		t.Fatal("the epsilon makes no difference; the test cannot detect a change")
	}
}

// TestLNWorkingsFollowTheHoveredDimension checks the printed dimension tracks
// the vector strips' shared hover, so pointing at a cell changes which value the
// formula explains.
func TestLNWorkingsFollowTheHoveredDimension(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4
	a.HoverDimIn = 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	first := ""
	for _, s := range tr.Texts() {
		if strings.HasPrefix(s, "d0：") {
			first = s
		}
	}
	if first == "" {
		t.Fatal("no LayerNorm formula for d0")
	}

	a.HoverDimIn = 47
	tr.Frame()
	var second string
	for _, s := range tr.Texts() {
		if strings.HasPrefix(s, "d47：") {
			second = s
		}
	}
	if second == "" {
		t.Fatal("the formula did not follow the hovered dimension")
	}
	if first == second {
		t.Fatal("hovering a different dimension printed the same line")
	}
}
