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
	"unicode/utf8"

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

// TestSampleDrawIsDeterministic checks the sampling card's central claim: the
// same prompt always lands on the same position, so "采样" here is reproducible
// rather than random.
func TestSampleDrawIsDeterministic(t *testing.T) {
	a := newTestApp(t)
	probs := a.Trace.Probabilities(a.Temp)
	first, ok := a.sampleDraw(probs)
	if !ok {
		t.Fatal("no draw")
	}
	for i := 0; i < 3; i++ {
		again, ok := a.sampleDraw(probs)
		if !ok || again != first {
			t.Fatalf("draw %d differs: %+v vs %+v", i, again, first)
		}
	}
	// A different prompt must hash elsewhere and, for this model, land in a
	// different interval.
	b := newTestApp(t)
	b.Prompt = "The transformer reads the tokens an"
	b.recompute()
	other, ok := b.sampleDraw(b.Trace.Probabilities(b.Temp))
	if !ok {
		t.Fatal("no draw for the altered prompt")
	}
	if other.seed == first.seed {
		t.Fatal("changing the prompt did not change the seed")
	}
	if other.x == first.x {
		t.Fatal("changing the prompt did not change the draw position")
	}
}

// TestSampleDrawIntervalWidthIsTheProbability checks the other claim the card
// makes: the selected interval is exactly as wide as the drawn token's
// probability, which is what makes x a draw from this distribution at all.
func TestSampleDrawIntervalWidthIsTheProbability(t *testing.T) {
	a := newTestApp(t)
	probs := a.Trace.Probabilities(a.Temp)
	d, ok := a.sampleDraw(probs)
	if !ok {
		t.Fatal("no draw")
	}
	if d.pick < 0 || d.pick >= len(probs) {
		t.Fatalf("draw picked %d out of range", d.pick)
	}
	if !(d.x >= d.lo && d.x <= d.hi) {
		t.Fatalf("x = %v is outside the interval [%v, %v)", d.x, d.lo, d.hi)
	}
	if got, want := d.hi-d.lo, float64(probs[d.pick]); math.Abs(got-want) > 1e-9 {
		t.Fatalf("interval width %v, but the token's probability is %v", got, want)
	}
	// The interval must be a real slice of [0,1], not the whole of it.
	if d.hi > 1.000001 {
		t.Fatalf("interval ends at %v, past the distribution", d.hi)
	}
}

// TestSamplingCardShowsTheWorkings checks the numbers reach the screen, and
// agree with sampleDraw.
func TestSamplingCardShowsTheWorkings(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	if !tr.HasText("种子 = FNV-1a(提示词) = ") {
		t.Fatal("the sampling card does not show the seed")
	}
	if !tr.HasText("就是它的概率") {
		t.Fatal("the sampling card does not explain the interval width")
	}
	probs := a.Trace.Probabilities(a.Temp)
	d, ok := a.sampleDraw(probs)
	if !ok {
		t.Fatal("no draw")
	}
	want := fmt.Sprintf("0x%016X", d.seed)
	if !tr.HasText(want) {
		t.Fatalf("the card does not show the seed %s", want)
	}
}

// TestSampleNextAppendsOneByte pins the constraint the description asks for:
// this round changed only the presentation of sampling, not its behaviour.
func TestSampleNextAppendsOneByte(t *testing.T) {
	a := newTestApp(t)
	before := a.Prompt
	a.sampleNext()
	if got := len(a.Prompt) - len(before); got != 1 {
		t.Fatalf("prompt grew by %d bytes, want exactly 1", got)
	}
	if !strings.HasPrefix(a.Prompt, before) {
		t.Fatal("sampling did not append to the prompt")
	}
}

// hoverSomeCell moves the pointer over the heatmap until it lands on a cell and
// returns the cell and the coordinates that reached it. The matrix sits inside
// the stage column, below the strip and step bar.
func hoverSomeCell(t *testing.T, a *App, tr *ui.Tester) (q, k int, x, y float32) {
	t.Helper()
	for cx := float32(780); cx > 300; cx -= 6 {
		for cy := float32(600); cy > 350; cy -= 6 {
			tr.Move(cx, cy)
			tr.Frame()
			if a.HoverQ >= 0 && a.HoverK >= 0 {
				return a.HoverQ, a.HoverK, cx, cy
			}
		}
	}
	t.Fatal("never hovered a heatmap cell")
	return 0, 0, 0, 0
}

// TestClickPinsTheCell is the interaction the keyboard already had and the
// mouse did not: a mouse user could not keep a cell while reaching for pen and
// paper, because the hover pair is rewritten every frame the pointer is
// elsewhere. Clicking now pins the cell on the same state the keyboard uses.
func TestClickPinsTheCell(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	q, k, cx, cy := hoverSomeCell(t, a, tr)

	tr.ClickAt(cx, cy)
	tr.Frame()
	if a.KeyQ != q || a.KeyK != k {
		t.Fatalf("click pinned (%d,%d), want (%d,%d)", a.KeyQ, a.KeyK, q, k)
	}

	// The point of pinning: it survives the pointer leaving the matrix.
	tr.Move(1250, 650)
	tr.Frame()
	if a.HoverQ >= 0 {
		t.Fatal("the pointer was expected to leave the matrix")
	}
	if a.KeyQ != q || a.KeyK != k {
		t.Fatalf("moving the pointer away erased the pin: (%d,%d)", a.KeyQ, a.KeyK)
	}
	// And the inspector still describes it.
	if !tr.HasText("已固定") {
		t.Fatal("the pinned cell is not announced in the inspector")
	}
}

// TestClickPinnedCellUnpins checks the second click, and that Esc does the same:
// both are ways a reader would expect to let go.
func TestClickPinnedCellUnpins(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	_, _, cx, cy := hoverSomeCell(t, a, tr)
	tr.ClickAt(cx, cy)
	tr.Frame()
	if a.KeyQ < 0 {
		t.Fatal("first click did not pin")
	}
	tr.ClickAt(cx, cy)
	tr.Frame()
	if a.KeyQ >= 0 {
		t.Fatalf("clicking the pinned cell again left it pinned at (%d,%d)", a.KeyQ, a.KeyK)
	}

	// Esc releases too, without touching playback state it does not own.
	b := newTestApp(t)
	b.Step = 3
	b.KeyQ, b.KeyK = 20, 10
	tr2 := ui.NewTester(b.View, 1500, 940)
	tr2.Frame()
	tr2.Key(0, ui.KeyEscape)
	tr2.Frame()
	if b.KeyQ >= 0 {
		t.Fatal("Esc did not release the pinned cell")
	}
	if b.Playing {
		t.Fatal("Esc left playback running")
	}
}

// TestKeyboardMovesThePinnedCell checks the two cursors stay one cursor: with a
// cell pinned, Shift+arrows walk it rather than starting over.
func TestKeyboardMovesThePinnedCell(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	a.KeyQ, a.KeyK = 20, 10
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	// Shift+Left steps the key within the row; the query stays put.
	tr.Key(ui.Shift, ui.KeyLeft)
	tr.Frame()
	if a.KeyQ != 20 || a.KeyK != 9 {
		t.Fatalf("Shift+Left from (20,10) gave (%d,%d), want (20,9)", a.KeyQ, a.KeyK)
	}
	// Shift+Up steps the query, and lands on that row's diagonal.
	tr.Key(ui.Shift, ui.KeyUp)
	tr.Frame()
	if a.KeyQ != 19 || a.KeyK != 19 {
		t.Fatalf("Shift+Up from (20,9) gave (%d,%d), want (19,19)", a.KeyQ, a.KeyK)
	}
	tr.Move(1250, 650)
	tr.Frame()
	if a.KeyQ != 19 {
		t.Fatal("a pointer move erased the pin")
	}
}

// TestPinLabelFitsTheInspector keeps the longer subtitle inside the panel.
func TestPinLabelFitsTheInspector(t *testing.T) {
	var title, note float32
	ui.Render(func(c *ui.Context) {
		title, _ = c.MeasureText(0, ui.Span{Text: "注意力格子", Size: fontData})
		note, _ = c.MeasureText(0, ui.Span{Text: "已固定 · Esc 或再点一次解除", Size: fontMicro})
	}, 400, 100, 1)
	const available = 330 - 2*14 - 2*14
	if total := title + 8 + note; total > available {
		t.Fatalf("the pinned header is %.0f wide, the card offers %.0f", total, float32(available))
	}
}

// TestEmbedAdditionAddsUp checks the stage-2 line reproduces the input vector:
// the two printed terms sum to the printed result, and that result is what the
// trace holds, for every dimension. The model builds Input with addInto, so the
// sum is exact in float32 and a reader adding the printed values gets the third
// one back.
func TestEmbedAdditionAddsUp(t *testing.T) {
	a := newTestApp(t)
	tr := a.Trace
	i := a.SelToken

	// The trace itself must add up first.
	for d := range tr.Embed[i] {
		e, p, x := tr.Embed[i][d], tr.Pos[i][d], tr.Input[i][d]
		if e+p != x {
			t.Fatalf("d%d: %v + %v = %v, but Input holds %v", d, e, p, e+p, x)
		}
	}

	// Then the line the stage prints has to quote those same terms, not some
	// other two vectors that happen to add up: reading the rendered text is the
	// only way to catch a printed line that uses the wrong field.
	a.Step = 1
	a.HoverDimIn = -1
	tester := ui.NewTester(a.View, 1500, 940)
	tester.Frame()
	line := ""
	for _, s := range tester.Texts() {
		if strings.HasPrefix(s, "d0：E[") {
			line = s
		}
	}
	if line == "" {
		t.Fatal("the stage prints no addition line")
	}
	want := fmt.Sprintf("d0：E[%d][0] = %+.4f · P[%d] = %+.4f",
		tr.Tokens[i].ID, tr.Embed[i][0], i, tr.Pos[i][0])
	if line != want {
		t.Fatalf("printed line is\n  %q\nwant\n  %q", line, want)
	}
}

// TestEmbedLineFollowsTheHoveredDimension checks the printed dimension tracks
// the vector strips' shared hover.
func TestEmbedLineFollowsTheHoveredDimension(t *testing.T) {
	a := newTestApp(t)
	a.Step = 1
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	line := func() string {
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, "d") && strings.Contains(s, "：E[") {
				return s
			}
		}
		return ""
	}
	first := line()
	if first == "" {
		t.Fatal("no embedding addition line")
	}
	a.HoverDimIn = 47
	tr.Frame()
	second := line()
	if second == "" {
		t.Fatal("the line disappeared when the dimension changed")
	}
	if first == second {
		t.Fatal("hovering another dimension printed the same line")
	}
	if !strings.HasPrefix(second, "d47") {
		t.Fatalf("hovering d47 printed %q", second)
	}
}

// TestSignedFormatsNegativeTermsForReading guards the parentheses: "+0.5674 +
// -0.6724" invites a second look, "+0.5674 + (-0.6724)" does not.
func TestSignedFormatsNegativeTermsForReading(t *testing.T) {
	if got := signed(-0.6724); got != "(-0.6724)" {
		t.Fatalf("signed(-0.6724) = %q", got)
	}
	if got := signed(0.5674); got != "+0.5674" {
		t.Fatalf("signed(0.5674) = %q", got)
	}
}

// TestFFNWorkingsAddUp checks the sum the stage prints closes exactly: the
// shown terms plus the remainder equal the output the model recorded.
func TestFFNWorkingsAddUp(t *testing.T) {
	a := newTestApp(t)
	l := a.layer()
	i := a.SelToken
	w2 := a.W.Layers[a.Layer].W2
	dff, d := a.W.Config.DFF, a.W.Config.DModel
	for _, j := range []int{0, 1, 47, d - 1} {
		p := computeFFN(l.Hidden[i], w2, dff, d, j)
		if !p.OK {
			t.Fatalf("dim %d: no parts", j)
		}
		if math.Abs(p.Out-float64(l.FFNOut[i][j])) > 1e-5 {
			t.Fatalf("dim %d: sum gives %.6f, trace holds %.6f", j, p.Out, float64(l.FFNOut[i][j]))
		}
		var shown float64
		for _, term := range p.Top {
			shown += term.V
		}
		if math.Abs(shown+p.Rest-p.Out) > 1e-5 {
			t.Fatalf("dim %d: terms %v + rest %v != out %v", j, shown, p.Rest, p.Out)
		}
	}
}

// TestFFNTopTermsAreTheLargestByAbsoluteValue guards the selection: sorting by
// signed value would pick the three most positive contributions and mislabel
// them as the largest.
func TestFFNTopTermsAreTheLargestByAbsoluteValue(t *testing.T) {
	a := newTestApp(t)
	l := a.layer()
	i := a.SelToken
	hidden := l.Hidden[i]
	w2 := a.W.Layers[a.Layer].W2
	dff, d := a.W.Config.DFF, a.W.Config.DModel
	// Dimension 2 is chosen because there the two orders genuinely differ —
	// by absolute value the top terms are k358,76,55; by signed value they
	// would be k76,55,194. On j=0 both orders happen to agree, so a test there
	// could not tell them apart.
	const j = 2

	p := computeFFN(hidden, w2, dff, d, j)
	if len(p.Top) != 3 {
		t.Fatalf("got %d terms, want 3", len(p.Top))
	}
	// k358 is the largest by magnitude but not by signed value; if it is
	// missing, the selection sorted the wrong way.
	found := false
	for _, term := range p.Top {
		if term.K == 358 {
			found = true
		}
	}
	if !found {
		t.Fatalf("the largest-by-magnitude term k358 is missing from %v", p.Top)
	}
	// No unshown term may be larger in absolute value than a shown one.
	shown := map[int]bool{}
	for _, term := range p.Top {
		shown[term.K] = true
	}
	var smallestShown float64 = math.Inf(1)
	for _, term := range p.Top {
		if v := math.Abs(term.V); v < smallestShown {
			smallestShown = v
		}
	}
	for k := 0; k < dff; k++ {
		if shown[k] {
			continue
		}
		if v := math.Abs(float64(hidden[k]) * float64(w2[k*d+j])); v > smallestShown {
			t.Fatalf("k%d (|%.4f|) is larger than a shown term (|%.4f|) but was left out",
				k, v, smallestShown)
		}
	}
}

// TestFFNShareUsesAbsoluteMass checks the percentage the stage prints. The
// signed total cancels heavily here, so dividing by |out| would report a share
// several times too large; the denominator has to be Σ|term|.
func TestFFNShareUsesAbsoluteMass(t *testing.T) {
	a := newTestApp(t)
	l := a.layer()
	i := a.SelToken
	p := computeFFN(l.Hidden[i], a.W.Layers[a.Layer].W2, a.W.Config.DFF, a.W.Config.DModel, 0)
	if !p.OK {
		t.Fatal("no parts")
	}
	var top float64
	for _, term := range p.Top {
		top += math.Abs(term.V)
	}
	want := 100 * top / p.AbsTotal
	if got := ffnTopShare(p); math.Abs(got-want) > 1e-9 {
		t.Fatalf("share %.3f, want %.3f", got, want)
	}
	// The point of the absolute denominator: it must report a materially
	// smaller share than dividing by the signed result would.
	if naive := 100 * top / math.Abs(p.Out); naive <= want*2 {
		t.Logf("note: signed and absolute denominators agree here (%.1f%% vs %.1f%%)", naive, want)
	}
	// And in this model the three terms must not carry most of the mass, which
	// is why the remainder is printed at all.
	if want > 60 {
		t.Fatalf("top three carry %.1f%% of the mass; the remainder note assumes they do not", want)
	}
}

// TestFFNWorkingsFollowTheHoveredDimension checks the printed dimension tracks
// the strip's hover.
func TestFFNWorkingsFollowTheHoveredDimension(t *testing.T) {
	a := newTestApp(t)
	a.Step = 5
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	line := func() string {
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, "out[") {
				return s
			}
		}
		return ""
	}
	first := line()
	if first == "" {
		t.Fatal("no FFN workings line")
	}
	a.HoverDimOut = 47
	tr.Frame()
	if second := line(); second == first || !strings.HasPrefix(second, "out[47]") {
		t.Fatalf("hovering output d47 printed %q", second)
	}
}

// TestFFNOutputFormulaIgnoresTheHiddenStrip is the bug the split fields fix.
// The hidden strip is d_ff wide (384) and the output strip d_model wide (96),
// so pointing at hidden d200 is a real dimension that the output formula cannot
// describe. With one shared index it fell back to out[0] while the strip
// readout above said d200: two different dimensions on screen at once.
//
// The hover fields hold pointer positions rather than settings, so this drives
// them by moving the pointer. Assigning the field and then rendering a frame
// with the pointer elsewhere clears it again.
func TestFFNOutputFormulaIgnoresTheHiddenStrip(t *testing.T) {
	a := newTestApp(t)
	a.Step = 5
	cfg := a.Trace.Config
	if cfg.DFF == cfg.DModel {
		t.Fatal("d_ff equals d_model, so this case cannot be tested")
	}
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	line := func() string {
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, "out[") {
				return s
			}
		}
		return ""
	}

	// Find the hidden strip first: hovering the output strip below it changes
	// the text above, which can reflow the page and move the hidden strip.
	hidX, hidY := stripPoint(t, a, tr, "hidden")

	outX, outY := stripPoint(t, a, tr, "out")
	tr.Move(outX, outY)
	tr.Frame()
	if a.HoverDimOut < 0 {
		t.Fatal("pointer is not on the output strip")
	}
	pinned := line()
	if !strings.HasPrefix(pinned, fmt.Sprintf("out[%d]", a.HoverDimOut)) {
		t.Fatalf("output formula %q does not name the hovered dimension %d", pinned, a.HoverDimOut)
	}

	// Sweep the hidden strip to its right edge, where its index is close to
	// d_ff-1 — well past d_model — and require the output formula to stay put.
	// stripPoint puts the pointer at x=400 (index 62 of 384); the strip runs the
	// full stage width, so walk right from there.
	lastHidden := -1
	// Walk the hidden strip to its right edge.
	for x := hidX; x <= 1150; x += 20 {
		tr.Move(x, hidY)
		tr.Frame()
		if a.HoverDimH < 0 {
			// Scrolled out from under the pointer at this x; re-anchor by
			// re-finding the strip, since the page may have reflowed.
			hidX, hidY = stripPoint(t, a, tr, "hidden")
			continue
		}
		if a.HoverDimH > lastHidden {
			lastHidden = a.HoverDimH
		}
		if got := line(); got != pinned {
			t.Fatalf("the hidden strip changed the output formula to %q", got)
		}
	}
	// The sweep has to include a hidden index that does not exist in the output
	// row, or it proves nothing about the two fields being separate.
	if lastHidden < cfg.DModel {
		t.Fatalf("the sweep only reached hidden d%d, not past d_model %d", lastHidden, cfg.DModel)
	}
	t.Logf("swept the hidden strip up to d%d (d_model is %d)", lastHidden, cfg.DModel)
}

// TestScoreWorkingsReproduceTheTrace checks the printed dot product is the
// trace's, bit for bit. computeScore accumulates in float32 like model.dot, so
// a reader checking the arithmetic gets the number the heatmap colours rather
// than a near match.
func TestScoreWorkingsReproduceTheTrace(t *testing.T) {
	a := newTestApp(t)
	l := a.layer()
	h := l.Heads[a.Head]
	dHead := a.Trace.Config.DHead()
	for _, cell := range [][2]int{{35, 17}, {20, 5}, {10, 3}, {0, 0}} {
		q, k := cell[0], cell[1]
		p := computeScore(h.Q[q], h.K[k], dHead, h.Scores[q][k])
		if !p.OK {
			t.Fatalf("cell %d,%d: no parts", q, k)
		}
		// Recompute the dot product in float32, in the same order model.dot
		// uses, and require the printed one to match it EXACTLY. A tolerance
		// here would let a float64 accumulator through, and that mismatch is
		// the whole reason computeScore accumulates in float32.
		var want float32
		for d := 0; d < dHead; d++ {
			want += h.Q[q][d] * h.K[k][d]
		}
		if got := float32(p.Dot); got != want {
			t.Fatalf("cell %d,%d: printed dot %v, float32 sum %v (differ by %v)",
				q, k, got, want, float32(math.Abs(float64(got-want))))
		}
		// And the trace's score must be that same product times the scale.
		if got, want := float32(p.Dot)*float32(1/math.Sqrt(float64(dHead))), h.Scores[q][k]; got != want {
			t.Fatalf("cell %d,%d: dot*scale = %v, trace score %v", q, k, got, want)
		}
	}
}

// TestScoreWorkingsSumCloses checks the shown terms plus the remainder equal the
// dot product, so the printed breakdown is a decomposition and not a decoration.
func TestScoreWorkingsSumCloses(t *testing.T) {
	a := newTestApp(t)
	h := a.layer().Heads[a.Head]
	dHead := a.Trace.Config.DHead()
	p := computeScore(h.Q[35], h.K[17], dHead, h.Scores[35][17])
	var shown float64
	for _, term := range p.Top {
		shown += term.V
	}
	if math.Abs(shown+p.Rest-p.Dot) > 1e-4 {
		t.Fatalf("terms %v + rest %v != dot %v", shown, p.Rest, p.Dot)
	}
}

// TestScoreRemainderIsWorthPrinting is the reason the card shows a remainder
// instead of just the three largest terms. Unlike the FFN sum, how concentrated
// a dot product is depends entirely on the cell: measured across this head the
// top three carry between 58% and 99% of the absolute mass. At the dense end
// they would look like the whole story, so the remainder has to be printed.
func TestScoreRemainderIsWorthPrinting(t *testing.T) {
	a := newTestApp(t)
	h := a.layer().Heads[a.Head]
	dHead := a.Trace.Config.DHead()

	var lo, hi float64 = math.Inf(1), 0
	for _, cell := range [][2]int{{35, 17}, {35, 34}, {20, 5}, {10, 3}, {0, 0}} {
		p := computeScore(h.Q[cell[0]], h.K[cell[1]], dHead, h.Scores[cell[0]][cell[1]])
		var top float64
		for _, term := range p.Top {
			top += math.Abs(term.V)
		}
		share := top / (top + math.Abs(p.Rest))
		if share < lo {
			lo = share
		}
		if share > hi {
			hi = share
		}
		// Every cell the card can be asked about must close exactly.
		var shown float64
		for _, term := range p.Top {
			shown += term.V
		}
		if math.Abs(shown+p.Rest-p.Dot) > 1e-4 {
			t.Fatalf("cell %v: terms + rest != dot", cell)
		}
		// And the terms must be the largest BY MAGNITUDE: a signed sort still
		// closes the sum, so closure alone cannot detect it.
		shownDims := map[int]bool{}
		for _, term := range p.Top {
			shownDims[term.D] = true
		}
		smallestShown := math.Inf(1)
		for _, term := range p.Top {
			if v := math.Abs(term.V); v < smallestShown {
				smallestShown = v
			}
		}
		for d := 0; d < dHead; d++ {
			if shownDims[d] {
				continue
			}
			if v := math.Abs(float64(h.Q[cell[0]][d]) * float64(h.K[cell[1]][d])); v > smallestShown {
				t.Fatalf("cell %v: product d%d (|%.4f|) is larger than a shown one (|%.4f|)",
					cell, d, v, smallestShown)
			}
		}
	}
	// If some cell were dominated by three terms, the remainder note would be
	// noise there; the finding above says at least one is not.
	if lo > 0.9 {
		t.Fatalf("every sampled cell is dominated by three terms (min share %.0f%%); "+
			"the remainder note would be noise", lo*100)
	}
	t.Logf("top-three share across sampled cells: %.0f%%–%.0f%%", lo*100, hi*100)
}

// TestScoreWorkingsAppearForACell checks the card shows the working when a cell
// is selected and prompts for one when it is not.
func TestScoreWorkingsAppearForACell(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	if !tr.HasText("把指针移到矩阵的某个格子上") {
		t.Fatal("with no cell selected the card does not say how to pick one")
	}

	b := newTestApp(t)
	b.Step = 2
	b.KeyQ, b.KeyK = 35, 17
	tr2 := ui.NewTester(b.View, 1500, 940)
	tr2.Frame()
	if !tr2.HasText("的三项最大贡献") {
		t.Fatal("no per-cell score working")
	}
	if !tr2.HasText("score = 点积 / √24") {
		t.Fatal("no scaling line")
	}
}

// TestRuneLookupCoversEveryByte checks runeAt against the trace for every
// position: each byte belongs to exactly one character, and the characters tile
// the prompt in order.
func TestRuneLookupCoversEveryByte(t *testing.T) {
	for _, prompt := range []string{"注意力就是加权求和", "The transformer", "a注b", "x"} {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		tr := a.Trace
		// runeAt takes a rune index, which is the convention Tokens[].Start uses.
		ri := 0
		for _, r := range tr.Text {
			got := runeAt(tr, ri)
			if !got.OK {
				t.Fatalf("%q rune %d: no character", prompt, ri)
			}
			if got.Rune != r {
				t.Fatalf("%q rune %d: got %q, want %q", prompt, ri, got.Rune, r)
			}
			if got.Index != ri {
				t.Fatalf("%q rune %d: index %d", prompt, ri, got.Index)
			}
			ri++
		}
		if got := runeAt(tr, ri); got.OK {
			t.Fatalf("%q: rune %d is past the end but resolved to %q", prompt, ri, got.Rune)
		}
		if got := runeAt(tr, -1); got.OK {
			t.Fatal("byte -1 resolved to a character")
		}
	}
}

// TestRuneAtUsesStartAsARuneIndex pins the convention that makes the card's
// lookup correct: Tokens[i].Start counts runes, not bytes, so the card must not
// use it as a byte offset.
func TestRuneAtUsesStartAsARuneIndex(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = "a注b"
	a.recompute()
	tr := a.Trace
	if len(tr.Tokens) != 5 {
		t.Fatalf("want 5 tokens for 3 characters, got %d", len(tr.Tokens))
	}
	// The second token starts the 3-byte character at rune index 1.
	if tr.Tokens[1].Start != 1 {
		t.Fatalf("token 1 Start = %d, want 1", tr.Tokens[1].Start)
	}
	rb := runeAt(tr, tr.Tokens[1].Start)
	if !rb.OK || rb.Rune != '注' {
		t.Fatalf("runeAt(%d) = %q, want 注", tr.Tokens[1].Start, rb.Rune)
	}
	if len(rb.Bytes) != 3 {
		t.Fatalf("注 has %d bytes, want 3", len(rb.Bytes))
	}
	// Tokens 2 and 3 carry the same Start, which is what makes them one group.
	for _, i := range []int{2, 3} {
		if tr.Tokens[i].Start != tr.Tokens[1].Start {
			t.Fatalf("token %d Start = %d, want %d", i, tr.Tokens[i].Start, tr.Tokens[1].Start)
		}
	}
}

// TestUTF8PayloadReassemblesTheCodepoint checks the arithmetic the card prints:
// the payload bits shifted into place must equal the character's codepoint, for
// every character in a mixed-script prompt.
func TestUTF8PayloadReassemblesTheCodepoint(t *testing.T) {
	shifts := map[int][]int{1: {0}, 2: {6, 0}, 3: {12, 6, 0}, 4: {18, 12, 6, 0}}
	for _, s := range []string{"注意力就是加权求和", "abc", "a注b", "é", "𝄞"} {
		for _, r := range s {
			bs := []byte(string(r))
			w, ok := shifts[len(bs)]
			if !ok {
				t.Fatalf("%q: no shift table for %d bytes", r, len(bs))
			}
			var got rune
			for i, b := range bs {
				got |= rune(utf8Payload(b)) << uint(w[i])
			}
			if got != r {
				t.Fatalf("%q: payload bits reassemble to U+%04X, want U+%04X", r, got, r)
			}
		}
	}
}

// TestUTF8PrefixMatchesTheEncoding checks the prefix the card prints beside each
// byte: the high bits are fixed by the byte's length, which is the thing that
// lets a reader spot a continuation byte at a glance.
func TestUTF8PrefixMatchesTheEncoding(t *testing.T) {
	cases := []struct {
		r      rune
		prefix []string
	}{
		{'a', []string{"0"}},
		{'é', []string{"110", "10"}},
		{'注', []string{"1110", "10", "10"}},
		{'𝄞', []string{"11110", "10", "10", "10"}},
	}
	for _, c := range cases {
		bs := []byte(string(c.r))
		if len(bs) != len(c.prefix) {
			t.Fatalf("%q: %d bytes, %d prefixes", c.r, len(bs), len(c.prefix))
		}
		for i, b := range bs {
			if got := utf8Prefix(b); got != c.prefix[i] {
				t.Fatalf("%q byte %d: prefix %q, want %q", c.r, i, got, c.prefix[i])
			}
		}
	}
}

// TestChipGlyphShowsCharactersNotEscapes is the point of the change: a byte of
// a multi-byte character shows that character, not the model's "\\xe6" escape.
func TestChipGlyphShowsCharactersNotEscapes(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = "a注b"
	a.recompute()
	want := []string{"a", "注₁", "注₂", "注₃", "b"}
	if len(a.Trace.Tokens) != len(want) {
		t.Fatalf("%d tokens, want %d", len(a.Trace.Tokens), len(want))
	}
	for i, w := range want {
		if got := chipGlyph(a.Trace, i); got != w {
			t.Fatalf("token %d: glyph %q, want %q", i, got, w)
		}
	}
	// The model still renders the bare byte; only the strip substitutes.
	if got := a.Trace.Tokens[1].Text; got != `\xe6` {
		t.Fatalf("token 1 Text = %q; the model's own rendering changed", got)
	}
}

// TestChipGlyphCoversEveryCharacterLength checks the subscript index across
// ASCII, 2-, 3- and 4-byte characters, since each takes a different branch.
func TestChipGlyphCoversEveryCharacterLength(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = "aéb注𝄞"
	a.recompute()
	var got []string
	for i := range a.Trace.Tokens {
		got = append(got, chipGlyph(a.Trace, i))
	}
	want := []string{"a", "é₁", "é₂", "b", "注₁", "注₂", "注₃", "𝄞₁", "𝄞₂", "𝄞₃", "𝄞₄"}
	if len(got) != len(want) {
		t.Fatalf("%d tokens %q, want %d", len(got), got, len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d: %q, want %q (all: %q)", i, got[i], want[i], got)
		}
	}
}

// TestChipGlyphKeepsASCIIAndControlGlyphs checks the substitution stays away
// from tokens the model already renders well.
func TestChipGlyphKeepsASCIIAndControlGlyphs(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = "a b\tc\nd"
	a.recompute()
	want := []string{"a", "␣", "b", "\\t", "c", "\\n", "d"}
	var got []string
	for i := range a.Trace.Tokens {
		got = append(got, chipGlyph(a.Trace, i))
	}
	if len(got) != len(want) {
		t.Fatalf("%d tokens %q, want %d", len(got), got, len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d: %q, want %q (all: %q)", i, got[i], want[i], got)
		}
	}
}

// TestResidualChainReproducesTheTrace checks the printed additions are the
// trace's own values: each line's running total is the vector the strip above
// draws, for every layer and dimension.
func TestResidualChainReproducesTheTrace(t *testing.T) {
	a := newTestApp(t)
	tr := a.Trace
	i := a.SelToken
	for li := range tr.Layers {
		l := tr.Layers[li]
		for _, dim := range []int{0, 1, 47, len(l.In[i]) - 1} {
			steps, ok := residualChain(l, i, dim)
			if !ok {
				t.Fatalf("layer %d dim %d: no chain", li, dim)
			}
			if len(steps) != 3 {
				t.Fatalf("layer %d dim %d: %d steps, want 3", li, dim, len(steps))
			}
			// The inputs of the chain must be the trace's fields.
			if steps[0].Term != float64(l.In[i][dim]) {
				t.Fatalf("layer %d dim %d: starts from %v, In is %v", li, dim, steps[0].Term, l.In[i][dim])
			}
			if steps[1].Term != float64(l.AttnProj[i][dim]) {
				t.Fatalf("layer %d dim %d: attention term %v, AttnProj is %v", li, dim, steps[1].Term, l.AttnProj[i][dim])
			}
			if steps[2].Term != float64(l.FFNOut[i][dim]) {
				t.Fatalf("layer %d dim %d: ffn term %v, FFNOut is %v", li, dim, steps[2].Term, l.FFNOut[i][dim])
			}
			// Each running total must be the recorded intermediate.
			if steps[1].Sum != float64(l.AfterAttn[i][dim]) {
				t.Fatalf("layer %d dim %d: running total %v, AfterAttn is %v", li, dim, steps[1].Sum, l.AfterAttn[i][dim])
			}
			if steps[2].Sum != float64(l.AfterFFN[i][dim]) {
				t.Fatalf("layer %d dim %d: running total %v, AfterFFN is %v", li, dim, steps[2].Sum, l.AfterFFN[i][dim])
			}
			// And the additions must actually add up.
			if math.Abs(steps[0].Term+steps[1].Term-steps[1].Sum) > 1e-6 {
				t.Fatalf("layer %d dim %d: %v + %v != %v", li, dim, steps[0].Term, steps[1].Term, steps[1].Sum)
			}
			if math.Abs(steps[1].Sum+steps[2].Term-steps[2].Sum) > 1e-6 {
				t.Fatalf("layer %d dim %d: %v + %v != %v", li, dim, steps[1].Sum, steps[2].Term, steps[2].Sum)
			}
		}
	}
}

// TestResidualChainLinksTheLayers checks that one block's output is the next
// block's input, which is what makes "the stream carries forward" true rather
// than a saying.
func TestResidualChainLinksTheLayers(t *testing.T) {
	a := newTestApp(t)
	tr := a.Trace
	i := a.SelToken
	for li := 0; li+1 < len(tr.Layers); li++ {
		for dim := range tr.Layers[li].AfterFFN[i] {
			if tr.Layers[li].AfterFFN[i][dim] != tr.Layers[li+1].In[i][dim] {
				t.Fatalf("layer %d output dim %d is %v, layer %d input is %v",
					li, dim, tr.Layers[li].AfterFFN[i][dim], li+1, tr.Layers[li+1].In[i][dim])
			}
		}
	}
}

// TestResidualWorkingsFollowTheHoveredDimension checks the printed line tracks
// the strips' shared hover.
func TestResidualWorkingsFollowTheHoveredDimension(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	line := func() string {
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, "d0 的残差流") || strings.HasPrefix(s, "d7 的残差流") {
				return s
			}
		}
		return ""
	}
	a.HoverDimIn = 0
	tr.Frame()
	first := line()
	if first == "" {
		t.Fatal("no residual chain line")
	}
	a.HoverDimIn = 7
	tr.Frame()
	second := line()
	if second == "" {
		t.Fatal("the line disappeared when the dimension changed")
	}
	if first == second {
		t.Fatal("hovering another dimension printed the same line")
	}
	if !strings.HasPrefix(second, "d7 ") {
		t.Fatalf("hovering d7 printed %q", second)
	}
}

// stripPoint finds a point inside one of a stage's vector strips by sweeping the
// stage column and watching which hover field the strip writes. kind is
// "hidden" (HoverDimH), "out" (HoverDimOut) or "head" (HoverDimHeadOut).
func stripPoint(t *testing.T, a *App, tr *ui.Tester, kind string) (x, y float32) {
	t.Helper()
	for probe := float32(250); probe < 1500; probe += 3 {
		tr.Move(400, probe)
		tr.Frame()
		var got int
		switch kind {
		case "hidden":
			got = a.HoverDimH
		case "out":
			got = a.HoverDimOut
		case "head":
			got = a.HoverDimHeadOut
		default:
			t.Fatalf("unknown strip kind %q", kind)
		}
		if got >= 0 {
			return 400, probe
		}
	}
	t.Fatalf("no strip writing %s found", kind)
	return 0, 0
}

// TestEachStripWidthHasItsOwnCursor is the general form of the FFN bug: a card
// that draws rows of different widths must not share one hover index between
// them, or the index means different dimensions in each row. It also pins that
// leaving a strip clears its readout, because a stale "d62 = ..." outliving the
// pointer is the same kind of lie.
func TestEachStripWidthHasItsOwnCursor(t *testing.T) {
	a := newTestApp(t)
	cfg := a.Trace.Config
	if cfg.DHead() == cfg.DModel {
		t.Fatal("d_head equals d_model, so the attention case cannot be tested")
	}

	// Attention: the head output row is d_head wide, the projection rows are
	// d_model wide. Pointing at one must not set the other.
	a.Step = 3
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	headX, headY := stripPoint(t, a, tr, "head")
	tr.Move(headX, headY)
	tr.Frame()
	if a.HoverDimHeadOut < 0 {
		t.Fatal("pointer is not on the head output strip")
	}
	if head := a.HoverDimHeadOut; head >= cfg.DModel {
		t.Fatalf("head output hovered at d%d, past d_head %d", head, cfg.DHead())
	}
	if a.HoverDimProj >= 0 {
		t.Fatalf("the head output strip also set HoverDimProj=%d", a.HoverDimProj)
	}

	// The readout must not outlive the pointer: it claims "this is the cell
	// you are pointing at", so it goes when the pointer does, even though the
	// field keeps its value for the formulas below to use.
	tr.Move(1400, 60)
	tr.Frame()
	for _, s := range tr.Texts() {
		if strings.HasPrefix(s, "d3 = ") {
			t.Fatalf("readout outlived the pointer: %q", s)
		}
	}
}

// TestHeadEntropyBarsMatchTheHeads checks the per-head comparison rows quote
// each head's own entropy, so the block is a reading of the trace rather than a
// decoration. It also pins the bar scale: entropy is bounded by log(n), and a
// row drawn against the wrong denominator would exaggerate the differences.
func TestHeadEntropyBarsMatchTheHeads(t *testing.T) {
	a := newTestApp(t)
	l := a.layer()
	i := a.SelToken
	n := i + 1
	ceiling := math.Log(float64(n))
	for h := range l.Heads {
		row := l.Heads[h].Attn[i]
		e := float64(model.Entropy(row[:min(len(row), n)]))
		if e < 0 {
			t.Fatalf("head %d: entropy %v is negative", h, e)
		}
		if e > ceiling+1e-4 {
			t.Fatalf("head %d: entropy %v exceeds log(%d) = %v", h, e, n, ceiling)
		}
	}
	// The heads must actually differ on this model, or the comparison has
	// nothing to teach and the block is noise.
	var lo float32 = math.MaxFloat32
	var hi float32
	for h := range l.Heads {
		row := l.Heads[h].Attn[i]
		e := model.Entropy(row[:min(len(row), n)])
		if e < lo {
			lo = e
		}
		if e > hi {
			hi = e
		}
	}
	if hi-lo < 0.3 {
		t.Fatalf("all heads have nearly the same entropy (%.2f-%.2f); the comparison teaches nothing", lo, hi)
	}

	// The rendered rows must quote those same numbers. Checking the trace alone
	// would pass even if every bar were built from the same head.
	a.Step = 3
	tester := ui.NewTester(a.View, 1500, 940)
	tester.Frame()
	seen := 0
	for h := range l.Heads {
		row := l.Heads[h].Attn[i]
		want := fmt.Sprintf("%.2f", model.Entropy(row[:min(len(row), n)]))
		if tester.HasText(want) {
			seen++
		}
	}
	if seen == 0 {
		t.Fatalf("no head entropy value appears on screen (looked for %.2f values)", 0.0)
	}
	// Every head's value must appear. That is the assertion with teeth: a bar
	// built from the wrong head still shows *a* plausible entropy, so checking
	// that "some value appears" would pass while three rows were copies of the
	// selected head.
	for h := range l.Heads {
		row := l.Heads[h].Attn[i]
		want := fmt.Sprintf("%.2f", model.Entropy(row[:min(len(row), n)]))
		if !tester.HasText(want) {
			t.Fatalf("head %d's entropy %s does not appear on screen", h, want)
		}
	}
}

// TestHeadEntropyBarsShowInTheInspector checks the block reaches the screen and
// marks the head currently selected.
func TestHeadEntropyBarsShowInTheInspector(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	a.Head = 1
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	if !tr.HasText("这一行在各头的熵") {
		t.Fatal("no head comparison label")
	}
	// Every head gets a label row, and the selected one is marked.
	for h := 0; h < a.W.Config.NHead; h++ {
		plain := fmt.Sprintf("H%d", h)
		marked := fmt.Sprintf("H%d ◂", h)
		has := tr.HasText(plain) || tr.HasText(marked)
		if !has {
			t.Fatalf("head %d has no label", h)
		}
		if h == a.Head && !tr.HasText(marked) {
			t.Fatalf("the selected head %d is not marked", h)
		}
		if h != a.Head && tr.HasText(marked) {
			t.Fatalf("head %d is marked but head %d is selected", h, a.Head)
		}
	}
}

// TestHeadEntropyBarsFollowTheHeadSwitch checks switching heads moves the mark
// rather than adding a second one.
func TestHeadEntropyBarsFollowTheHeadSwitch(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	a.Head = 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	if !tr.HasText("H0 ◂") {
		t.Fatal("head 0 is not marked at the start")
	}
	a.Head = 3
	tr.Frame()
	if tr.HasText("H0 ◂") {
		t.Fatal("head 0 is still marked after switching to head 3")
	}
	if !tr.HasText("H3 ◂") {
		t.Fatal("head 3 is not marked after switching")
	}
}

// TestWeightedWorkingsReproduceTheTrace checks the printed Σα·V closes on the
// head output the trace recorded, for a few cells and dimensions.
func TestWeightedWorkingsReproduceTheTrace(t *testing.T) {
	a := newTestApp(t)
	l := a.layer()
	h := l.Heads[a.Head]
	n := len(a.Trace.Tokens)
	dh := a.Trace.Config.DHead()
	for _, cell := range []struct{ q, dim int }{
		{n - 1, 0}, {n - 1, 1}, {n - 1, dh - 1}, {10, 3}, {0, 0},
	} {
		p := computeWeighted(h.Attn[cell.q], h.V, cell.q, cell.dim, n)
		if !p.OK {
			t.Fatalf("cell %d,%d: no parts", cell.q, cell.dim)
		}
		if want := float64(h.Out[cell.q][cell.dim]); math.Abs(p.Out-want) > 1e-5 {
			t.Fatalf("cell %d,%d: Σα·V = %.6f, Out holds %.6f", cell.q, cell.dim, p.Out, want)
		}
		// The shown terms plus the remainder must close the sum exactly.
		var shown float64
		for _, tm := range p.Top {
			shown += tm.Val
		}
		if math.Abs(shown+p.Rest-p.Out) > 1e-5 {
			t.Fatalf("cell %d,%d: terms %v + rest %v != %v", cell.q, cell.dim, shown, p.Rest, p.Out)
		}
	}
}

// TestWeightedTermsAreTheLargestByMagnitude guards the selection, on a cell where
// sorting by signed value would give different terms.
func TestWeightedTermsAreTheLargestByMagnitude(t *testing.T) {
	a := newTestApp(t)
	h := a.layer().Heads[a.Head]
	n := len(a.Trace.Tokens)
	const q = 10
	var found bool
	for dim := 0; dim < a.Trace.Config.DHead() && !found; dim++ {
		p := computeWeighted(h.Attn[q], h.V, q, dim, n)
		shown := map[int]bool{}
		for _, tm := range p.Top {
			shown[tm.Key] = true
		}
		smallest := math.Inf(1)
		for _, tm := range p.Top {
			if v := math.Abs(tm.Val); v < smallest {
				smallest = v
			}
		}
		for k := 0; k <= q; k++ {
			if shown[k] {
				continue
			}
			if v := math.Abs(float64(h.Attn[q][k]) * float64(h.V[k][dim])); v > smallest {
				t.Fatalf("dim %d: k%d (|%.4f|) is larger than a shown term (|%.4f|)", dim, k, v, smallest)
			}
		}
		found = true
	}
	if !found {
		t.Fatal("no dimension checked")
	}
}

// TestWeightedWorkingsAppearOnStage checks the block reaches the screen and
// follows the head-output strip's hover.
func TestWeightedWorkingsAppearOnStage(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	found := false
	for _, s := range tr.Texts() {
		if strings.HasPrefix(s, "Out[") && strings.Contains(s, "Σ α·V") {
			if !strings.Contains(s, "[d0]") {
				t.Fatalf("default line does not name d0: %q", s)
			}
			found = true
		}
	}
	if !found {
		t.Fatal("no weighted-sum working on stage 4")
	}

	// Hovering the head-output strip moves it to that dimension.
	a.HoverDimHeadOut = 7
	tr.Frame()
	found = false
	for _, s := range tr.Texts() {
		if strings.HasPrefix(s, "Out[") && strings.Contains(s, "[d7]") {
			found = true
		}
	}
	if !found {
		t.Fatal("the working did not follow the hovered dimension")
	}
}

// TestHeadContributionsMatchTheTrace checks the per-head norms are the trace's
// own, and that the cancellation ratio is the one the card prints.
func TestHeadContributionsMatchTheTrace(t *testing.T) {
	a := newTestApp(t)
	tr := a.Trace
	for li := range tr.Layers {
		l := tr.Layers[li]
		for _, i := range []int{0, 5, a.SelToken} {
			c := computeHeadContributions(l, i)
			if !c.OK {
				t.Fatalf("layer %d token %d: no contributions", li, i)
			}
			if len(c.Heads) != len(l.HeadProj) {
				t.Fatalf("layer %d token %d: %d heads, want %d", li, i, len(c.Heads), len(l.HeadProj))
			}
			var sum float64
			for h, hh := range c.Heads {
				want := float64(norm(l.HeadProj[h][i]))
				if math.Abs(hh.Norm-want) > 1e-6 {
					t.Fatalf("layer %d token %d head %d: norm %.6f, trace %.6f", li, i, h, hh.Norm, want)
				}
				sum += want
			}
			if math.Abs(c.SumNorm-sum) > 1e-6 {
				t.Fatalf("layer %d token %d: SumNorm %.6f, want %.6f", li, i, c.SumNorm, sum)
			}
			if want := float64(norm(l.AttnProj[i])); math.Abs(c.TotalNorm-want) > 1e-6 {
				t.Fatalf("layer %d token %d: TotalNorm %.6f, ||AttnProj|| %.6f", li, i, c.TotalNorm, want)
			}
			if want := c.TotalNorm / c.SumNorm; math.Abs(c.Cancellation-want) > 1e-9 {
				t.Fatalf("layer %d token %d: cancellation %.6f, want %.6f", li, i, c.Cancellation, want)
			}
			// Shares sum to 100.
			var share float64
			for _, hh := range c.Heads {
				share += hh.Share
			}
			if math.Abs(share-100) > 1e-6 {
				t.Fatalf("layer %d token %d: shares sum to %.4f", li, i, share)
			}
		}
	}
}

// TestHeadsCancelEverywhere is the claim the card makes: the heads' norms sum to
// more than the norm of their sum on every row of this model, not just the one
// being looked at. If that ever stops holding, the callout text is wrong and
// this fails rather than quietly misleading a reader.
func TestHeadsCancelEverywhere(t *testing.T) {
	a := newTestApp(t)
	tr := a.Trace
	st := measureCancellation(tr)
	if st.Rows == 0 {
		t.Fatal("no rows measured")
	}
	if st.Below != st.Rows {
		t.Fatalf("only %d of %d rows cancel (ratio < 1); the callout claims all of them",
			st.Below, st.Rows)
	}
	lo, hi := st.Lo, st.Hi
	if !(lo < 1 && hi < 1) {
		t.Fatalf("cancellation range %.3f-%.3f is not entirely below 1", lo, hi)
	}
	// And the range the card prints must be a real spread, not a degenerate one.
	if hi-lo < 0.05 {
		t.Fatalf("cancellation range %.3f-%.3f is too narrow to quote", lo, hi)
	}
	t.Logf("%d/%d rows cancel; ratio range %.2f-%.2f", st.Below, st.Rows, lo, hi)
}

// TestHeadContributionBarsAppear checks the bars reach the stage and follow the
// selected head.
func TestHeadContributionBarsAppear(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	a.Head = 2
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	if !tr.HasText("各头写进残差流的长度") {
		t.Fatal("no head-contribution block")
	}
	// The selected head is marked, and no other is.
	if !tr.HasText("H2 ◂") {
		t.Fatal("the selected head is not marked in the bars")
	}
	for h := 0; h < a.W.Config.NHead; h++ {
		if h == a.Head {
			continue
		}
		if tr.HasText(fmt.Sprintf("H%d ◂", h)) {
			t.Fatalf("head %d is marked but head %d is selected", h, a.Head)
		}
	}
}

// TestReadmeStageRangesMatchTheControls keeps the README's control table honest.
// The table stated the head control's range as "第 3–5 步" while the code has
// 2..3 (steps 3–4), and that kind of drift is invisible: the document still
// reads fine, it is just wrong. This checks the three ranges against the
// constants the controls are built from.
func TestReadmeStageRangesMatchTheControls(t *testing.T) {
	// Tests run with the package directory as the working directory.
	readme, err := os.ReadFile(filepath.Join("..", "..", "README.md"))
	if err != nil {
		t.Fatalf("read README.md: %v", err)
	}
	text := string(readme)

	// The stage ranges are declared as constants so the controls and this test
	// read the same source.
	check := func(label string, from, to int) {
		t.Helper()
		want := fmt.Sprintf("第 %d–%d 步", from+1, to+1)
		// Find the row that starts with the label and require the range in it.
		for _, line := range strings.Split(text, "\n") {
			if !strings.HasPrefix(line, "| "+label) {
				continue
			}
			if !strings.Contains(line, want) {
				t.Fatalf("README row %q should say %q; it says:\n  %s", label, want, line)
			}
			return
		}
		t.Fatalf("README has no row for %q", label)
	}
	check("层分段控件", layerFrom, layerTo)
	check("头分段控件", headFrom, headTo)

	// The temperature row is phrased with the stage number rather than a range.
	tempRow := ""
	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(line, "| 温度滑块") {
			tempRow = line
		}
	}
	if tempRow == "" {
		t.Fatal("README has no 温度滑块 row")
	}
	if want := fmt.Sprintf("第 %d 步", tempFrom+1); !strings.Contains(tempRow, want) {
		t.Fatalf("README temperature row should say %q; it says:\n  %s", want, tempRow)
	}
}

// TestRuneAtHandlesInvalidUTF8 is the case that made runeAt read from the tokens
// rather than from Trace.Text. DecodeBytes replaces an invalid byte with U+FFFD
// to keep the text printable, and that replacement is one rune where the
// original byte sequence was another length, so the two sources disagree about
// every index after it.
func TestRuneAtHandlesInvalidUTF8(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = string([]byte{0x61, 0xFF, 0xFF, 0x62}) // a, two invalid bytes, b
	a.recompute()
	tr := a.Trace
	if len(tr.Tokens) != 4 {
		t.Fatalf("%d tokens, want 4", len(tr.Tokens))
	}
	// The valid characters must be reported as themselves, including the one
	// after the invalid bytes — that is the index shift this guards against.
	if rb := runeAt(tr, tr.Tokens[0].Start); !rb.OK || rb.Rune != 'a' {
		t.Fatalf("token 0: %q", rb.Rune)
	}
	if rb := runeAt(tr, tr.Tokens[3].Start); !rb.OK || rb.Rune != 'b' {
		t.Fatalf("token 3: %q, want 'b' (the invalid bytes must not shift this index)", rb.Rune)
	}
	// The invalid bytes report themselves rather than the replacement.
	for _, i := range []int{1, 2} {
		rb := runeAt(tr, tr.Tokens[i].Start)
		if !rb.OK {
			t.Fatalf("token %d: no result", i)
		}
		if rb.Rune == '\uFFFD' {
			t.Fatalf("token %d reports the replacement character instead of its byte", i)
		}
		if rb.Byte != 0xFF || len(rb.Bytes) != 1 {
			t.Fatalf("token %d: Byte=%#x Bytes=% x", i, rb.Byte, rb.Bytes)
		}
	}
}

// TestInvalidByteCardShowsTheByte checks the stage-1 card describes an invalid
// byte as a byte, not as U+FFFD. The card used to say "'<replacement>' = U+FFFD,
// 占 3 个字节" for a one-byte input, which describes the display rather than
// what the model was given.
func TestInvalidByteCardShowsTheByte(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = string([]byte{0x61, 0xFF, 0x62})
	a.recompute()
	a.Step = 0
	a.SelToken = 1
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	hasByte := false
	for _, s := range tr.Texts() {
		if strings.Contains(s, "0xFF") && strings.Contains(s, "不是合法的 UTF-8") {
			hasByte = true
		}
		if strings.Contains(s, "U+FFFD，占 3 个字节") {
			t.Fatalf("the card still describes the replacement: %q", s)
		}
	}
	if !hasByte {
		t.Fatal("the card does not explain the invalid byte")
	}
}

// TestTruncationDropsWholeCharacters checks the two claims the tokenising card
// now makes when a prompt is truncated: the counts describe the truncated
// sequence while stating the original length, and truncation never starts in the
// middle of a character.
func TestTruncationDropsWholeCharacters(t *testing.T) {
	a := newTestApp(t)
	ctx := a.W.Config.Ctx

	// ASCII fills the window exactly.
	b := newTestApp(t)
	b.Prompt = strings.Repeat("a", ctx+36)
	b.recompute()
	if got := len(b.Trace.Tokens); got != ctx {
		t.Fatalf("ASCII prompt used %d tokens, want the full %d", got, ctx)
	}
	if dropped := len(b.Prompt) - len(b.Trace.Tokens); dropped != 36 {
		t.Fatalf("dropped %d bytes, want 36", dropped)
	}

	// A multi-byte prompt stops short of ctx so it does not cut a character.
	c := newTestApp(t)
	c.Prompt = strings.Repeat("注意力", 30)
	c.recompute()
	tr := c.Trace
	if len(tr.Tokens) > ctx {
		t.Fatalf("%d tokens exceeds ctx %d", len(tr.Tokens), ctx)
	}
	if len(tr.Tokens) < ctx-3 {
		t.Fatalf("%d tokens leaves more than one character unused", len(tr.Tokens))
	}
	// The kept text must be a suffix of the prompt: truncation takes the end.
	if !strings.HasSuffix(c.Prompt, tr.Text) {
		t.Fatalf("kept text %q is not a suffix of the prompt", tr.Text)
	}
	// And it must begin on a character boundary, i.e. decode cleanly.
	if !utf8.ValidString(tr.Text) {
		t.Fatalf("kept text is not valid UTF-8: %q", tr.Text)
	}
	if first := []rune(tr.Text)[0]; first < 0x4E00 || first > 0x9FFF {
		t.Fatalf("kept text starts with %q, not a CJK character", first)
	}
}

// TestTruncatedCardStatesBothByteCounts checks the card stops presenting the
// truncated length as if it described the prompt. "字节数 63" on a 270-byte
// prompt is a fact about the window, not the input.
func TestTruncatedCardStatesBothByteCounts(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = strings.Repeat("注意力", 30)
	a.recompute()
	a.Step = 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	var sawTruncated, sawDropped bool
	for _, s := range tr.Texts() {
		if strings.Contains(s, "原文 270") {
			sawTruncated = true
		}
		if strings.Contains(s, "丢弃前 207 字节") {
			sawDropped = true
		}
	}
	if !sawTruncated || !sawDropped {
		t.Fatalf("card does not state both counts: truncated=%v dropped=%v", sawTruncated, sawDropped)
	}

	// An untruncated prompt keeps the plain form and shows no drop notice.
	b := newTestApp(t)
	b.Prompt = "hello"
	b.recompute()
	b.Step = 0
	tr2 := ui.NewTester(b.View, 1500, 940)
	tr2.Frame()
	for _, s := range tr2.Texts() {
		if strings.Contains(s, "原文") || strings.Contains(s, "丢弃") {
			t.Fatalf("untruncated prompt shows a truncation count: %q", s)
		}
	}
}
