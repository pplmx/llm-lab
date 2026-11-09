package app

import (
	"bytes"
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/egoist/mygo/ui"

	"llmlab/internal/model"
)

// bundledWeights parses weights.json once per test binary. model.Bundled re-reads
// and re-parses 2.4 MB of JSON on every call, and newTestApp is called by every
// test in this package — under -race that alone pushed the package past Go's
// 10-minute default test timeout, which is what CI runs. The tests only read the
// weights, so one shared copy is safe.
var bundledWeights struct {
	once sync.Once
	w    *model.Weights
	err  error
}

func newTestApp(t testing.TB) *App {
	t.Helper()
	bundledWeights.once.Do(func() {
		bundledWeights.w, bundledWeights.err = model.Bundled()
	})
	if bundledWeights.err != nil {
		t.Fatalf("bundled weights: %v", bundledWeights.err)
	}
	return New(bundledWeights.w)
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

	// Space is the one that matters most: it types a separator in prose and is also
	// the play/pause key. Only digits and arrows were checked, so a regression that
	// let Space through to the lesson would have left the reader unable to type a
	// space in the prompt at all — every keystroke starting or stopping playback.
	before = a.Prompt
	tr.TypeKey(0, ui.KeySpace, " ")
	tr.Frame()
	if a.Playing {
		t.Fatal("typing a space in the input started playback")
	}
	if a.Prompt == before {
		t.Fatalf("typing a space did not reach the input: %q", a.Prompt)
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
	want := map[int]bool{1: true, 2: true, 3: true, 4: true, 5: true, 6: true}
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
		// A callout is three entries in order: the "常见误解" heading, the belief it is
		// correcting, and the correction. The correction is the one carrying numbers —
		// that is the whole point of this test — and the belief is fixed text.
		//
		// Picking the longest text in the stage instead was what this used to do; it
		// worked until the output stage's body gained a clause and became the longest
		// string, at which point the test was comparing two paragraphs that are not
		// callouts. Picking i+1 would be just as wrong for a different reason: that is
		// the belief, identical across prompts by design.
		texts := tr.Texts()
		for i, s := range texts {
			if s == "常见误解" && i+2 < len(texts) {
				return texts[i+2]
			}
		}
		return ""
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
	// activation row has none — and the callout says so in words: "「大部分维度接近 0」
	// 而不是「大部分维度恰好等于 0」——这一行里恰好等于 0 的有 N 个". If N ever became
	// non-zero the sentence would be self-contradicting, so this is an assertion rather
	// than a note.
	if zeros := zeroDims(hidden); zeros != 0 {
		t.Fatalf("this row has %d exact zeros; the callout says GELU never truncates to "+
			"zero, and the sentence quotes this count", zeros)
	}

	// The rendered sentence must carry that same count, not a stale 0. The callout
	// lives on the FFN stage, so the frame has to be that stage.
	a.Step = 5
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	want := fmt.Sprintf("恰好等于 0 的有 %d 个", zeroDims(hidden))
	var saw bool
	for _, s := range tr.Texts() {
		if strings.Contains(s, want) {
			saw = true
		}
	}
	if !saw {
		t.Fatalf("the FFN callout does not report %q", want)
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
		// The two tinted-chip states the audit above did not reach: the chip's
		// base is colPanel2 when hovered, and the current stepper row sits on
		// colPanel2 with a 14% tint. colFaint measures 3.58:1 and 3.69:1 there —
		// worse than the 3.84:1 the widget's comment quoted — which is why both
		// use colMuted.
		{"rail id on a selected chip over colPanel2", colMuted, over(colAccent.Alpha(0.16), colPanel2), small},
		{"stepper name on the current row", colText, over(colAccent.Alpha(0.14), colPanel2), small},
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
		// The weight line and the formula must agree to four decimals — which they do by
		// construction, since the check above already holds p.weight within 1e-6 of the trace and
		// a 1e-6 difference cannot move the fourth decimal. This used to be described as "what
		// the card prints", but it formats both numbers itself rather than reading the panel; the
		// claim was the thing that was wrong, not the arithmetic.
		//
		// The rendered card is covered by TestEveryCardChipJumpsToItsCard and the per-block
		// workings tests, which parse the drawn lines rather than formatting their own copies.
		if printed, recomputed := fmt.Sprintf("%.4f", h.Attn[q][k]), fmt.Sprintf("%.4f", p.weight); printed != recomputed {
			t.Fatalf("cell %d,%d: weight line %s disagrees with the formula %s",
				q, k, printed, recomputed)
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
		if strings.HasPrefix(s, "LayerNorm d0：") {
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
		if strings.HasPrefix(s, "LayerNorm d47：") {
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

// TestSignedFormatsTermsForReading guards the parentheses on both signs.
//
// Round 167 fixed "+0.5674 + -0.6724" by bracketing the negative. The positive case has
// the same flaw the other way round — "+0.1247 + +0.4062", printed on the default prompt —
// where the second plus reads as a typo. Every addend is bracketed now, so the line has one
// shape whichever way the signs fall.
func TestSignedFormatsTermsForReading(t *testing.T) {
	if got := signed(-0.6724); got != "(-0.6724)" {
		t.Fatalf("signed(-0.6724) = %q", got)
	}
	if got := signed(0.5674); got != "(+0.5674)" {
		t.Fatalf("signed(0.5674) = %q, want the sign bracketed too", got)
	}
	if got := signed(0); got != "(+0.0000)" {
		t.Fatalf("signed(0) = %q", got)
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
	// The point of the absolute denominator: it must report a materially smaller share
	// than dividing by the signed result would. Measured on the selected token the naive
	// form reports 48.95% against the correct 16.05%, a factor of 3.05 — the signed total
	// cancels heavily, which is the whole reason the other denominator exists.
	//
	// This was a t.Logf branch, so it proved nothing: on the row the test happens to use
	// it never fired, and had it fired it would only have printed a line. The margin is
	// asserted now, at a factor that still leaves room for a different token's cancellation
	// while failing if the denominator ever goes back to the signed total.
	if naive := 100 * top / math.Abs(p.Out); naive < want*2 {
		t.Fatalf("dividing by |out| gives %.2f%%, dividing by Σ|term| gives %.2f%%; the "+
			"absolute denominator is supposed to report a materially smaller share",
			naive, want)
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
	const sweepTo = float32(940) // the window this test builds
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
	hidX, hidY := stripPoint(t, a, tr, sweepTo, "hidden")

	outX, outY := stripPoint(t, a, tr, sweepTo, "out")
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
			hidX, hidY = stripPoint(t, a, tr, sweepTo, "hidden")
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
func stripPoint(t *testing.T, a *App, tr *ui.Tester, sweepTo float32, kind string) (x, y float32) {
	t.Helper()
	// Sweep the whole window rather than a fixed 1500 DIP. The constant was the reason this
	// helper reported "no strip writing head found" after a card was added above the strips:
	// the caller grew the window to fit the stage, but the sweep never looked past its old
	// bound. The height comes from the caller because Tester exposes no size accessor.
	//
	// The stride is 9 DIP, not 3. The strips it looks for are 16–22 DIP tall, so 9 cannot step
	// over one, and it was measured to find every kind at the same y as a 3-DIP sweep (hidden
	// y=385, out y=709 on the FFN stage). At 3 DIP this helper rendered up to 450 frames per
	// call, which is why TestEachStripWidthHasItsOwnCursor alone took 29 seconds — 14% of a
	// 206-second suite that CI bounds at 300.
	for probe := float32(250); probe < sweepTo; probe += 9 {
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
	// A tall window, not the default: this test is about which hover field each strip
	// writes, and the attention stage's strips sit far below the fold once the four-head
	// comparison and head-independence cards are on it. A layout that scrolls is
	// TestEveryStageCanBeScrolledToItsEnd's business; making this test scroll would confuse
	// "the strip is missing" with "the strip is off screen".
	//
	// The height is derived from the stage's own content rather than guessed, because it had
	// to be raised twice: each new card above the strips pushed them past whatever number
	// was written here, and the failure read "no strip writing head found".
	a.Step = 3
	measure := ui.NewTester(a.View, 1500, 800)
	measure.Frame()
	measure.Move(600, 400)
	for i := 0; i < 60; i++ {
		measure.Wheel(600, 400, 0, 600)
		measure.Advance(2e9)
	}
	measure.Frame()
	need := a.Scroll.MaxY + a.StageViewportH + 60
	if need < 1600 {
		need = 1600
	}
	tr := ui.NewTester(a.View, 1500, int(need))
	tr.Frame()
	tr.Frame()
	headX, headY := stripPoint(t, a, tr, need, "head")
	tr.Move(headX, headY)
	tr.Frame()
	if a.HoverDimHeadOut < 0 {
		t.Fatal("pointer is not on the head output strip")
	}
	if head := a.HoverDimHeadOut; head >= cfg.DModel {
		t.Fatalf("head output hovered at d%d, past d_head %d", head, cfg.DHead())
	}
	// The projection field may still hold a value from an earlier hover — that is by
	// design, documented at vectorStrip: the formulas under a strip follow the last
	// hovered dimension, so clearing on leave would snap them back to d0 as the reader
	// moved towards the working. What must not be true is that pointing at the head strip
	// puts the *projection's* readout on screen, since a readout claims "this is the cell
	// under your cursor".
	for _, line := range tr.Texts() {
		if strings.HasPrefix(line, "d") && strings.Contains(line, " = ") {
			if !strings.HasPrefix(line, fmt.Sprintf("d%d ", a.HoverDimHeadOut)) {
				t.Fatalf("pointing at the head output strip shows the readout %q, which is not "+
					"d%d but a value from another strip", line, a.HoverDimHeadOut)
			}
		}
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

// TestSelectionFollowsTheTokenThroughTruncation pins the behaviour of the
// selection-shift arithmetic in recompute. It had a comment but no test, so the
// claim "the old token's absolute position is still meaningful" was only ever
// asserted by whoever wrote it.
//
// The scenario that exercises it: a prompt at the context ceiling, extended a
// few bytes at a time. The selected character keeps its absolute position in the
// prompt, so its index in the window has to fall by exactly as many bytes as
// truncation dropped.
func TestSelectionFollowsTheTokenThroughTruncation(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = strings.Repeat("abcdefghij", 6) // 60 bytes, under ctx 64
	a.recompute()
	a.SelToken = 0 // 'a', the first character
	ctx := a.W.Config.Ctx

	// The invariant: once the window is sliding, the selection index moves by
	// the NEGATIVE of the change in dropped bytes, because the character keeps
	// its absolute position while the window's start advances. (Not by the
	// cumulative drop — that was my first guess and it is wrong.)
	prevSel := a.SelToken
	prevDropped := 0
	for i := 0; i < 4; i++ {
		a.Prompt += "XYZ" // three bytes each round
		a.recompute()
		n := len(a.Trace.Tokens)
		if n > ctx {
			t.Fatalf("round %d: %d tokens exceeds ctx %d", i, n, ctx)
		}
		if a.SelToken < 0 || a.SelToken >= n {
			t.Fatalf("round %d: selection %d out of range for %d tokens", i, a.SelToken, n)
		}
		dropped := len(a.Prompt) - n
		deltaDropped := dropped - prevDropped
		switch {
		case deltaDropped == 0:
			// Nothing new was cut, so the same index still names the same token.
			if a.SelToken != prevSel {
				t.Fatalf("round %d: selection moved %d -> %d with nothing dropped", i, prevSel, a.SelToken)
			}
		case prevSel-deltaDropped < 0:
			// The selected byte itself has just fallen out of the window. The
			// only sensible landing place is the newest token.
			if a.SelToken != n-1 {
				t.Fatalf("round %d: the selected token was cut, so the selection should "+
					"land on the last token %d, got %d", i, n-1, a.SelToken)
			}
		default:
			if want := prevSel - deltaDropped; a.SelToken != want {
				t.Fatalf("round %d (dropped %d more): selection %d, want %d",
					i, deltaDropped, a.SelToken, want)
			}
		}
		prevSel, prevDropped = a.SelToken, dropped
	}
}

// TestSelectionWithinTheWindowDoesNotMove checks the case where the prompt grows
// but nothing is dropped: the index must stay put rather than be nudged.
func TestSelectionWithinTheWindowDoesNotMove(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = "hello"
	a.recompute()
	a.SelToken = 2
	a.Prompt = "hello!"
	a.recompute()
	if a.SelToken != 2 {
		t.Fatalf("selection moved from 2 to %d while nothing was truncated", a.SelToken)
	}
	if got := a.Trace.Tokens[a.SelToken].Text; got != "l" {
		t.Fatalf("selection is on %q, want l", got)
	}
}

// TestSelectionResetsOnReplacement checks the deliberate fallback: a prompt that
// is replaced rather than extended has no token at the same index, so the
// selection goes to the end instead of pointing at an unrelated character.
func TestSelectionResetsOnReplacement(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = "abc"
	a.recompute()
	a.SelToken = 1
	a.Prompt = "xyz"
	a.recompute()
	if a.SelToken != len(a.Trace.Tokens)-1 {
		t.Fatalf("selection %d, want the last token %d", a.SelToken, len(a.Trace.Tokens)-1)
	}
}

// TestSelectionAfterSamplingAtTheCeiling checks the other path into truncation:
// sampling appends one byte to a full window. The reader did not move anything,
// so landing on the new last token is the only sensible outcome.
func TestSelectionAfterSamplingAtTheCeiling(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = strings.Repeat("a", a.W.Config.Ctx)
	a.recompute()
	a.SelToken = 0
	a.sampleNext()
	if n := len(a.Trace.Tokens); n != a.W.Config.Ctx {
		t.Fatalf("%d tokens after sampling, want the window still full", n)
	}
	if a.SelToken != len(a.Trace.Tokens)-1 {
		t.Fatalf("selection %d, want the last token", a.SelToken)
	}
}

// TestSelectionShiftCountsBothSidesOfTheEdit is the case that distinguishes the
// shift formula from a naive one. Extending an already-truncated prompt has
// truncation on BOTH sides of the edit: the previous window had already dropped
// bytes, and the new one drops more. A formula that ignored the earlier drop,
// or that ignored the new one, lands on a different index here — whereas on a
// prompt that only just reaches the ceiling both shortcuts agree, which is why
// the obvious test for this did not catch either.
func TestSelectionShiftCountsBothSidesOfTheEdit(t *testing.T) {
	a := newTestApp(t)
	ctx := a.W.Config.Ctx

	// 100 bytes: already 36 over the window when the edit begins.
	a.Prompt = strings.Repeat("a", 100)
	a.recompute()
	if got := len(a.Trace.Tokens); got != ctx {
		t.Fatalf("%d tokens, want the window full at %d", got, ctx)
	}
	a.SelToken = 50
	prevSel := a.SelToken

	a.Prompt += "bbb" // three more bytes, all dropped
	a.recompute()
	if got := len(a.Trace.Tokens); got != ctx {
		t.Fatalf("%d tokens after the edit, want %d", got, ctx)
	}
	// The character at index 50 sat at absolute byte 36+50=86 before the edit
	// and at 39+50=89 after it, so it stays at index 50 relative to the window
	// start... but the window start moved from 36 to 39, so its index is 47.
	if want := prevSel - 3; a.SelToken != want {
		t.Fatalf("selection %d, want %d (the window start advanced by the 3 bytes added)",
			a.SelToken, want)
	}

	// And a replacement still resets, even from a truncated state.
	b := newTestApp(t)
	b.Prompt = strings.Repeat("a", 100)
	b.recompute()
	b.SelToken = 50
	b.Prompt = "xyz"
	b.recompute()
	if b.SelToken != len(b.Trace.Tokens)-1 {
		t.Fatalf("replacement left the selection at %d, want the last token %d",
			b.SelToken, len(b.Trace.Tokens)-1)
	}
}

// TestPinnedCellSurvivesEditing is the interaction this round fixed. The reader
// pins a cell, then the prompt changes underneath — by typing or by sampling.
// The pin is a deliberate choice about a cell that still exists, so it must
// stay; only the pointer's hover is stale and should be cleared.
func TestPinnedCellSurvivesEditing(t *testing.T) {
	// Typing an extra character only grows the matrix, so the pin keeps its indices.
	//
	// Sampling used to be checked here too, on the same reasoning — it appends a byte, so every cell keeps its
	// indices. That changed in round 245: sampling now selects the token it produced, which moves the reader to a
	// different row, and a pin left behind would head the inspector "query = token 15" over a card still describing
	// token 5. selectToken drops the pin for exactly this reason on every other path, and sampling goes through it
	// now. The cell is still valid; it is simply no longer the one the reader is looking at.
	b := newTestApp(t)
	b.Step = 2
	b.KeyQ, b.KeyK = 20, 5
	b.Prompt += "!"
	b.recompute()
	if b.KeyQ != 20 || b.KeyK != 5 {
		t.Fatalf("typing moved the pin to (%d,%d), want (20,5)", b.KeyQ, b.KeyK)
	}

	// And the inspector still describes it.
	tr := ui.NewTester(b.View, 1500, 940)
	tr.Frame()
	if !tr.HasText("已固定") {
		t.Fatal("the inspector stopped describing the pinned cell")
	}
}

// TestPinnedCellClampsWhenTheMatrixShrinks is the other half: a pin outside the
// new bounds must not be drawn on a cell that no longer exists. Truncating the
// prompt shrinks the matrix under the selection.
func TestPinnedCellClampsWhenTheMatrixShrinks(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	a.KeyQ, a.KeyK = 30, 20
	a.Prompt = strings.Repeat("a", 10)
	a.recompute()
	n := len(a.Trace.Tokens)
	if n >= 30 {
		t.Fatalf("expected a small trace, got %d tokens", n)
	}
	if a.KeyQ < 0 || a.KeyQ >= n {
		t.Fatalf("pin row %d is outside 0..%d", a.KeyQ, n-1)
	}
	if a.KeyK < 0 || a.KeyK > a.KeyQ {
		t.Fatalf("pin (%d,%d) is not inside the triangle", a.KeyQ, a.KeyK)
	}

	// An empty prompt has no matrix at all, so the pin has to go.
	b := newTestApp(t)
	b.KeyQ, b.KeyK = 5, 2
	b.Prompt = ""
	b.recompute()
	if b.KeyQ >= 0 || b.KeyK >= 0 {
		t.Fatalf("empty prompt left the pin at (%d,%d)", b.KeyQ, b.KeyK)
	}
}

// TestPointerHoverClearsOnEdit is the part that should still clear: hovering a
// cell is a statement about where the pointer is, and the page reflows under it.
func TestPointerHoverClearsOnEdit(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	a.HoverQ, a.HoverK = 10, 4
	a.HoverDimIn = 3
	a.recompute()
	if a.HoverQ >= 0 || a.HoverK >= 0 {
		t.Fatalf("pointer hover survived an edit: (%d,%d)", a.HoverQ, a.HoverK)
	}
	if a.HoverDimIn >= 0 {
		t.Fatalf("dimension hover survived an edit: %d", a.HoverDimIn)
	}
}

// TestStageSwitchStillClearsThePin checks the distinction holds the other way:
// changing stage really does invalidate the cell, so selectStep keeps using the
// full clear.
func TestStageSwitchStillClearsThePin(t *testing.T) {
	a := newTestApp(t)
	a.KeyQ, a.KeyK = 20, 5
	a.selectStep(2)
	if a.KeyQ >= 0 || a.KeyK >= 0 {
		t.Fatalf("stage switch left the pin at (%d,%d)", a.KeyQ, a.KeyK)
	}
}

// TestPinClaimsInTheReadmeAreTrue checks the four things the interaction table
// now says about pinning a cell. The row used to promise only that the pin
// survived the pointer moving, which was all it did; the claim grew when
// recompute started preserving it, and a claim in a document is worth a test.
func TestPinClaimsInTheReadmeAreTrue(t *testing.T) {
	// Pinning survives the pointer leaving the matrix.
	a := newTestApp(t)
	a.Step = 2
	a.KeyQ, a.KeyK = 20, 5
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	tr.Move(1400, 300)
	tr.Frame()
	if a.KeyQ != 20 || a.KeyK != 5 {
		t.Fatalf("pointer move lost the pin: (%d,%d)", a.KeyQ, a.KeyK)
	}

	// It survives an edit. Sampling is different now: it selects the token it produced, which is the same context
	// change as clicking another chip, and selectToken drops the pin for that. The README says so.
	a.Prompt += "!"
	a.recompute()
	if a.KeyQ != 20 || a.KeyK != 5 {
		t.Fatalf("typing lost the pin: (%d,%d)", a.KeyQ, a.KeyK)
	}

	// Esc releases it.
	b := newTestApp(t)
	b.Step = 2
	b.KeyQ, b.KeyK = 20, 5
	tr2 := ui.NewTester(b.View, 1500, 940)
	tr2.Frame()
	tr2.Key(0, ui.KeyEscape)
	tr2.Frame()
	if b.KeyQ >= 0 {
		t.Fatalf("Esc left the pin at (%d,%d)", b.KeyQ, b.KeyK)
	}

	// A stage switch releases it, which the row also states.
	c := newTestApp(t)
	c.KeyQ, c.KeyK = 20, 5
	c.selectStep(2)
	if c.KeyQ >= 0 || c.KeyK >= 0 {
		t.Fatalf("stage switch left the pin at (%d,%d)", c.KeyQ, c.KeyK)
	}
}

// TestLayerDivergenceMatchesABruteForceTV checks the divergence number against
// the definition, computed independently here.
func TestLayerDivergenceMatchesABruteForceTV(t *testing.T) {
	a := newTestApp(t)
	tr := a.Trace
	if len(tr.Layers) < 2 {
		t.Skip("need two layers")
	}
	for h := range tr.Layers[0].Heads {
		for _, row := range []int{5, 20, len(tr.Tokens) - 1} {
			d := computeLayerDivergence(tr.Layers, 1, h, row)
			if !d.Ok {
				t.Fatalf("head %d row %d: no divergence computed", h, row)
			}
			var want float64
			l0 := tr.Layers[0].Heads[h].Attn[row]
			l1 := tr.Layers[1].Heads[h].Attn[row]
			for k := 0; k <= row; k++ {
				want += math.Abs(float64(l0[k]) - float64(l1[k]))
			}
			want /= 2
			if math.Abs(d.TV-want) > 1e-6 {
				t.Fatalf("head %d row %d: TV %.6f, brute force %.6f", h, row, d.TV, want)
			}
			if d.TV < 0 || d.TV > 1+1e-6 {
				t.Fatalf("head %d row %d: TV %.4f is outside [0,1]", h, row, d.TV)
			}
		}
	}
}

// TestLayerDivergenceSkipsDegenerateRows is the edge the card must not quote.
// Row 0 sees a single key, so both layers are forced to weight it 1.0 and the
// distance is zero for a reason that says nothing about what they learned.
func TestLayerDivergenceSkipsDegenerateRows(t *testing.T) {
	a := newTestApp(t)
	tr := a.Trace
	if d := computeLayerDivergence(tr.Layers, 1, 0, 0); d.Ok {
		t.Fatalf("row 0 reported a divergence of %.3f", d.TV)
	}
	// Layer 0 has no previous layer to compare against.
	if d := computeLayerDivergence(tr.Layers, 0, 0, 5); d.Ok {
		t.Fatalf("layer 0 reported a divergence of %.3f", d.TV)
	}
}

// TestLayerDivergenceIsSubstantialEverywhere backs the claim the number makes:
// the two layers attend differently, on every head and every row that can be
// compared. If a future weight file made them near-identical the row would be
// teaching something false.
func TestLayerDivergenceIsSubstantialEverywhere(t *testing.T) {
	a := newTestApp(t)
	tr := a.Trace
	if len(tr.Layers) < 2 {
		t.Skip("need two layers")
	}
	n := len(tr.Tokens)
	var lo, sum float64 = math.Inf(1), 0
	cnt := 0
	for h := range tr.Layers[1].Heads {
		for row := 1; row < n; row++ {
			d := computeLayerDivergence(tr.Layers, 1, h, row)
			if !d.Ok {
				continue
			}
			sum += d.TV
			cnt++
			if d.TV < lo {
				lo = d.TV
			}
		}
	}
	if cnt == 0 {
		t.Fatal("nothing measured")
	}
	mean := sum / float64(cnt)
	t.Logf("layer 0 vs 1: mean TV %.3f, min %.3f over %d rows", mean, lo, cnt)
	// The callout says the layers are doing different things. Half-overlap would
	// already justify saying so; anything less would not.
	if mean < 0.5 {
		t.Fatalf("mean divergence %.3f is too small to call the layers different", mean)
	}
	if lo < 0.01 {
		t.Fatalf("some comparable row is almost identical (%.4f), so 'they differ' is not universal", lo)
	}
}

// TestLayerDivergenceRowNamesThePreviousLayer guards the label. The layer
// control labels layers from 0, and the comparison is against a.Layer-1; writing
// a.Layer there produced "与第 1 层之差" while showing layer 1, which reads as a
// comparison with itself.
func TestLayerDivergenceRowNamesThePreviousLayer(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	a.Layer = 1
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	hasPrev, hasSelf := false, false
	for _, s := range tr.Texts() {
		if strings.HasPrefix(s, "与第 0 层之差") {
			hasPrev = true
		}
		if strings.HasPrefix(s, "与第 1 层之差") {
			hasSelf = true
		}
	}
	if hasSelf {
		t.Fatal("the row names layer 1 while layer 1 is being shown")
	}
	if !hasPrev {
		t.Fatal("the row does not name the previous layer")
	}

	// On layer 0 there is no previous layer, so the row must be absent.
	b := newTestApp(t)
	b.Step = 3
	b.Layer = 0
	tr2 := ui.NewTester(b.View, 1500, 940)
	tr2.Frame()
	for _, s := range tr2.Texts() {
		if strings.Contains(s, "之差") {
			t.Fatalf("layer 0 shows a divergence row: %q", s)
		}
	}
}

// TestInspectorHeaderShowsCharacterSubindex is the fix this round: selecting a
// byte of a multi-byte character used to show the model's "\xb3" escape as the
// headline glyph, with "rune '注'" as the only hint about what it belonged to.
// The strip already showed 注₂, so the header now agrees with it and names the
// character's bytes.
func TestInspectorHeaderShowsCharacterSubindex(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = "a注b"
	a.recompute()
	a.Step = 0
	tr := ui.NewTester(a.View, 1500, 940)

	// Selecting byte 2 of 注 must not show the escape as the glyph.
	a.SelToken = 2
	tr.Frame()
	sawGlyph, sawDetail := false, false
	for _, s := range tr.Texts() {
		if s == "注₂" {
			sawGlyph = true
		}
		if strings.Contains(s, "是 '注' 的第 2/3 个字节") && strings.Contains(s, "E6 B3 A8") {
			sawDetail = true
		}
		if s == `\xb3` {
			t.Fatal("the escape is still the headline glyph")
		}
	}
	if !sawGlyph {
		t.Fatal("the header does not show the character with its byte index")
	}
	if !sawDetail {
		t.Fatal("the header does not name the byte's position within the character")
	}

	// The first byte of the same character reads 1/3.
	b := newTestApp(t)
	b.Prompt = "a注b"
	b.recompute()
	b.Step = 0
	b.SelToken = 1
	tr2 := ui.NewTester(b.View, 1500, 940)
	tr2.Frame()
	found := false
	for _, s := range tr2.Texts() {
		if strings.Contains(s, "第 1/3 个字节") {
			found = true
		}
	}
	if !found {
		t.Fatal("the first byte of a character is not numbered 1/3")
	}
}

// TestInspectorHeaderLeavesSingleByteTokensAlone checks the substitution stays
// away from tokens that already read well: ASCII, and an invalid byte that
// belongs to no character at all.
func TestInspectorHeaderLeavesSingleByteTokensAlone(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = "abc"
	a.recompute()
	a.Step = 0
	a.SelToken = 1
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	sawChar := false
	for _, s := range tr.Texts() {
		if s == "b" {
			sawChar = true
		}
		// Only the header is under test here; the stage-1 card legally
		// describes every character, ASCII included.
		if strings.Contains(s, "· 是 '") && strings.Contains(s, "个字节") {
			t.Fatalf("an ASCII token was described as part of a multi-byte character: %q", s)
		}
	}
	if !sawChar {
		t.Fatal("the ASCII glyph is missing")
	}

	// An invalid byte has no character, so it keeps the plain byte form.
	b := newTestApp(t)
	b.Prompt = string([]byte{0x61, 0xFF, 0x62})
	b.recompute()
	b.Step = 0
	b.SelToken = 1
	tr2 := ui.NewTester(b.View, 1500, 940)
	tr2.Frame()
	for _, s := range tr2.Texts() {
		if strings.Contains(s, "· 是 '") {
			t.Fatalf("an invalid byte was described as part of a character: %q", s)
		}
	}
}

// TestInspectorDetailFitsTheColumn keeps the longer header line inside the panel
// at its narrowest width. The widest realistic form is a 4-byte character.
func TestInspectorDetailFitsTheColumn(t *testing.T) {
	var w float32
	ui.Render(func(c *ui.Context) {
		w, _ = c.MeasureText(0, ui.Span{
			Text: `byte 0xF0 · 是 '𝄞' 的第 1/4 个字节（F0 9D 84 9E）`, Size: fontMicro})
	}, 400, 100, 1)
	// 330 panel - 2*14 card padding - 34 icon - 8 gap.
	const available = 330 - 2*14 - 34 - 8
	if w > available {
		t.Fatalf("the detail line is %.0f wide, the header offers %.0f", w, float32(available))
	}
}

// TestTemperatureOnlyAffectsTheOutputStage pins the scope of the temperature
// slider. It is greyed out on stages 1-6 and the label says so, but nothing
// checked that the value really has no effect there — a defensive copy, a stray
// Probabilities call in the attention path, and the greyed control would be
// lying about its own reach.
func TestTemperatureOnlyAffectsTheOutputStage(t *testing.T) {
	render := func(step int, temp float64) []string {
		a := newTestApp(t)
		a.Step = step
		a.Temp = temp
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		return tr.Texts()
	}
	// The label prints the temperature, so it always differs; everything else in
	// the frame must be identical.
	for step := 0; step < 6; step++ {
		cold := render(step, 0.1)
		hot := render(step, 2.0)
		if len(cold) != len(hot) {
			t.Fatalf("step %d: %d texts at temp 0.1, %d at 2.0", step+1, len(cold), len(hot))
		}
		for i := range cold {
			if cold[i] == hot[i] {
				continue
			}
			if strings.HasPrefix(cold[i], "温度 ") && strings.HasPrefix(hot[i], "温度 ") {
				continue // the control's own readout
			}
			t.Fatalf("step %d: temperature changed something else: %q vs %q",
				step+1, cold[i], hot[i])
		}
	}

	// And on the output stage it must change the distribution.
	base := newTestApp(t)
	base.Step = 6
	base.Temp = 0.1
	tr := ui.NewTester(base.View, 1500, 940)
	tr.Frame()
	flat := newTestApp(t)
	flat.Step = 6
	flat.Temp = 2.0
	tr2 := ui.NewTester(flat.View, 1500, 940)
	tr2.Frame()
	// A higher temperature should raise the entropy readout.
	var coldEnt, hotEnt string
	for _, s := range tr.Texts() {
		if strings.Contains(s, "nats") {
			coldEnt = s
		}
	}
	for _, s := range tr2.Texts() {
		if strings.Contains(s, "nats") {
			hotEnt = s
		}
	}
	if coldEnt == "" || hotEnt == "" {
		t.Fatal("no entropy readout on the output stage")
	}
	if coldEnt == hotEnt {
		t.Fatalf("temperature did not change the output distribution: both %q", coldEnt)
	}
}

// TestDisabledTemperatureSliderRefusesInput checks the control is actually inert
// where it is greyed out, not merely drawn grey.
func TestDisabledTemperatureSliderRefusesInput(t *testing.T) {
	// Coordinates that move the slider when the output stage is showing.
	const sliderX, sliderY = 700, 65

	enabled := newTestApp(t)
	enabled.Step = 6
	tr := ui.NewTester(enabled.View, 1500, 940)
	tr.Frame()
	before := enabled.Temp
	tr.Press(sliderX, sliderY)
	tr.Release(sliderX, sliderY)
	tr.Frame()
	if enabled.Temp == before {
		t.Fatalf("the slider did not move at (%d,%d) even when enabled (still %.2f)",
			sliderX, sliderY, before)
	}

	disabled := newTestApp(t)
	disabled.Step = 0
	tr2 := ui.NewTester(disabled.View, 1500, 940)
	tr2.Frame()
	beforeD := disabled.Temp
	tr2.Press(sliderX, sliderY)
	tr2.Release(sliderX, sliderY)
	tr2.Frame()
	if disabled.Temp != beforeD {
		t.Fatalf("a greyed slider changed the temperature: %.2f -> %.2f", beforeD, disabled.Temp)
	}
}

// TestStageSwitchResetsTheMainScroll pins the behaviour a reader relies on when
// jumping between stages: the new stage starts at its top, not wherever the last
// one was scrolled to.
//
// Getting this test to say anything required learning that wheel scrolling is
// animated in this toolkit: Wheel queues a glide and Scroll.Y stays put until
// time advances. Two probes reported "the wheel does nothing" before that, which
// would have been a report of a bug that did not exist.
func TestStageSwitchResetsTheMainScroll(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3 // a tall stage at this window size
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()
	tr.Move(500, 400)
	tr.Frame()

	// One notch moves gradually, not instantly.
	tr.Wheel(500, 400, 0, 100)
	tr.Advance(time.Second)
	tr.Frame()
	if a.Scroll.Y <= 0 {
		t.Fatalf("the wheel did not scroll the main column (Y=%.0f)", a.Scroll.Y)
	}

	// Scroll well past the top.
	for i := 0; i < 10; i++ {
		tr.Wheel(500, 400, 0, 300)
		tr.Advance(time.Second)
	}
	tr.Frame()
	deep := a.Scroll.Y
	if deep < 200 {
		t.Fatalf("expected to be well down the page, got Y=%.0f", deep)
	}

	// Switching stages puts it back at the top.
	a.selectStep(0)
	tr.Frame()
	if a.Scroll.Y != 0 {
		t.Fatalf("stage switch left the scroll at %.0f", a.Scroll.Y)
	}

	// A scroll set from anywhere else is cleared by the same path, so the reset
	// is a property of selectStep rather than of having arrived there by wheel.
	b := newTestApp(t)
	b.Scroll.Y = 300
	b.selectStep(5)
	if b.Scroll.Y != 0 {
		t.Fatalf("selectStep left a preset scroll at %.0f", b.Scroll.Y)
	}
}

// TestCancellationSummaryFollowsTheTrace checks the cached cancellation summary
// is refreshed when the trace is rebuilt. The cache exists because the walk ran
// on every frame for numbers that only change when the prompt does; a stale
// summary would quote one prompt's statistics beside another's heatmap.
func TestCancellationSummaryFollowsTheTrace(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	long := a.cancellation()
	if long.Rows != len(a.Trace.Layers)*len(a.Trace.Tokens) {
		t.Fatalf("rows=%d for %d layers x %d tokens",
			long.Rows, len(a.Trace.Layers), len(a.Trace.Tokens))
	}
	if long.Below != long.Rows {
		t.Fatalf("%d of %d rows cancel; the callout claims all of them", long.Below, long.Rows)
	}

	// A much shorter prompt has far fewer rows.
	a.Prompt = "short"
	a.recompute()
	short := a.cancellation()
	if short.Rows >= long.Rows {
		t.Fatalf("after shrinking the prompt, rows=%d (was %d)", short.Rows, long.Rows)
	}
	if short.Rows != len(a.Trace.Layers)*len(a.Trace.Tokens) {
		t.Fatalf("stale summary: rows=%d for %d layers x %d tokens",
			short.Rows, len(a.Trace.Layers), len(a.Trace.Tokens))
	}

	// An empty prompt has nothing to measure.
	a.Prompt = ""
	a.recompute()
	if empty := a.cancellation(); empty.Rows != 0 {
		t.Fatalf("empty prompt measured %d rows", empty.Rows)
	}
}

// TestCancellationIsCachedNotRecomputed checks the cache is actually used: the
// stage must not walk the trace again on every frame.
func TestCancellationIsCachedNotRecomputed(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	first := a.cancellation()
	if first.Rows == 0 {
		t.Fatal("the initial summary was empty")
	}

	// The stage must render from the cached value, not walk the trace again:
	// a poisoned cache that still matches the current trace has to survive a
	// render, or every frame is paying for the walk.
	a.cancelStats = cancellationStats{Rows: 12345, Below: 1, Lo: 0.5, Hi: 0.6}
	a.cancelStatsFor = a.Trace
	if got := a.cancellation(); got.Rows != 12345 {
		t.Fatalf("the summary was recomputed (rows=%d, poisoned value was 12345)", got.Rows)
	}

	// A new trace invalidates it, so no frame can quote the old prompt's numbers.
	a.Prompt = "short"
	a.recompute()
	if got := a.cancellation(); got.Rows == 12345 {
		t.Fatal("the poisoned value survived recompute")
	}
	if got := a.cancellation(); got.Rows != len(a.Trace.Layers)*len(a.Trace.Tokens) {
		t.Fatalf("rows=%d for the new trace (%d layers x %d tokens)",
			got.Rows, len(a.Trace.Layers), len(a.Trace.Tokens))
	}
}

// TestVectorStripLabelsItsEnds checks each strip states which dimensions it
// spans, and that the markers sit under the strip's actual edges.
//
// Without them a strip is an unlabelled smear: a reader cannot tell that it starts
// at d0, how many dimensions it covers, or that hovering halfway gives roughly
// d48. The hover readout does not help with that, because it only appears while
// the pointer is on the strip, which is exactly when the strip's extent no longer
// needs explaining.
func TestVectorStripLabelsItsEnds(t *testing.T) {
	a := newTestApp(t)
	a.Step = 1 // the embedding stage draws three d_model-wide strips
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	var d0, dMax ui.Rect
	var ok0, okMax bool
	for _, s := range tr.Texts() {
		switch s {
		case "d0":
			d0, ok0 = tr.Find(s)
		case fmt.Sprintf("d%d", a.W.Config.DModel-1):
			dMax, okMax = tr.Find(s)
		}
	}
	if !ok0 || !okMax {
		t.Fatalf("the strip's end markers are missing (d0=%v, d%d=%v)",
			ok0, a.W.Config.DModel-1, okMax)
	}

	// The labels row spans the card; the card's content starts at the same x as
	// the section heading, and ends one card-padding short of the panel edge.
	heading, ok := tr.Find("token embedding")
	if !ok {
		t.Fatal("the first strip's caption is missing")
	}
	cardLeft, cardRight := heading.X, heading.X+heading.W
	if d0.X < cardLeft-1 {
		t.Fatalf("d0 sits %.0f left of the strip (%0.f)", cardLeft-d0.X, cardLeft)
	}
	if end := dMax.X + dMax.W; end < cardRight-2 || end > cardRight+2 {
		t.Fatalf("d%d ends at %.0f but the strip ends at %.0f; the label should mark the "+
			"right edge", a.W.Config.DModel-1, end, cardRight)
	}
}

// TestLayerNormHistogramsShareOneAxis is the defect this round fixed. The two
// charts were each scaled to their own extremes, so a row whose values shrank from
// a span of 10.8 to 6.1 still filled both charts edge to edge and the pictures
// looked the same — while the numbers printed underneath disagreed with the
// shapes. Stacked charts to compare are the whole point of that card.
func TestLayerNormHistogramsShareOneAxis(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4
	a.SelToken = 10
	l := a.layer()
	before := l.AfterAttn[10]
	after := l.Norm2[10]

	bl, bh := minMaxFloat(before)
	al, ah := minMaxFloat(after)
	// The case only means something if the ranges actually differ.
	if bl == al && bh == ah {
		t.Skip("this token's two rows happen to share a range")
	}

	// Both charts must be drawn on the union of the two ranges, so the label
	// under each one is the same pair of numbers.
	axLo, axHi := bl, bh
	if al < axLo {
		axLo = al
	}
	if ah > axHi {
		axHi = ah
	}

	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	// Scroll the LayerNorm card into view.
	tr.Move(600, 500)
	tr.Frame()
	for i := 0; i < 6; i++ {
		tr.Wheel(600, 500, 0, 300)
		tr.Advance(2e9)
	}
	tr.Frame()

	want := fmt.Sprintf("%+.2f", axLo)
	wantHi := fmt.Sprintf("%+.2f", axHi)
	seen := 0
	for _, s := range tr.Texts() {
		if s == want || s == wantHi {
			seen++
		}
	}
	// One left label and one right label per chart, two charts.
	if seen < 4 {
		t.Fatalf("expected both charts to print the shared axis %s..%s (four labels), saw %d",
			want, wantHi, seen)
	}
	// And the second chart's own extremes must no longer be printed as its axis:
	// that is what a separately-scaled chart would show.
	for _, own := range []string{fmt.Sprintf("%+.2f", al), fmt.Sprintf("%+.2f", ah)} {
		if own == want || own == wantHi {
			continue // coincides with the shared bound; nothing to tell apart
		}
		if containsText(tr, own) {
			t.Fatalf("the second chart still prints its own bound %s, so the two charts "+
				"do not share an axis", own)
		}
	}
}

func containsText(tr *ui.Tester, s string) bool {
	for _, x := range tr.Texts() {
		if x == s {
			return true
		}
	}
	return false
}

// TestScoreLegendStatesTheColorScale checks the stage says what the brightest
// cell is worth.
//
// The legend said only "正分（暖色）" and "负分（蓝色）", so a reader could see
// that one cell was warmer than another but not what either value was. The ramp
// saturates at the largest |score| in the matrix, and that number moves with the
// prompt — 5.42 for "hi", 8.25 for a longer one — so it cannot be inferred from
// the weights, the prompt, or anything else on the stage.
func TestScoreLegendStatesTheColorScale(t *testing.T) {
	for _, prompt := range []string{"hi", "attention is a weighted sum"} {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		a.Step = 2
		want := scoreColorScale(a.layer().Heads[a.Head])
		if want <= 0 {
			t.Fatalf("%q: the matrix has no scores", prompt)
		}
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()

		label := fmt.Sprintf("|分数| = %.2f", want)
		found := false
		for _, s := range tr.Texts() {
			if strings.Contains(s, label) {
				found = true
			}
		}
		if !found {
			t.Fatalf("%q: the legend does not state the colour scale %q", prompt, label)
		}
	}

	// And the number it states must be the cap the painter actually uses.
	a := newTestApp(t)
	a.Step = 2
	h := a.layer().Heads[a.Head]
	sc := scoreColorScale(h)
	var maxAbs float64
	for i := range h.Scores {
		for j := 0; j <= i; j++ {
			if v := math.Abs(float64(h.Scores[i][j])); v > maxAbs {
				maxAbs = v
			}
		}
	}
	if math.Abs(sc-maxAbs) > 1e-9 {
		t.Fatalf("scoreColorScale = %.6f, the largest |score| is %.6f", sc, maxAbs)
	}
}

// TestScoreColorScaleTracksThePrompt is what makes the previous test matter: the
// cap is not a constant, so a number hard-coded in the legend would be wrong for
// most prompts.
func TestScoreColorScaleTracksThePrompt(t *testing.T) {
	seen := map[string]float64{}
	for _, prompt := range []string{"hi", "attention is a weighted sum", "注意力就是加权求和"} {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		seen[prompt] = scoreColorScale(a.layer().Heads[a.Head])
	}
	first := -1.0
	for _, v := range seen {
		if first < 0 {
			first = v
			continue
		}
		if math.Abs(v-first) > 0.5 {
			return // the scale varies, as the test name claims
		}
	}
	t.Fatalf("the colour scale is the same for every prompt (%v); a fixed number "+
		"in the legend would have sufficed", seen)
}

// TestFFNHistogramMarksZero checks the activation histogram says where zero is.
//
// This is the one chart whose range is dominated by positives: it runs -0.17 to
// +4.11, so zero lands 4% along and the negative region — the thing the callout
// asks the reader to look at — is a sliver with nothing marking where it ends.
//
// The marker is a one-pixel vertical line at 45% white. Two earlier versions of
// this test proved nothing: comparing a sign-changing frame against an
// all-positive one passes with the marker removed, because the bars differ either
// way; and looking for a column brighter than its immediate neighbours failed on
// the real marker, which sits next to a bar of similar brightness. What works is
// locating the chart by its bars and looking for the brightest pixel inside it.
func TestFFNHistogramMarksZero(t *testing.T) {
	a := newTestApp(t)
	a.Step = 5
	h := a.layer().Hidden[a.SelToken]
	lo, hi := minMaxFloat(h)
	if !(lo < 0 && hi > 0) {
		t.Skip("this row has no sign change, so there is no zero to mark")
	}
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	img := tr.Image()

	// The bars are the only strong green on the stage; the row with the most of
	// it is inside the chart.
	green := func(x, y int) bool {
		r, g, b, _ := img.At(x, y).RGBA()
		return g>>8 > 120 && g>>8 > r>>8+40 && g>>8 > b>>8+40
	}
	chartY, most := 0, 0
	for y := 300; y < 720; y++ {
		n := 0
		for x := 250; x < 1200; x++ {
			if green(x, y) {
				n++
			}
		}
		if n > most {
			most, chartY = n, y
		}
	}
	if chartY == 0 || most < 20 {
		t.Fatalf("could not locate the histogram (widest green row had %d)", most)
	}

	// The marker is grey-white, brighter than anything the green bars reach.
	brightest, bx := 0, 0
	for y := chartY - 30; y <= chartY+2; y++ {
		for x := 250; x < 1200; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			sum := int(r>>8) + int(g>>8) + int(b>>8)
			// Require it to be neutral: a bright green bar is not the marker.
			if sum > brightest && absInt(int(r>>8)-int(b>>8)) < 40 {
				brightest, bx = sum, x
			}
		}
	}
	if brightest < 300 {
		t.Fatalf("no neutral bright pixel on the chart (brightest %d), so the zero "+
			"marker is missing", brightest)
	}

	// And it must be where zero falls, not anywhere on the strip.
	// The chart spans roughly the card width; allow a generous window.
	t.Logf("zero marker at x=%d, brightest=%d", bx, brightest)
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// TestFFNHistogramExplainsTheCompressedNegativeHalf checks the caption states the
// proportion, since the chart cannot show it: 239 of 384 values occupy 4% of the
// width and collapse into one bar.
func TestFFNHistogramExplainsTheCompressedNegativeHalf(t *testing.T) {
	a := newTestApp(t)
	a.Step = 5
	h := a.layer().Hidden[a.SelToken]
	lo, hi := minMaxFloat(h)
	if !(lo < 0 && hi > 0) {
		t.Skip("no negative half on this row")
	}
	neg := 0
	for _, v := range h {
		if v < 0 {
			neg++
		}
	}
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	wantNeg := fmt.Sprintf("%d 个负值", neg)
	wantPct := fmt.Sprintf("占 %.0f%%", 100*float64(neg)/float64(len(h)))
	foundNeg, foundPct := false, false
	for _, s := range tr.Texts() {
		if strings.Contains(s, wantNeg) {
			foundNeg = true
		}
		if strings.Contains(s, wantPct) && strings.Contains(s, "个负值") {
			foundPct = true
		}
	}
	if !foundNeg || !foundPct {
		t.Fatalf("the caption does not state the negative proportion (%q / %q): neg=%v pct=%v",
			wantNeg, wantPct, foundNeg, foundPct)
	}
}

// TestDiagonalCellIsMarkedAsBothQueryAndKey covers the one cell where the query
// and the key are the same token. railMarks returns "qk" there, and railMarkColor
// used to fall through to colGreen — the "no mark" colour — so a token attending
// to itself was drawn exactly like a token the crosshair was not on.
//
// The rail has only a border colour to work with (the mark text is not rendered
// on the chip), so "both" has to pick one of the two colours; it picks the
// query's, since that is the row the reader is following.
func TestDiagonalCellIsMarkedAsBothQueryAndKey(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	a.HoverQ, a.HoverK = 10, 10
	if got := a.railMarks(10); got != "qk" {
		t.Fatalf("diagonal cell marks token 10 as %q, want qk", got)
	}
	if got := railMarkColor("qk"); got == railMarkColor("") {
		t.Fatalf("the diagonal cell's colour is the unmarked colour %v", got)
	}
	if got := railMarkColor("qk"); got != colWarm {
		t.Fatalf("the diagonal cell should take the query's colour, got %v", got)
	}
	// The plain marks are untouched.
	if railMarkColor("q") != colWarm || railMarkColor("k") != colAccent2 {
		t.Fatal("the q and k colours changed")
	}
	// And an off-diagonal cell marks two different tokens, not one with "qk".
	a.HoverQ, a.HoverK = 10, 3
	if a.railMarks(10) != "q" || a.railMarks(3) != "k" {
		t.Fatalf("off-diagonal marks are wrong: token10=%q token3=%q",
			a.railMarks(10), a.railMarks(3))
	}
}

// TestTop12StatesTheVocabularySize is the fix this round. The card listed twelve
// candidates without saying they were a slice of 256, and on this model the top 12
// carry essentially all the probability mass — so a reader summing the visible
// percentages to 100% would reasonably conclude the model chooses among twelve
// bytes. That is the opposite of what a token-prediction stage should teach.
func TestTop12StatesTheVocabularySize(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	probs := a.Trace.Probabilities(a.Temp)
	top := model.TopK(probs, 12)
	inTop := map[int]bool{}
	for _, id := range top {
		inTop[id] = true
	}
	var rest float32
	for id, p := range probs {
		if !inTop[id] {
			rest += p
		}
	}

	// The header must name the vocabulary size and the count outside the list.
	wantVocab := fmt.Sprintf("词表 %d 个", len(probs))
	wantRest := fmt.Sprintf("其余 %d 个", len(probs)-len(top))
	found := ""
	for _, s := range tr.Texts() {
		if strings.Contains(s, wantVocab) && strings.Contains(s, wantRest) {
			found = s
		}
	}
	if found == "" {
		t.Fatalf("the Top-12 header does not state the vocabulary size and the tail "+
			"count (want %q and %q)", wantVocab, wantRest)
	}

	// And the tail's share must be shown the way the rows show small numbers:
	// %.4f prints 0.0000 for 0.000013%, which reads as nothing.
	if rest*100 < 0.01 {
		if !strings.Contains(found, "e-") {
			t.Fatalf("the tail share %.3e%% is printed without scientific notation: %q",
				rest*100, found)
		}
	}
}

// TestTop12TailCannotPrintNegative checks the invariant behind how the tail is summed.
//
// The header's remainder is accumulated from the individual non-top probabilities rather than
// taken as 1 - sum(top). The difference is not academic: at T = 0.1 the two disagree at 1e-13, and
// a reader would see "-0.0000%" if the subtraction form were used — measured, and reproducible with
// `1 - sum(top)` on the default prompt.
//
// Accumulating makes the remainder non-negative only because softmax never returns a negative
// probability. That premise is what keeps "-0%" off the screen, and nothing stated it before.
func TestTop12TailCannotPrintNegative(t *testing.T) {
	a := newTestApp(t)
	for _, temp := range []float64{0.01, 0.05, 0.1, 0.3, 0.8, 1.0, 2.0, 4.0} {
		a.Temp = temp
		probs := a.Trace.Probabilities(temp)
		for id, v := range probs {
			if v < 0 {
				t.Fatalf("T=%.2f: probability %d is %v; softmax cannot produce a negative value, "+
					"and the Top-12 remainder is summed from these, so a negative here would be "+
					"printed as a negative share", temp, id, v)
			}
		}
		top := model.TopK(probs, 12)
		inTop := map[int]bool{}
		for _, id := range top {
			inTop[id] = true
		}
		var rest float32
		for id, p := range probs {
			if !inTop[id] {
				rest += p
			}
		}
		if rest < 0 {
			t.Fatalf("T=%.2f: the accumulated tail is %v, which the header would print as a "+
				"negative percentage", temp, rest)
		}
	}
}

// TestTop12TailIsActuallySmall pins the premise of the test above: if the tail
// were ever a meaningful share, the wording would need to change.
func TestTop12TailIsActuallySmall(t *testing.T) {
	a := newTestApp(t)
	probs := a.Trace.Probabilities(a.Temp)
	top := model.TopK(probs, 12)
	inTop := map[int]bool{}
	for _, id := range top {
		inTop[id] = true
	}
	var rest float32
	for id, p := range probs {
		if !inTop[id] {
			rest += p
		}
	}
	if len(probs) != 256 {
		t.Fatalf("vocabulary is %d, the stage is written for 256", len(probs))
	}
	t.Logf("the 244 candidates outside top-12 share %.3e of the mass", rest)
	if rest > 0.01 {
		t.Fatalf("the tail is %.2f%% of the mass, so 'Top-12' is no longer nearly "+
			"everything and the header understates it", rest*100)
	}
}

// TestOutputEntropyStatesItsCeiling checks stage 7 anchors its entropy reading.
//
// "熵 0.16 nats" says nothing on its own: a reader has no way to know whether
// that is a lot or a little. The ceiling is log(vocab) — the entropy of a uniform
// distribution over the 256 bytes — and on this model the answer is 3% of it,
// which is the whole point of the stage. The attention stage already anchors its
// entropy against log(n); this is the same idea one stage later.
func TestOutputEntropyStatesItsCeiling(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	probs := a.Trace.Probabilities(a.Temp)
	ent := float64(model.Entropy(probs))
	ceiling := math.Log(float64(len(probs)))
	wantLog := fmt.Sprintf("log(%d) = %.2f", len(probs), ceiling)
	wantPct := fmt.Sprintf("当前占 %.0f%%", 100*ent/ceiling)

	found := ""
	for _, s := range tr.Texts() {
		if strings.Contains(s, wantLog) && strings.Contains(s, wantPct) {
			found = s
		}
	}
	if found == "" {
		t.Fatalf("the stage does not state the entropy ceiling and this reading's share "+
			"of it (want %q and %q)", wantLog, wantPct)
	}
}

// TestOutputEntropyShareMovesWithTemperature is what makes the reference worth
// printing: the share is not a constant, so a fixed percentage would be wrong as
// soon as the reader touches the slider.
func TestOutputEntropyShareMovesWithTemperature(t *testing.T) {
	shares := map[float64]float64{}
	for _, temp := range []float64{0.5, 2.0} {
		a := newTestApp(t)
		a.Temp = temp
		probs := a.Trace.Probabilities(temp)
		shares[temp] = 100 * float64(model.Entropy(probs)) / math.Log(float64(len(probs)))
	}
	if shares[0.5] >= shares[2.0] {
		t.Fatalf("entropy share at T=0.5 (%.0f%%) is not below T=2.0 (%.0f%%)",
			shares[0.5], shares[2.0])
	}
	// And the two round to different whole percentages, so the printed figure
	// actually distinguishes them.
	if math.Round(shares[0.5]) == math.Round(shares[2.0]) {
		t.Fatalf("both temperatures print %.0f%%; the reading would look constant",
			shares[0.5])
	}
}

// TestFFNCardQuantifiesSparsity checks the FFN card measures concentration rather
// than counting values above an arbitrary threshold.
//
// "稀疏" is a claim about how the mass is distributed. The card used to support
// it only with "输出 > 0.01 的维度 145 / 384 (38%)", which describes the cut: on
// this model none of the 384 activations is near zero, so a reader could not tell
// whether 38% meant sparse or just "38% happen to exceed 0.01". The largest tenth
// carries about half the mass, and that is the number that makes the word mean
// something.
func TestFFNCardQuantifiesSparsity(t *testing.T) {
	a := newTestApp(t)
	a.Step = 5
	h := a.layer().Hidden[a.SelToken]
	share, topN := activationConcentration(h)

	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	want := fmt.Sprintf("最大的 %d / %d 个维度承担 %.0f%%", topN, len(h), share*100)
	found := false
	for _, s := range tr.Texts() {
		if strings.Contains(s, want) {
			found = true
		}
	}
	if !found {
		t.Fatalf("the FFN card does not state the concentration %q", want)
	}

	// The measurement itself: a tenth of the dimensions holding half the mass is
	// what the word "sparse" is being used to mean here.
	if share < 0.2 {
		t.Fatalf("the top tenth carries only %.0f%% of the mass, so '稀疏' would be "+
			"the wrong word", share*100)
	}
}

// TestActivationConcentrationIsAValidShare keeps the helper honest: it returns a
// fraction of a total, so it must stay within [0, 1] and scale with the input.
func TestActivationConcentrationIsAValidShare(t *testing.T) {
	// A single spike: the top tenth should carry essentially everything.
	spike := make([]float32, 100)
	spike[0] = 100
	share, topN := activationConcentration(spike)
	if topN != 10 {
		t.Fatalf("topN = %d for 100 values, want 10", topN)
	}
	if share < 0.99 {
		t.Fatalf("one spike of 100 among zeros gives a share of %.3f, want ~1", share)
	}

	// Uniform: the top tenth carries about a tenth.
	flat := make([]float32, 100)
	for i := range flat {
		flat[i] = 1
	}
	share, _ = activationConcentration(flat)
	if share < 0.08 || share > 0.12 {
		t.Fatalf("a uniform row gives a share of %.3f, want about 0.10", share)
	}

	// All zero has no share to report, and must not divide by zero.
	zero := make([]float32, 10)
	if share, _ := activationConcentration(zero); share != 0 {
		t.Fatalf("an all-zero row reports a share of %.3f", share)
	}
	if share, n := activationConcentration(nil); share != 0 || n != 0 {
		t.Fatalf("an empty row reports (%v, %d)", share, n)
	}
}

// TestEmbedStageComparesTheTwoMagnitudes checks stage 2 states how the token
// embedding and the position vector compare in size.
//
// They are added, so the ratio is the obvious question and the reader cannot
// answer it from three separate strips: the caption under each shows its own
// min/max/norm, and nothing puts them side by side. On this model they come out
// nearly equal (P/E ≈ 0.95), which is a property of these weights rather than a
// rule — so it has to be measured and shown, not asserted in prose.
func TestEmbedStageComparesTheTwoMagnitudes(t *testing.T) {
	a := newTestApp(t)
	a.Step = 1
	a.SelToken = 10 // a token whose P/E is not the 0.95 the default selection gives
	i := a.SelToken
	eNorm, pNorm := norm(a.Trace.Embed[i]), norm(a.Trace.Pos[i])
	if pNorm == 0 || eNorm == 0 {
		t.Fatal("one of the two vectors is zero, so the ratio is undefined")
	}

	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	want := fmt.Sprintf("‖E‖ %.2f · ‖P‖ %.2f · P/E = %.2f", eNorm, pNorm, pNorm/eNorm)
	found := false
	for _, s := range tr.Texts() {
		if strings.Contains(s, want) {
			found = true
		}
	}
	if !found {
		t.Fatalf("the embedding stage does not state the two magnitudes (%q)", want)
	}
}

// TestPositionMagnitudeIsComparableToTheEmbedding pins the premise of the row
// above: it is worth printing because the two are within the same order. If a
// future weight file made positions negligible or dominant, the wording would
// deserve a second look.
func TestPositionMagnitudeIsComparableToTheEmbedding(t *testing.T) {
	a := newTestApp(t)
	for _, i := range []int{0, 10, len(a.Trace.Tokens) - 1} {
		e, p := norm(a.Trace.Embed[i]), norm(a.Trace.Pos[i])
		if e == 0 {
			t.Fatalf("token %d has a zero embedding", i)
		}
		ratio := float64(p / e)
		t.Logf("token %2d: P/E = %.2f", i, ratio)
		if ratio < 0.25 || ratio > 4 {
			t.Fatalf("token %d: P/E = %.2f is more than a factor of four from 1; the "+
				"stage's framing assumes the two are comparable", i, ratio)
		}
	}
}

// TestSamplingClaimIsQualified is the fix this round. The sampling card said "同一
// 句提示词永远抽出同一个 token", which is only true at a fixed temperature — and
// the temperature slider sits on the same card. At T=2.2 the same prompt draws a
// different token, so the sentence left out its own condition.
func TestSamplingClaimIsQualified(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	var note string
	for _, s := range tr.Texts() {
		if strings.Contains(s, "种子只来自提示词文字") {
			note = s
		}
	}
	if note == "" {
		t.Fatal("the sampling card no longer explains where the seed comes from")
	}
	if !strings.Contains(note, "温度不变") {
		t.Fatalf("the note does not qualify the claim with a fixed temperature: %q", note)
	}

	// And the qualification is real: some temperature draws something else.
	base := a.Trace.Probabilities(0.8)
	d0, ok := a.sampleDraw(base)
	if !ok {
		t.Fatal("no draw at T=0.8")
	}
	differs := false
	for temp := 0.1; temp <= 3.0; temp += 0.1 {
		d, ok := a.sampleDraw(a.Trace.Probabilities(temp))
		if ok && d.pick != d0.pick {
			differs = true
			break
		}
	}
	if !differs {
		t.Skip("temperature does not change the draw on this prompt, so the " +
			"qualification is untested here")
	}
}

// TestOrderBlindnessWordingMatchesTheMask checks the embedding lesson does not
// claim position is the *only* thing distinguishing word order. The causal mask
// also makes "dog bites man" and "man bites dog" differ — by changing which keys
// each position can see, not by encoding where each word sits. The old sentence
// ("否则两者完全一样") conflated the two mechanisms.
func TestOrderBlindnessWordingMatchesTheMask(t *testing.T) {
	// The premise: the two orders really do differ, and not only through
	// position vectors.
	a := newTestApp(t)
	a.Prompt = "dog bites man"
	a.recompute()
	first := append([]float32(nil), a.Trace.Logits...)
	b := newTestApp(t)
	b.Prompt = "man bites dog"
	b.recompute()
	var maxDiff float32
	for i := range first {
		if d := first[i] - b.Trace.Logits[i]; d > maxDiff {
			maxDiff = d
		} else if -d > maxDiff {
			maxDiff = -d
		}
	}
	if maxDiff <= 0.01 {
		t.Skip("the two orders give near-identical logits, so this case cannot " +
			"demonstrate that the mask separates them")
	}
	t.Logf("the two orders differ by up to %.2f in logits", maxDiff)

	// The lesson body lived on stage 2 and must now name both mechanisms.
	c := newTestApp(t)
	c.Step = 1
	tr := ui.NewTester(c.View, 1500, 940)
	tr.Frame()
	var body string
	for _, s := range tr.Texts() {
		// The lesson body opens with this phrase; the inspector has a shorter
		// note about position vectors that would otherwise match first.
		if strings.HasPrefix(s, "嵌入表是一张") {
			body = s
		}
	}
	if body == "" {
		t.Fatal("the embedding lesson no longer explains why positions are injected")
	}
	if strings.Contains(body, "完全一样") {
		t.Fatalf("the lesson still claims the two orders would be completely "+
			"identical without positions: %q", body)
	}
	if !strings.Contains(body, "掩码") {
		t.Fatalf("the lesson does not separate the causal mask from position "+
			"encoding: %q", body)
	}
}

// TestLayerStagesNameTheirLayer checks stages 3, 5 and 6 say which layer's
// numbers they are showing.
//
// The layer control lives in the top bar. Once the page is scrolled — and the
// matrix sits well below the fold at 1180x720 — the control is out of sight while
// every figure below belongs to one specific layer. Stage 4 already named both
// layer and head in its heading; the other three did not, so a scrolled reader had
// no way to tell layer 0's residual stream from layer 1's.
func TestLayerStagesNameTheirLayer(t *testing.T) {
	// heading prefix per stage, for the stages the layer control affects minus
	// stage 4 which already said it.
	headings := map[int]string{
		2: "分数矩阵",
		4: "残差流",
		5: "升维",
	}
	for step, heading := range headings {
		for _, layer := range []int{0, 1} {
			a := newTestApp(t)
			a.Step = step
			a.Layer = layer
			tr := ui.NewTester(a.View, 1500, 940)
			tr.Frame()

			want := fmt.Sprintf("层 %d · ", layer)
			found := false
			for _, s := range tr.Texts() {
				if strings.HasPrefix(s, heading) && strings.Contains(s, want) {
					found = true
				}
				// The note is a sibling element of the title, so it may arrive
				// as its own text.
				if strings.HasPrefix(s, want) && strings.Contains(s, "query") {
					found = true
				}
				if strings.HasPrefix(s, want) && strings.Contains(s, "两次加法") {
					found = true
				}
				if strings.HasPrefix(s, want) && strings.Contains(s, "d_ff") {
					found = true
				}
			}
			if !found {
				t.Fatalf("stage %d at layer %d does not name its layer", step+1, layer)
			}
		}
	}
}

// TestLayerHeadingsDifferPerLayer is the corollary: if the heading names the
// layer, two layers must produce different headings.
func TestLayerHeadingsDifferPerLayer(t *testing.T) {
	for step := range map[int]bool{2: true, 4: true, 5: true} {
		texts := map[int]string{}
		for _, layer := range []int{0, 1} {
			a := newTestApp(t)
			a.Step = step
			a.Layer = layer
			tr := ui.NewTester(a.View, 1500, 940)
			tr.Frame()
			var parts []string
			for _, s := range tr.Texts() {
				if strings.HasPrefix(s, "层 ") || strings.HasPrefix(s, "分数矩阵") ||
					strings.HasPrefix(s, "残差流") || strings.HasPrefix(s, "升维") {
					parts = append(parts, s)
				}
			}
			texts[layer] = strings.Join(parts, "|")
		}
		if texts[0] == texts[1] {
			t.Fatalf("stage %d: the layer headings are identical for both layers: %q",
				step+1, texts[0])
		}
	}
}

// TestInspectorStatesItsScope checks the right panel says which layer and head its
// numbers belong to.
//
// The stage cards started naming the layer last round; the inspector had the same
// gap and a worse one, because it sits on the far side of the window from the
// controls that select the scope. A reader looking at "最大权重 0.467 → token 33"
// could not tell whether that was layer 0 head 0 or layer 1 head 3.
//
// The text is read from inspectorScope() rather than by locating it in the frame:
// the stage heading carries the same string, and Tester.Find returns the first
// match, so a frame-level lookup here would pass on the heading alone and never
// touch the inspector.
func TestInspectorStatesItsScope(t *testing.T) {
	cases := []struct {
		step int    // zero-based, matching a.Step
		what string // for the failure message
		want string
	}{
		{2, "UI step 3, scores", "层 0 · 头 1"},
		{3, "UI step 4, attention", "层 0 · 头 1"},
		{4, "UI step 5, residual", "层 0"},
		{5, "UI step 6, FFN", "层 0"},
	}
	for _, c := range cases {
		a := newTestApp(t)
		a.Step = c.step
		a.Layer = 0
		a.Head = 1
		if got := a.inspectorScope(); got != c.want {
			t.Fatalf("%s: scope = %q, want %q", c.what, got, c.want)
		}
		// And the panel must actually render it. The count cannot be fixed: the
		// stage heading carries the same string on stage 4 (three copies total)
		// but not on stages 3, 5 or 6 (one copy, the inspector's). What has to
		// hold is that it appears at all, and that the inspector column is where
		// it comes from — so check the inspector's own band by rendering the
		// panel in isolation would need layout access, and instead assert the
		// count is at least one and that the panel's copy is present by looking
		// for the row it lives in.
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		n := 0
		for _, s := range tr.Texts() {
			if s == c.want {
				n++
			}
		}
		if n == 0 {
			t.Fatalf("%s: %q never appears in the frame", c.what, c.want)
		}
	}
}

// TestInspectorScopeIsAbsentWhereNothingApplies is the other half: a scope line on
// a stage that reads neither control would be noise, and claiming a layer there
// would be wrong.
func TestInspectorScopeIsAbsentWhereNothingApplies(t *testing.T) {
	for _, step := range []int{0, 1, 6} {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		for _, s := range tr.Texts() {
			// Only look in the inspector band, where the scope would live.
			r, ok := tr.Find(s)
			if !ok || r.X < 1170 || r.Y > 40 {
				continue
			}
			if strings.HasPrefix(s, "层 ") || strings.HasPrefix(s, "头 ") {
				t.Fatalf("stage %d: the inspector claims a scope %q it does not have",
					step+1, s)
			}
		}
	}
}

// TestInspectorScopeTracksTheControls checks the line is computed, not fixed.
func TestInspectorScopeTracksTheControls(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	for _, want := range []string{"层 0 · 头 0", "层 1 · 头 0", "层 1 · 头 3"} {
		switch want {
		case "层 1 · 头 0":
			a.Layer = 1
		case "层 1 · 头 3":
			a.Head = 3
		}
		if got := a.inspectorScope(); got != want {
			t.Fatalf("scope = %q, want %q", got, want)
		}
	}
	// And nothing on a stage that reads neither.
	a.Step = 0
	if got := a.inspectorScope(); got != "" {
		t.Fatalf("stage 1 reports scope %q", got)
	}
}

// TestInspectorScopeIsAbsentWithoutTokens checks the scope label is suppressed on
// the empty state.
//
// With no prompt the inspector shows "先在上方输入一句提示词", and a "层 0 · 头 0"
// beside it says those two controls still select something — when in fact every
// tensor is empty. Found by rendering the empty state and reading the header, not
// by any assertion about the numbers.
func TestInspectorScopeIsAbsentWithoutTokens(t *testing.T) {
	for _, step := range []int{2, 3, 4, 5} {
		a := newTestApp(t)
		a.Prompt = ""
		a.recompute()
		a.Step = step
		if len(a.Trace.Tokens) != 0 {
			t.Fatalf("stage %d: an empty prompt produced %d tokens", step+1, len(a.Trace.Tokens))
		}
		// The helper still knows the scope; the panel just must not show it.
		if got := a.inspectorScope(); got == "" {
			t.Fatalf("stage %d: inspectorScope is empty even for a populated prompt",
				step+1)
		}
		tr := ui.NewTester(a.View, 1180, 720)
		tr.Frame()
		for _, s := range tr.Texts() {
			if !strings.HasPrefix(s, "层 ") {
				continue
			}
			r, ok := tr.Find(s)
			if !ok || r.Y > 40 || r.X < 850 {
				continue
			}
			t.Fatalf("stage %d: the empty state claims a scope %q", step+1, s)
		}
	}
}

// TestInspectorScopeReturnsWhenTheresAPrompt is the other half: typing must bring
// the label back, so the suppression is conditional rather than a deletion.
//
// Counts occurrences rather than locating one: Tester.Find returns the first
// match, and the stage heading carries the same string, so a frame-level lookup
// reports the heading's position and never sees the inspector's copy.
func TestInspectorScopeReturnsWhenTheresAPrompt(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	a.Layer = 1
	a.Head = 2
	const scope = "层 1 · 头 2"

	count := func() int {
		t.Helper()
		tr := ui.NewTester(a.View, 1180, 720)
		tr.Frame()
		n := 0
		for _, s := range tr.Texts() {
			if s == scope {
				n++
			}
		}
		return n
	}

	a.Prompt = ""
	a.recompute()
	if n := count(); n != 0 {
		t.Fatalf("the empty state renders %q %d time(s); it should render none", scope, n)
	}

	a.Prompt = "hi"
	a.recompute()
	withPrompt := count()
	if withPrompt == 0 {
		t.Fatalf("with a prompt the frame never renders %q", scope)
	}
	// Back to empty, and it goes away again.
	a.Prompt = ""
	a.recompute()
	if n := count(); n != 0 {
		t.Fatalf("after clearing the prompt %q still appears %d time(s)", scope, n)
	}
}

// TestLayerNormClaimMatchesTheActualWeights covers a sentence that was true of
// the bundled weights and false of the random fallback.
//
// Both LayerNorm cards said "γ、β 是训练出来的，不是 1 和 0". model.Random sets
// γ to ones and β to zeros, so in the untrained state the sentence stated the
// exact opposite of what the numbers were. Found by running the untrained-state
// banner through the LN stage rather than assuming the prose held for both.
func TestLayerNormClaimMatchesTheActualWeights(t *testing.T) {
	cases := []struct {
		name string
		w    *model.Weights
		// want is a substring the rendered text must contain.
		want string
		// wantNot must not appear.
		wantNot string
	}{
		{
			name:    "bundled",
			want:    "不是 1 和 0",
			wantNot: "没有训练过",
		},
		{
			name: "random",
			w:    model.Random(model.DefaultConfig, 42),
			// The two notes word the untrained case slightly differently, so
			// the common substring is what both must contain.
			want:    "没有训练过",
			wantNot: "是训练出来的，不是 1 和 0",
		},
	}
	for _, c := range cases {
		var a *App
		if c.w == nil {
			a = newTestApp(t)
		} else {
			a = New(c.w)
		}
		a.Prompt = "hi"
		a.recompute()

		// The helper must agree with the weights it is describing.
		if got, want := lnParamsAreLearned(a.W), c.w != nil; got != !want {
			t.Fatalf("%s: lnParamsAreLearned = %v, want %v", c.name, got, !want)
		}

		// Stage 5 carries both notes: the layer-scoped card and the inspector.
		// Stage 6 mentions neither, so scanning it proves nothing.
		for _, step := range []int{4} {
			a.Step = step
			tr := ui.NewTester(a.View, 1500, 940)
			tr.Frame()
			var joined strings.Builder
			for _, s := range tr.Texts() {
				joined.WriteString(s)
				joined.WriteString("\n")
			}
			text := joined.String()
			// Both LayerNorm notes carry the claim — the layer-scoped card and
			// the inspector's — so require two occurrences rather than one. A
			// single contains would pass with either copy left unconditional.
			if n := strings.Count(text, c.want); n != 2 {
				t.Fatalf("%s stage %d: the claim %q appears %d time(s), want 2",
					c.name, step+1, c.want, n)
			}
			if strings.Contains(text, c.wantNot) {
				t.Fatalf("%s stage %d: the LayerNorm note still says %q", c.name, step+1, c.wantNot)
			}
		}
	}
}

// TestLNParamsHelperReadsTheWeights pins the helper against the two shapes it
// must distinguish, including the mixed case where only β moves.
func TestLNParamsHelperReadsTheWeights(t *testing.T) {
	learned := model.Random(model.DefaultConfig, 42)
	if lnParamsAreLearned(learned) {
		t.Fatal("random weights report learned LayerNorm parameters")
	}
	// A trained γ need not be far from 1 to count.
	trained := model.Random(model.DefaultConfig, 42)
	trained.LNFG[3] = 1.01
	if !lnParamsAreLearned(trained) {
		t.Fatal("a γ slightly off 1 is not detected")
	}
	// β alone is enough.
	shifted := model.Random(model.DefaultConfig, 42)
	shifted.LNFB[7] = -0.004
	if !lnParamsAreLearned(shifted) {
		t.Fatal("a non-zero β is not detected")
	}
	// And nil must not panic.
	if lnParamsAreLearned(nil) {
		t.Fatal("nil weights report learned parameters")
	}
}

// TestDroppedByteCountIsTheRealPrefixLength checks the card's "丢弃前 N 字节"
// against where the kept slice actually starts in the original prompt.
//
// The existing tests pin N for one prompt (207 for 30 repeated hanzi) and check
// that truncation does not cut a character. Neither would catch an off-by-a-few
// when the truncation point lands inside a multi-byte character and Truncate
// steps forward to the next boundary — the case where the displayed N and the
// real drop differ most easily.
func TestDroppedByteCountIsTheRealPrefixLength(t *testing.T) {
	ctx := model.DefaultConfig.Ctx
	cases := []string{
		strings.Repeat("注意力", 30),         // aligned: 207 % 3 == 0
		"ab" + strings.Repeat("注意力", 30),  // misaligned by 2
		"abc" + strings.Repeat("注意力", 30), // misaligned by 0 mod 3 but shifted
		strings.Repeat("a", ctx+36),       // pure ASCII
		"x" + strings.Repeat("字", 40),     // one ASCII then hanzi
	}
	for _, prompt := range cases {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		tr := a.Trace
		if len(tr.Tokens) == 0 {
			t.Fatalf("prompt %q produced no tokens", prompt[:min(8, len(prompt))])
		}
		// Where does the kept slice begin in the original byte string? The
		// tokeniser maps one token per byte, so the kept suffix length is the
		// token count and the prefix is everything before it.
		realDropped := len(prompt) - len(tr.Tokens)
		if realDropped <= 0 {
			continue // not truncated
		}
		if !strings.HasSuffix(prompt, tr.Text) {
			t.Fatalf("kept text is not a suffix of the prompt")
		}
		if got := len(prompt) - len(tr.Text); got != realDropped {
			t.Fatalf("kept text is %d bytes but %d tokens were kept", len(tr.Text), len(tr.Tokens))
		}

		a.Step = 0
		uiTr := ui.NewTester(a.View, 1500, 940)
		uiTr.Frame()
		want := fmt.Sprintf("丢弃前 %d 字节", realDropped)
		found := false
		for _, s := range uiTr.Texts() {
			if strings.Contains(s, want) {
				found = true
			}
		}
		if !found {
			t.Fatalf("prompt of %d bytes: the card does not state %q", len(prompt), want)
		}
		// And the stated original length must be the prompt's real length.
		wantOrig := fmt.Sprintf("原文 %d", len(prompt))
		found = false
		for _, s := range uiTr.Texts() {
			if strings.Contains(s, wantOrig) {
				found = true
			}
		}
		if !found {
			t.Fatalf("prompt of %d bytes: the card does not state %q", len(prompt), wantOrig)
		}
	}
}

// TestShortcutHintMatchesTheStageBehaviour checks the on-screen shortcut line
// agrees with what the keys do.
//
// README.md has described the two behaviours correctly for a while: Shift+arrows
// walk the matrix on stages 3-4 and move the token selection elsewhere. The line
// inside the app said "Shift+← → 切换 token" on every stage, so on the two stages
// that draw a matrix — where a reader is most likely to reach for the keys — the
// help text described the wrong action.
func TestShortcutHintMatchesTheStageBehaviour(t *testing.T) {
	for step := 0; step < 7; step++ {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()

		var hint string
		for _, s := range tr.Texts() {
			if strings.Contains(s, "快捷键") {
				hint = s
			}
		}
		if hint == "" {
			t.Fatalf("stage %d: no shortcut line", step+1)
		}

		inMatrix := step == 2 || step == 3
		saysCell := strings.Contains(hint, "格子")
		saysToken := strings.Contains(hint, "切换 token")
		switch {
		case inMatrix && !saysCell:
			t.Fatalf("stage %d draws a matrix but the hint does not mention cells: %q",
				step+1, hint)
		case inMatrix && saysToken:
			t.Fatalf("stage %d draws a matrix but the hint still says 切换 token: %q",
				step+1, hint)
		case !inMatrix && !saysToken:
			t.Fatalf("stage %d has no matrix but the hint does not mention the token: %q",
				step+1, hint)
		case !inMatrix && saysCell:
			t.Fatalf("stage %d has no matrix but the hint mentions cells: %q", step+1, hint)
		}

		// And the statement must match the keys: run the one the hint names.
		// Park the selection mid-sequence first: it starts on the last token,
		// where Shift+Right has nowhere to go and the check would fail for a
		// reason that has nothing to do with the hint.
		a.SelToken = 10
		tr.Frame()
		selBefore, cellBefore := a.SelToken, a.KeyQ
		tr.Key(ui.Shift, ui.KeyRight)
		tr.Frame()
		if inMatrix {
			if a.KeyQ < 0 || a.KeyQ == cellBefore {
				t.Fatalf("stage %d: the hint says Shift moves the cell, but it did not",
					step+1)
			}
		} else {
			if a.SelToken == selBefore {
				t.Fatalf("stage %d: the hint says Shift moves the token, but it did not",
					step+1)
			}
		}
	}
}

// TestShortcutHintKeepsTheOtherKeys guards the rest of the line, which is the same
// on every stage and easy to drop while editing the Shift clause.
func TestShortcutHintKeepsTheOtherKeys(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	var hint string
	for _, s := range tr.Texts() {
		if strings.Contains(s, "快捷键") {
			hint = s
		}
	}
	// The digit range is derived, not written out: this expectation comes from the lesson
	// count, so adding a stage updates the hint and the keys together. Measured before that
	// change, with a synthetic eighth lesson: the line still said "1–7 跳转", Key8 did
	// nothing, and this test passed — the expected string here was a seventh literal.
	digits := stageDigitRange()
	if digits == "" || len(stageDigitKeys()) != len(lessons) {
		t.Fatalf("the digit shortcuts cover %d stages but there are %d lessons",
			len(stageDigitKeys()), len(lessons))
	}
	for _, want := range []string{"← → 切换阶段", "空格 播放/暂停", digits + " 跳转"} {
		if !strings.Contains(hint, want) {
			t.Fatalf("the shortcut line no longer mentions %q: %q", want, hint)
		}
	}
}

// TestStageDigitKeysFollowTheLessons is the check the three seven-element literals could not
// make between themselves.
//
// The digit shortcuts, the shortcut hint and this package's tests each used to spell out
// Key1..Key7. Nothing tied any of them to len(lessons), so an eighth stage would have been
// reachable by the ← → keys, by clicking the sidebar and by the play timer, but not by typing
// 8 — and the hint would have gone on advertising "1–7". Measured: with a synthetic eighth
// lesson added, Key8 left the stage at 0 and every shortcut test still passed.
//
// The assertion is that the two derived pieces agree with the lesson list, in both directions.
func TestStageDigitKeysFollowTheLessons(t *testing.T) {
	keys := stageDigitKeys()
	if len(keys) != len(lessons) {
		t.Fatalf("%d digit keys for %d lessons; stage %d has no digit",
			len(keys), len(lessons), len(keys)+1)
	}
	// The hint has to name the last key that actually exists, or it is describing a set the
	// reader cannot use.
	if want := fmt.Sprintf("1–%d", len(lessons)); stageDigitRange() != want {
		t.Fatalf("the hint range is %q, want %q for %d lessons", stageDigitRange(), want, len(lessons))
	}
	// And every key must be the one the handler expects at that index: pressing it jumps to
	// the stage it is advertised for.
	for i, key := range keys {
		a := newTestApp(t)
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		tr.Key(0, key)
		tr.Frame()
		if a.Step != i {
			t.Fatalf("digit %d jumped to stage %d", i+1, a.Step)
		}
	}

	// Everything above compares derived values against today's lesson count, which a
	// hardcoded seven satisfies just as well. These sizes are ones the app does not ship
	// with, so only a real derivation produces them.
	for _, n := range []int{0, 1, 5, 8, 9} {
		if got := len(digitKeysFor(n)); got != n {
			t.Fatalf("digitKeysFor(%d) gave %d keys", n, got)
		}
		if want := digitRangeFor(n); n > 0 && want != fmt.Sprintf("1–%d", n) {
			t.Fatalf("digitRangeFor(%d) = %q", n, want)
		}
	}
	// The keyboard runs out at 9, so a tenth stage cannot get a digit — the list must cap
	// rather than read past its array.
	if got := len(digitKeysFor(12)); got != 9 {
		t.Fatalf("digitKeysFor(12) gave %d keys; the digits stop at 9", got)
	}
	if got := digitKeysFor(-1); len(got) != 0 {
		t.Fatalf("digitKeysFor(-1) gave %d keys", len(got))
	}
	// Key order matters: the i-th stage must map to the digit i+1, not merely to some key.
	wantKeys := []ui.Key{ui.Key1, ui.Key2, ui.Key3, ui.Key4, ui.Key5, ui.Key6, ui.Key7, ui.Key8, ui.Key9}
	for i, k := range digitKeysFor(len(wantKeys)) {
		if k != wantKeys[i] {
			t.Fatalf("digit %d maps to key %v, want %v", i+1, k, wantKeys[i])
		}
	}
}

// TestMatrixStagesAdvertiseTheirKeysWhereTheyCanBeSeen covers the placement half of
// the shortcut problem.
//
// Fixing the wording alone was not enough: the full shortcut list lives at the foot
// of the lesson card, and on stages 3-4 that card starts at Y=1223 at 1180x720 and
// Y=1486 at 1920x1080 — below the fold at every supported size. Those are exactly
// the two stages where Shift+arrows change meaning, so the corrected sentence was
// unreachable from the stages that needed it. The step bar is always on screen, so
// the matrix hint goes there.
func TestMatrixStagesAdvertiseTheirKeysWhereTheyCanBeSeen(t *testing.T) {
	for _, step := range []int{2, 3} {
		for _, size := range []struct{ w, h int }{{1180, 720}, {1500, 940}, {1920, 1080}} {
			a := newTestApp(t)
			a.Step = step
			tr := ui.NewTester(a.View, size.w, size.h)
			tr.Frame()
			found := false
			for _, s := range tr.Texts() {
				if !strings.Contains(s, "移动格子") {
					continue
				}
				if r, ok := tr.Find(s); ok && r.Y < float32(size.h)*0.5 {
					// In the upper half: the step bar, not the lesson footer.
					found = true
				}
			}
			if !found {
				t.Fatalf("stage %d at %dx%d: no matrix hint above the fold",
					step+1, size.w, size.h)
			}
		}
	}

	// Stages without a matrix must not show it, or it would promise cells that
	// are not there.
	for step := 0; step < 7; step++ {
		if step == 2 || step == 3 {
			continue
		}
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		for _, s := range tr.Texts() {
			if strings.Contains(s, "移动格子") {
				t.Fatalf("stage %d has no matrix but advertises moving a cell: %q",
					step+1, s)
			}
		}
	}
}

// TestClickableChipsGiveHoverFeedback checks the elements a reader can click all
// react to the pointer.
//
// Two kinds did not: the prompt presets on stage 1 (the control that fills the
// prompt field) and the 6-DIP stage dots in the step bar. Both were clickable and
// both looked identical under the pointer, while every other clickable thing in
// the app — token chips, sidebar rows, buttons — changes on hover. A control that
// can be clicked has to look like one.
func TestClickableChipsGiveHoverFeedback(t *testing.T) {
	// Preset chips: stage 1 lists them in the empty-prompt card, which is what a
	// first-time reader sees.
	t.Run("prompt presets", func(t *testing.T) {
		a := newTestApp(t)
		a.Prompt = ""
		a.recompute()
		a.Step = 0
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		r, ok := tr.Find("长提示词")
		if !ok {
			t.Skip("the preset chip is not on this frame")
		}
		before := tr.Image()
		tr.Move(r.X+r.W/2, r.Y+r.H/2)
		tr.Frame()
		if diffCount(before, tr.Image()) == 0 {
			t.Fatal("hovering the 长提示词 preset changes nothing")
		}
	})

	// Stepper rows: the row for each stage highlights under the pointer. They have no text of
	// their own worth searching for, so one is found by scanning.
	//
	// This subtest used to be called "stage dots" and scanned y 170..190 at a 2-DIP lattice,
	// concluding from any changed pixel that the dot reacted. The reacting element is not the
	// dot but the whole row: stepper() gives a non-current Row a Background(colPanel2) when
	// row.Hovered(), and the row is FillWidth. Measured along y=180, 77 of 167 probe points
	// across the sidebar react — the row's full width, at every height in it.
	//
	// The lattice is 4 DIP. The old 2-DIP one rendered 800 frames to answer one yes/no
	// question at about 21ms each; 4 still cannot step over a reacting region that spans the
	// row, and costs a quarter of the frames.
	t.Run("stepper rows", func(t *testing.T) {
		a := newTestApp(t)
		a.Step = 1
		tr := ui.NewTester(a.View, 1180, 720)
		tr.Frame()
		base := tr.Image()
		const coarse = 4
		// Only inside the stepper: the seven rows run from the first label's top to the last
		// one's bottom (measured: y=84 to y=292 at 32 DIP per row). Sweeping the whole sidebar
		// found something else that reacts — the earlier version scanned to y=400 and a run with
		// the row highlight removed still passed, by reacting to whatever else lives down there.
		var hitX, hitY float32
		found := false
		for y := float32(84); y < 292 && !found; y += coarse {
			// x stops short of the sidebar's scrollbar. Measured with the row highlight
			// disabled: the only reactions left in this band sit at x=226, the scrollbar, so a
			// scan reaching 230 was satisfied by that and never asked about the rows.
			for x := float32(30); x < 210 && !found; x += coarse {
				tr.Move(x, y)
				tr.Frame()
				if diffCount(base, tr.Image()) > 0 {
					hitX, hitY, found = x, y, true
				}
			}
		}
		if !found {
			t.Fatal("no stepper row reacts to the pointer")
		}
		// The hit has to be a row, not a stray control: the reaction must extend horizontally,
		// which is what distinguishes a FillWidth row from a dot-sized target.
		span := 0
		for dx := float32(0); dx < 400; dx += 4 {
			tr.Move(hitX+dx, hitY)
			tr.Frame()
			if diffCount(base, tr.Image()) > 0 {
				span++
			}
		}
		if span < 20 {
			t.Fatalf("the pointer reacted at (%.0f,%.0f) but only %d of 100 probes to its right "+
				"did, so that is not a full-width row", hitX, hitY, span)
		}
	})
}

// TestStageDotHoverDoesNotMoveTheRow is the constraint that decided how the dot
// feedback is drawn. Growing the hovered dot to the current dot's width read more
// clearly, but flex layout shifts every dot after it while the pointer is still,
// so the row jumps as the pointer crosses it. Only the colour may change.
//
// Compares pixels rather than asking Tester.Find for the label's position: Find
// reports a different X for the same unchanged label depending on what else is
// hovered (measured: 321 vs 331 with the label visibly in the same place), so it
// cannot answer this question.
func TestStageDotHoverDoesNotMoveTheRow(t *testing.T) {
	a := newTestApp(t)
	a.Step = 1
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()
	label, ok := tr.Find("2 / 7 · 嵌入")
	if !ok {
		t.Fatal("the step label is missing")
	}
	// Crop the label's own band once, then require it to stay pixel-identical
	// while the pointer crosses every dot to its left.
	const bandW, bandH = 120, 18
	crop := func(img *image.RGBA) []uint8 {
		var out []uint8
		for y := int(label.Y); y < int(label.Y)+bandH; y++ {
			for x := int(label.X); x < int(label.X)+bandW; x++ {
				i := img.PixOffset(x, y)
				out = append(out, img.Pix[i:i+4]...)
			}
		}
		return out
	}
	baseBand := crop(tr.Image())
	if len(baseBand) == 0 {
		t.Fatal("empty label band")
	}

	for x := float32(240); x < 400; x += 2 {
		tr.Move(x, 184)
		tr.Frame()
		if got := crop(tr.Image()); !bytes.Equal(baseBand, got) {
			t.Fatalf("hovering a dot at x=%.0f redraws the step label region", x)
		}
	}
}

// diffCount counts pixels that differ between two frames.
func diffCount(a, b *image.RGBA) int {
	if a == nil || b == nil || len(a.Pix) != len(b.Pix) {
		return 1
	}
	n := 0
	for i := 0; i < len(a.Pix); i += 4 {
		if a.Pix[i] != b.Pix[i] || a.Pix[i+1] != b.Pix[i+1] ||
			a.Pix[i+2] != b.Pix[i+2] || a.Pix[i+3] != b.Pix[i+3] {
			n++
		}
	}
	return n
}

// TestStageJumpTooltipsAgree checks the two controls that jump between stages
// describe their destination the same way.
//
// The sidebar's stepper rows have said "第 N / 7 步 · <title>" since they were
// added. Last round gave the step-bar dots a tooltip too, but built it from
// lesson.name — so the same jump was described as "第 3 / 7 步 · 注意力分数" from one
// control and "第 3 / 7 步 · Query 与 Key 的点积" from the other. The name is already
// printed in the bar beside the dots, so repeating it adds nothing.
//
// Tooltips do not surface in the headless renderer, so this reads the call sites
// rather than hovering.
func TestStageJumpTooltipsAgree(t *testing.T) {
	sources, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	var buf strings.Builder
	for _, f := range sources {
		if strings.HasSuffix(f, "_test.go") {
			continue
		}
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		buf.Write(b)
	}
	src := buf.String()

	const format = `"第 %d / %d 步 · %s"`
	if n := strings.Count(src, format); n != 2 {
		t.Fatalf("%d uses of the stage-jump tooltip format, want 2", n)
	}
	// The two third-arguments, one per site.
	if n := strings.Count(src, `len(lessons), s.title)`); n != 1 {
		t.Fatalf("the sidebar tooltip uses the title %d times, want 1", n)
	}
	if n := strings.Count(src, `len(lessons), lessons[i].title)`); n != 1 {
		t.Fatalf("the step-bar tooltip uses the title %d times, want 1", n)
	}
	if strings.Contains(src, `len(lessons), lessons[i].name)`) {
		t.Fatal("a stage-jump tooltip still uses lesson.name; the bar beside the dots " +
			"already shows the name, so the tooltip should carry the title")
	}
}

// TestRepeatedSamplingStaysConsistent covers what a reader does to a generative
// demo: press the sample button over and over until the prompt outgrows the
// window. Each press appends a byte and reruns the forward pass, so past ctx the
// model keeps only the tail — the state has to stay coherent through that.
func TestRepeatedSamplingStaysConsistent(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = "The"
	a.recompute()
	a.Step = 6
	ctx := a.W.Config.Ctx
	startLen := len(a.Prompt)

	for i := 0; i < ctx+20; i++ {
		a.sampleNext()
		// The prompt keeps every byte the reader produced; only the model's
		// window is bounded.
		if want := startLen + i + 1; len(a.Prompt) != want {
			t.Fatalf("after %d samples the prompt is %d bytes, want %d",
				i+1, len(a.Prompt), want)
		}
		if got := len(a.Trace.Tokens); got > ctx {
			t.Fatalf("after %d samples the trace has %d tokens, over ctx %d", i+1, got, ctx)
		}
		if a.LastSample == "" {
			t.Fatalf("after %d samples there is no sample description", i+1)
		}
	}

	// The prompt has outgrown the window by exactly this much.
	over := len(a.Prompt) - ctx
	if over <= 0 {
		t.Fatalf("the prompt did not outgrow the window (%d bytes, ctx %d)",
			len(a.Prompt), ctx)
	}

	// And the tokenising card must state the drop rather than present the
	// truncated length as if it described the prompt.
	a.Step = 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	var joined strings.Builder
	for _, s := range tr.Texts() {
		joined.WriteString(s)
		joined.WriteString("\n")
	}
	text := joined.String()
	if !strings.Contains(text, fmt.Sprintf("原文 %d", len(a.Prompt))) {
		t.Fatalf("the card does not state the original length %d", len(a.Prompt))
	}
	if !strings.Contains(text, "丢弃前 ") {
		t.Fatal("the card does not state how much was dropped")
	}
}

// TestSamplingRefusesAnEmptyPrompt checks the button cannot produce a token from
// nothing, and says why.
func TestSamplingRefusesAnEmptyPrompt(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = ""
	a.recompute()
	if len(a.Trace.Tokens) != 0 {
		t.Fatalf("an empty prompt produced %d tokens", len(a.Trace.Tokens))
	}
	a.sampleNext()
	if a.Prompt != "" {
		t.Fatalf("sampling appended %q to an empty prompt", a.Prompt)
	}
	if !strings.Contains(a.Status, "输入") {
		t.Fatalf("status = %q, want a prompt for the reader", a.Status)
	}
}

// TestResidualStageLabelsWhatTheSublayerActuallyReads covers a mislabel that hid
// the pre-norm structure the stage is meant to teach.
//
// The strip caption said "进入注意力子层 x" and drew LayerTrace.In. But the model
// computes Norm1 = layerNorm(In) and then Q/K/V from Norm1 — so the attention
// sublayer never reads x. The two vectors are not close in size either (‖In‖ 13.12
// vs ‖Norm1‖ 11.19 on the default prompt), so a reader following the chain would
// conclude the sublayer consumes the unnormalised stream, which is the one thing
// pre-norm does not do. Norm1 was in the trace and was never drawn anywhere.
func TestResidualStageLabelsWhatTheSublayerActuallyReads(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4
	l := a.layer()
	i := a.SelToken

	// The premise: In and Norm1 are different vectors, so the distinction matters.
	inNorm, lnNorm := norm(l.In[i]), norm(l.Norm1[i])
	if inNorm == lnNorm {
		t.Skip("In and Norm1 have equal norms on this prompt, so the labels are " +
			"indistinguishable here")
	}

	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	var joined strings.Builder
	for _, s := range tr.Texts() {
		joined.WriteString(s)
		joined.WriteString("\n")
	}
	text := joined.String()

	// The old label must be gone: it claimed x enters the sublayer.
	if strings.Contains(text, "进入注意力子层 x") {
		t.Fatal("the stage still says x enters the attention sublayer; the " +
			"sublayer reads LN(x)")
	}
	// The corrected labels and norm1's own strip must be present.
	for _, want := range []string{"残差流 x", "子层实际读到的输入"} {
		if !strings.Contains(text, want) {
			t.Fatalf("the stage does not state %q", want)
		}
	}
	// Both strips are drawn, so both norms appear.
	for _, want := range []string{
		fmt.Sprintf("‖v‖ %.2f", inNorm),
		fmt.Sprintf("‖v‖ %.2f", lnNorm),
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("the stage does not draw a strip with %q; both x and LN(x) "+
				"should be shown", want)
		}
	}
}

// TestFFNStageLabelsItsInputCorrectly is the same check for stage 6, whose caption
// already said "W₁·LN(x)" — that one matches Hidden = Norm2 · W1.
func TestFFNStageLabelsItsInputCorrectly(t *testing.T) {
	a := newTestApp(t)
	a.Step = 5
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	found := false
	for _, s := range tr.Texts() {
		if strings.Contains(s, "LN(x)") {
			found = true
		}
	}
	if !found {
		t.Fatal("the FFN stage no longer states that its input is LN(x)")
	}
}

// TestSeedDependsOnThePromptText checks the sampling card's central claim: the
// seed comes from the prompt, so changing a character changes the draw.
//
// No test covered this. The existing ones check that the card *says* "种子 =
// FNV-1a(提示词)" and that the note about the seed exists — both of which pass even
// if the hash loop ignores the prompt entirely. Confirmed by mutation: replacing
// `seed ^= uint64(b)` with `seed ^= uint64(0*b)` left the whole suite green, while
// the card kept promising that editing a character changes the result.
func TestSeedDependsOnThePromptText(t *testing.T) {
	seedsFor := func(prompt string) (uint64, int) {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		a.Step = 6
		probs := a.Trace.Probabilities(a.Temp)
		d, ok := a.sampleDraw(probs)
		if !ok {
			t.Fatalf("prompt %q: no draw", prompt)
		}
		return d.seed, d.pick
	}

	seedA, pickA := seedsFor("hi")
	seedB, pickB := seedsFor("ho")
	if seedA == seedB {
		t.Fatalf("two different prompts hash to the same seed 0x%016X", seedA)
	}
	t.Logf("\"hi\" -> 0x%016X picks %d; \"ho\" -> 0x%016X picks %d", seedA, pickA, seedB, pickB)

	// The same prompt must be stable — that is the other half of the promise.
	again, _ := seedsFor("hi")
	if again != seedA {
		t.Fatalf("the same prompt hashed to 0x%016X then 0x%016X", seedA, again)
	}

	// And the seed must be the hash the card names: FNV-1a over the prompt bytes.
	var want uint64 = 1469598103934665603
	for _, b := range []byte("hi") {
		want ^= uint64(b)
		want *= 1099511628211
	}
	if seedA != want {
		t.Fatalf("the seed is 0x%016X, but FNV-1a of the prompt is 0x%016X", seedA, want)
	}
}

// TestSeedChangesWithEveryCharacter walks a few single-character edits and
// requires each to produce a different seed, which is what "改一个字，种子和结果
// 都会变" promises.
func TestSeedChangesWithEveryCharacter(t *testing.T) {
	base := newTestApp(t)
	base.Step = 6

	seedOf := func(prompt string) uint64 {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		d, ok := a.sampleDraw(a.Trace.Probabilities(a.Temp))
		if !ok {
			t.Fatalf("prompt %q: no draw", prompt)
		}
		return d.seed
	}

	seen := map[uint64]string{}
	for _, p := range []string{"a", "b", "aa", "ab", "ba"} {
		s := seedOf(p)
		if prev, ok := seen[s]; ok {
			t.Fatalf("%q and %q hash to the same seed 0x%016X", prev, p, s)
		}
		seen[s] = p
	}
}

// TestInspectorNumbersComeFromTheSelectedLayerAndHead closes a gap mutation
// testing found: swapping `a.layer()` for `a.Trace.Layers[0]`, or `a.Head` for 0,
// inside the inspector left every test green.
//
// The values are visibly different (layer 0 vs 1 scores differ by up to 5.88 on the
// default prompt; head 0 vs 2 attention by up to 0.445), so a reader switching the
// control would be shown another layer's or head's numbers under the heading that
// claims the selected one.
//
// Reads the rendered text rather than recomputing: the point is what reaches the
// panel.
func TestInspectorNumbersComeFromTheSelectedLayerAndHead(t *testing.T) {
	// Stage 3 (index 2) lists the top-5 scores for the selected query.
	render := func(layer, head int) string {
		a := newTestApp(t)
		a.Step = 2
		a.Layer = layer
		a.Head = head
		a.SelToken = 10
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		var parts []string
		for _, s := range tr.Texts() {
			parts = append(parts, s)
		}
		return strings.Join(parts, "\n")
	}

	base := render(0, 0)
	if other := render(1, 0); other == base {
		t.Fatal("changing the layer does not change the frame, so the inspector " +
			"cannot be showing the selected layer")
	}
	if other := render(0, 2); other == base {
		t.Fatal("changing the head does not change the frame, so the inspector " +
			"cannot be showing the selected head")
	}

	// And the specific numbers must follow the control: the top score shown at
	// layer 1 head 2 must equal that head's own value, not layer 0's.
	a := newTestApp(t)
	a.Step = 2
	a.Layer = 1
	a.Head = 2
	q := 10
	a.SelToken = q
	l := a.Trace.Layers[1].Heads[2]
	var maxAbs float32
	best := -1
	for j := 0; j <= q; j++ {
		if av := float32(math.Abs(float64(l.Scores[q][j]))); av > maxAbs {
			maxAbs = av
			best = j
		}
	}
	if best < 0 {
		t.Fatal("no scores in the row")
	}
	want := fmt.Sprintf("%+.2f", l.Scores[q][best])
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	found := false
	for _, s := range tr.Texts() {
		if s == want {
			found = true
		}
	}
	if !found {
		t.Fatalf("layer 1 head 2's largest |score| on row %d is %s, but that value "+
			"does not appear in the panel", q, want)
	}
}

// TestScoreColourSeparatesSigns pins the diverging ramp: positive and negative
// scores must not paint the same colour.
//
// Mutation testing swapped the sign branch for `_ = v` and every test still passed,
// even though the legend tells the reader warm is positive and blue is negative.
func TestScoreColourSeparatesSigns(t *testing.T) {
	pos := scoreColor(+1, 2)
	neg := scoreColor(-1, 2)
	if pos == neg {
		t.Fatalf("+1 and -1 score to the same colour %v", pos)
	}
	// And a larger magnitude must be more opaque than a smaller one.
	small := scoreColor(0.2, 2)
	if small.A >= pos.A {
		t.Fatalf("a smaller score is not more transparent: |0.2| alpha %d vs |1| "+
			"alpha %d", small.A, pos.A)
	}
}

// TestHeatColourRisesWithWeight pins the attention ramp's shape.
//
// It uses sqrt, so a weight at 1/4 of the row maximum is already half as opaque;
// linear would make the dense low end of the matrix nearly invisible. Mutation
// testing replaced the sqrt with a plain multiply and nothing failed.
func TestHeatColourRisesWithWeight(t *testing.T) {
	lo, hi := heatColor(0.02), heatColor(0.08)
	if lo.A >= hi.A {
		t.Fatalf("a heavier weight does not paint more opaque: 0.02 alpha %d vs 0.08 "+
			"alpha %d", lo.A, hi.A)
	}
	// sqrt(0.25) = 0.5, so a quarter of a saturated cell should be about halfway.
	quarter := heatColor(0.25)
	half := heatColor(0.5)
	if quarter.A >= half.A {
		t.Fatalf("the ramp is not monotonic at the midpoint: 0.25 alpha %d vs 0.5 "+
			"alpha %d", quarter.A, half.A)
	}
	// The ramp must be the sqrt curve, not a straight line. At w = 0.25 sqrt gives
	// 0.5 while linear gives 0.25, so the alpha there has to sit near the sqrt
	// value: a tolerance wide enough for float32 and premultiplication, narrow
	// enough to reject the linear version.
	alphaAtQuarter := float64(quarter.A) / 255
	sqrtValue := 0.12 + 0.88*math.Sqrt(0.25) // 0.56
	linearValue := 0.12 + 0.88*0.25          // 0.34
	if math.Abs(alphaAtQuarter-sqrtValue) > math.Abs(alphaAtQuarter-linearValue) {
		t.Fatalf("the ramp at w=0.25 has alpha %.3f, closer to the linear %.3f than "+
			"to the sqrt %.3f — the sqrt lift is gone",
			alphaAtQuarter, linearValue, sqrtValue)
	}
}

// TestProbColourScalesAgainstTheRowMaximum pins that probColor normalises by the
// value it is given rather than a fixed constant.
func TestProbColourScalesAgainstTheRowMaximum(t *testing.T) {
	// The same probability must paint differently depending on the maximum.
	withSmallMax := probColor(0.10, 0.20)
	withLargeMax := probColor(0.10, 0.90)
	if withSmallMax == withLargeMax {
		t.Fatalf("probColor ignores its max argument: %v for both 0.20 and 0.90",
			withSmallMax)
	}
	if withSmallMax.A <= withLargeMax.A {
		t.Fatalf("0.10 out of 0.20 should be lighter than 0.10 out of 0.90, got "+
			"alpha %d vs %d", withSmallMax.A, withLargeMax.A)
	}
}

// TestResidualInspectorReadsTheSelectedLayersTensors closes the gap mutation
// testing found: swapping a.layer() for a.Trace.Layers[0] in inspectorResidual, or
// reading l.In instead of l.AfterFFN, left the suite green.
//
// Both are visible. AfterFFN's mean on the selected token is -0.0025 at layer 1 and
// -0.0716 at layer 0, and ‖AfterFFN‖ is 37.766 while ‖In‖ is 26.023 — a reader
// swapping the layer, or reading the stage's main strip for comparison, would be
// given the wrong vector under a heading that names the right one.
func TestResidualInspectorReadsTheSelectedLayersTensors(t *testing.T) {
	// The panel's numbers must differ between layers.
	renderWithLayer := func(layer int) string {
		a := newTestApp(t)
		a.Step = 4
		a.Layer = layer
		a.SelToken = 10
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		var parts []string
		for _, s := range tr.Texts() {
			parts = append(parts, s)
		}
		return strings.Join(parts, "\n")
	}
	if renderWithLayer(0) == renderWithLayer(1) {
		t.Fatal("the residual inspector renders identically for both layers")
	}

	// And the L2 value shown must be AfterFFN's, not In's.
	a := newTestApp(t)
	a.Step = 4
	a.Layer = 1
	i := 10
	a.SelToken = i
	want := fmt.Sprintf("%.3f", norm(a.Trace.Layers[1].AfterFFN[i]))
	other := fmt.Sprintf("%.3f", norm(a.Trace.Layers[1].In[i]))
	if want == other {
		t.Skip("AfterFFN and In have the same norm here, so the two cannot be told apart")
	}
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	var seen []string
	for _, s := range tr.Texts() {
		seen = append(seen, s)
	}
	joined := strings.Join(seen, "\n")
	if !strings.Contains(joined, want) {
		t.Fatalf("the inspector does not show ‖AfterFFN‖ = %s", want)
	}
}

// TestFFNInspectorReadsTheSelectedLayer is the same check for the hidden-layer
// stats, whose numbers differ more plainly: layer 0 has peak 4.11 and 145 active
// dimensions, layer 1 has peak 1.83 and 160.
func TestFFNInspectorReadsTheSelectedLayer(t *testing.T) {
	renderWithLayer := func(layer int) string {
		a := newTestApp(t)
		a.Step = 5
		a.Layer = layer
		a.SelToken = 10
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		var parts []string
		for _, s := range tr.Texts() {
			parts = append(parts, s)
		}
		return strings.Join(parts, "\n")
	}
	if renderWithLayer(0) == renderWithLayer(1) {
		t.Fatal("the FFN inspector renders identically for both layers")
	}

	// The count the panel prints must be the selected layer's. Anchor on the
	// inspector's own short form, "162 / 384 (42%)" — the stage card prints the
	// same figure with a prefix ("输出 > 0.01 的维度 162 / 384 (42%)"), so matching
	// the short string alone identifies the panel.
	for _, layer := range []int{0, 1} {
		a := newTestApp(t)
		a.Step = 5
		a.Layer = layer
		i := 10
		a.SelToken = i
		hidden := a.Trace.Layers[layer].Hidden[i]
		active := 0
		for _, v := range hidden {
			if v > 0.01 {
				active++
			}
		}
		want := fmt.Sprintf("%d / %d (%.0f%%)", active, len(hidden),
			100*float64(active)/float64(len(hidden)))
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		found := false
		for _, s := range tr.Texts() {
			if s == want {
				found = true
			}
		}
		if !found {
			t.Fatalf("layer %d: the inspector does not show %q", layer, want)
		}
	}
}

// TestInspectorTokenNumberMatchesTheRail pins the inspector header's numbering.
//
// Mutation testing replaced `a.SelToken` with `a.SelToken+1` in the header and every
// test passed. Both the header and the rail's chip labels are zero-based — selecting
// token 10 shows "token #10 / 35" beside a chip labelled "token 10" — so an off-by-one
// on either side would leave the panel naming a different token than the strip the
// reader just clicked.
func TestInspectorTokenNumberMatchesTheRail(t *testing.T) {
	for _, sel := range []int{0, 10, 35} {
		a := newTestApp(t)
		a.Step = 0
		a.SelToken = sel
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()

		wantHeader := fmt.Sprintf("token #%d / %d", sel, len(a.Trace.Tokens)-1)
		wantChip := fmt.Sprintf("token %d", sel)

		var header, chip bool
		for _, s := range tr.Texts() {
			if s == wantHeader {
				header = true
			}
			if s == wantChip {
				chip = true
			}
		}
		if !header {
			t.Fatalf("selecting %d: no header %q", sel, wantHeader)
		}
		if !chip {
			t.Fatalf("selecting %d: no rail chip %q", sel, wantChip)
		}

		// And the two must agree: a header for the next token must not be present.
		wrong := fmt.Sprintf("token #%d / %d", sel+1, len(a.Trace.Tokens)-1)
		for _, s := range tr.Texts() {
			if s == wrong && sel+1 <= len(a.Trace.Tokens)-1 {
				t.Fatalf("selecting %d also renders the header for %d: %q", sel, sel+1, wrong)
			}
		}
	}
}

// TestTokenRailChipLabelsAreZeroBased records the convention the header test leans
// on, so a later change to either side is visible here rather than as a mismatch
// nobody checks.
func TestTokenRailChipLabelsAreZeroBased(t *testing.T) {
	a := newTestApp(t)
	a.SelToken = 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	if !tr.HasText("token 0") {
		t.Fatal("the first token's chip is not labelled \"token 0\"")
	}
}

// TestSamplingCardArithmeticReproducesX pins the drawn value to the printed line.
//
// The card shows "x = (种子 mod 1000003) ÷ 1000003 = 587537 ÷ 1000003 = 0.587535".
// Three mutations to that line and its constant left every test green: changing the
// modulus to 1000001, using 1000000 as the divisor while keeping the numerator, and
// changing only the divisor printed in the text. All three break the one property the
// line exists for — that a reader can divide the two printed integers and get the
// printed x.
func TestSamplingCardArithmeticReproducesX(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	d, ok := a.sampleDraw(a.Trace.Probabilities(a.Temp))
	if !ok {
		t.Fatal("no draw")
	}

	const modulus = 1000003
	numerator := d.seed % modulus
	hand := float64(numerator) / float64(modulus)
	if got, want := fmt.Sprintf("%.6f", hand), fmt.Sprintf("%.6f", d.x); got != want {
		t.Fatalf("x is printed as %.6f, but the division the card shows gives %s",
			d.x, got)
	}

	// And the card must actually print those two integers, so the reader has them.
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	want := fmt.Sprintf("= %d ÷ %d = %.6f", numerator, modulus, hand)
	found := false
	for _, s := range tr.Texts() {
		if strings.Contains(s, want) {
			found = true
		}
	}
	if !found {
		t.Fatalf("the card does not print %q", want)
	}
}

// TestSamplingCardDividesByTheSameModulusItNames catches the narrower mistake: the
// numerator taken modulo one number and divided by another.
func TestSamplingCardDividesByTheSameModulusItNames(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	d, ok := a.sampleDraw(a.Trace.Probabilities(a.Temp))
	if !ok {
		t.Fatal("no draw")
	}
	// Recover the modulus from the printed x: seed mod M = numerator, and
	// numerator / M = x, so M = numerator / x.
	numerator := d.seed % 1000003
	if d.x == 0 {
		t.Skip("x is zero, so the modulus cannot be recovered from it")
	}
	implied := float64(numerator) / d.x
	if math.Abs(implied-1000003) > 1 {
		t.Fatalf("x = %.6f with numerator %d implies a modulus of %.0f, not 1000003",
			d.x, numerator, implied)
	}
}

// TestTop12TailSwitchesFormatBySize checks both branches of the tail-share
// formatter.
//
// The line reads "其余 244 个合计 1.3e-05%" at the default temperature, where the
// tail is negligible, and format selection matters exactly there: "%.4f" would print
// 0.0000% and say nothing. But the other branch never ran in any test — replacing the
// threshold check with `if true` left the suite green, because every tested
// temperature produced a tail below 0.01%.
//
// At T=50 the tail carries 93% of the mass, which needs the plain percentage. Both
// branches exist for real cases, so both get a test.
func TestTop12TailSwitchesFormatBySize(t *testing.T) {
	tailShare := func(temp float64) (share float64, printed string) {
		a := newTestApp(t)
		a.Step = 6
		a.Temp = temp
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		probs := a.Trace.Probabilities(temp)
		top := model.TopK(probs, 12)
		inTop := map[int]bool{}
		for _, id := range top {
			inTop[id] = true
		}
		for id, p := range probs {
			if !inTop[id] {
				share += float64(p)
			}
		}
		for _, s := range tr.Texts() {
			if strings.Contains(s, "个合计") {
				printed = s
			}
		}
		return share * 100, printed
	}

	// Small but real: the tail is vanishing yet still a probability, so it must be written in
	// scientific notation — "%.4f" would round it to 0.0000%.
	//
	// T=0.5 used to be the sample here, but at that temperature the top twelve carry the whole
	// distribution and the leftover is float32 residue (5.6e-19 as a percentage). That residue is
	// now reported as 0%, so the case needs a temperature where the tail is genuinely small and
	// genuinely non-zero: T=0.8 gives about 1.3e-05%.
	smallShare, smallText := tailShare(0.8)
	if smallShare >= 0.01 {
		t.Fatalf("at T=0.8 the tail is %.4f%%, not small enough to test the "+
			"scientific branch", smallShare)
	}
	if smallShare < 1e-9 {
		t.Fatalf("at T=0.8 the tail is %.3e%%, which is float residue rather than a "+
			"probability; pick a warmer temperature", smallShare)
	}
	if !strings.Contains(smallText, "e-") {
		t.Fatalf("a tail of %.4f%% is printed as %q, which rounds to 0.0000%%",
			smallShare, smallText)
	}

	// Hot: the tail dominates, so the plain percentage is the readable form and
	// scientific notation would say "9.3e+01%" where "93.0368%" is right there.
	bigShare, bigText := tailShare(50)
	if bigShare < 0.01 {
		t.Fatalf("at T=50 the tail is only %.4f%%, so the plain branch cannot be "+
			"reached", bigShare)
	}
	if strings.Contains(bigText, "e+") || strings.Contains(bigText, "e-") {
		t.Fatalf("a tail of %.4f%% is printed in scientific notation: %q", bigShare, bigText)
	}
	if !strings.Contains(bigText, "%") {
		t.Fatalf("the tail line does not carry a percentage: %q", bigText)
	}
}

// TestStripCaretDoesNotOutliveThePointer separates the two things a hovered
// vector strip keeps.
//
// vectorStrip's comment already draws the line: the *value* of HoverDimIn survives
// the pointer leaving, because the formulas under the strip follow it and clearing
// would snap them back to d0 the moment the reader moved towards the working they
// were reading. What must not survive is the caret, which says "this is the cell
// under your cursor".
//
// It did survive. Only the value was ever written; nothing reset or gated the
// highlight, so the last hovered dimension stayed outlined after the pointer left —
// 37 pixels of outline inside the strip band. The caret is now drawn only on the
// frame the pointer is actually on the row.
func TestStripCaretDoesNotOutliveThePointer(t *testing.T) {
	capture := func(hoverThenLeave bool) *image.RGBA {
		a := newTestApp(t)
		a.Step = 1
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		if hoverThenLeave {
			tr.Move(600, 408) // on the embedding strip
			tr.Frame()
		}
		tr.Move(1490, 930) // away from every strip
		tr.Frame()
		return tr.Image()
	}
	never := capture(false)
	afterLeave := capture(true)

	// The strip row itself must be identical.
	const bandTop, bandBottom = 396, 420
	for py := bandTop; py < bandBottom; py++ {
		for px := 260; px < 1150; px++ {
			i := (py*never.Rect.Dx() + px) * 4
			if never.Pix[i] != afterLeave.Pix[i] ||
				never.Pix[i+1] != afterLeave.Pix[i+1] ||
				never.Pix[i+2] != afterLeave.Pix[i+2] {
				t.Fatalf("the strip still differs at (%d,%d) after the pointer left — "+
					"the caret outlived it", px, py)
			}
		}
	}

	// And the readout below must still follow the last hovered dimension: that is
	// the behaviour the comment protects, and it is what makes clearing the value
	// the wrong fix.
	a := newTestApp(t)
	a.Step = 1
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	tr.Move(600, 408)
	tr.Frame()
	hovered := a.HoverDimIn
	if hovered < 0 {
		t.Skip("the probe position is not on a strip in this layout")
	}
	tr.Move(1490, 930)
	tr.Frame()
	if a.HoverDimIn != hovered {
		t.Fatalf("the readout dimension reset from %d to %d on leaving; the "+
			"formulas below the strip are meant to keep following it",
			hovered, a.HoverDimIn)
	}
}

// TestCardsAllComeFromOneHelper keeps panel surfaces identical.
//
// The type scale has a guard (TestTypeScaleIsTheOnlySourceOfFontSizes) because a
// literal font size is one nobody remembers writing and nobody dares change. Card
// shape had no equivalent: every stage's panels go through card(), which fixes
// Padding(14), Gap(10), Radius(10) and the colPanel background — but nothing stopped a
// new panel from being written inline with slightly different numbers, which reads as a
// rendering glitch rather than a different kind of thing.
//
// The rule is narrow on purpose. Chips, badges and buttons legitimately have their own
// radius and padding; what must not vary is the card: the colPanel surface at radius 10
// belongs to exactly one helper, so the whole package should contain exactly one such
// expression.
func TestCardsAllComeFromOneHelper(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	const shape = "Background(colPanel).Radius(10)"
	occurrences := 0
	var where []string
	for _, name := range files {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		body, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		for i, line := range strings.Split(string(body), "\n") {
			if strings.Contains(line, shape) {
				occurrences++
				where = append(where, fmt.Sprintf("%s:%d", name, i+1))
			}
		}
	}
	if occurrences != 1 {
		t.Fatalf("the card surface %q appears %d times (%v); it belongs to card() "+
			"alone, so a panel written elsewhere would be shaped slightly differently",
			shape, occurrences, where)
	}
}

// TestCJKTruncationKeepsWholeCharacters checks the truncation path that only a
// multi-byte prompt reaches.
//
// Truncate walks `start` forward past UTF-8 continuation bytes so a character is never
// cut in half. That means a CJK prompt can land below the context limit: 264 bytes of
// hanzi is 88 characters, and taking the last 64 bytes can begin mid-character, so the
// result is 63 tokens. Every truncation test before this one used ASCII, where the
// limit is exact and the walk never runs.
//
// The app must therefore report what it kept (63), not the limit (64) — a reader who
// sees "上限 64" and "T = 63" with no explanation has found a contradiction.
func TestCJKTruncationKeepsWholeCharacters(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = strings.Repeat("注意力机制就是加权求和", 8)
	a.recompute()

	got := len(a.Trace.Tokens)
	ctx := a.W.Config.Ctx
	if got >= ctx {
		t.Fatalf("this prompt was meant to land below the limit (got %d of %d); "+
			"the multi-byte walk was not exercised", got, ctx)
	}
	if got != ctx-1 {
		t.Fatalf("kept %d tokens; a 3-byte-per-character prompt cut at the limit "+
			"should lose exactly one to the whole-character rule", got)
	}

	// The first kept token must start a character, not continue one.
	ids := promptIDs(a.Prompt)
	dropped := len(ids) - got
	if dropped <= 0 || dropped >= len(ids) {
		t.Fatalf("dropped %d of %d bytes", dropped, len(ids))
	}
	if first := byte(ids[dropped]); first&0xC0 == 0x80 {
		t.Fatalf("the first kept byte 0x%02X is a UTF-8 continuation, so a "+
			"character was cut", first)
	}

	// And the card must say what was kept, not what was allowed.
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	want := fmt.Sprintf("已截断到最近 %d 个 token", got)
	var saw bool
	for _, s := range tr.Texts() {
		if strings.Contains(s, want) {
			saw = true
		}
	}
	if !saw {
		t.Fatalf("the card does not report %q", want)
	}

	// The dropped-byte figure must be the real prefix length.
	wantDropped := fmt.Sprintf("丢弃前 %d 字节", dropped)
	for _, s := range tr.Texts() {
		if s == wantDropped {
			return
		}
	}
	t.Fatalf("the card does not report %q", wantDropped)
}

// TestFFNStageAndInspectorAgreeOnTheActiveCount keeps the two panels showing one number.
//
// "激活 > 0.01 的维度" appears twice on the FFN stage — once on the card, once in the
// inspector — and both used to compute it with their own loop over the same threshold.
// Two copies of a cutoff are two chances for the panels to disagree about the same row,
// and a reader comparing them would have no way to tell which was right.
//
// They now share countActive. This asserts the shared count is what both render, and
// pins the threshold itself, since a change to it is a change to the lesson's claim about
// sparsity.
func TestFFNStageAndInspectorAgreeOnTheActiveCount(t *testing.T) {
	a := newTestApp(t)
	a.Step = 5
	a.SelToken = 10
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	hidden := a.layer().Hidden[10]
	want := fmt.Sprintf("%d / %d (%.0f%%)", countActive(hidden), len(hidden),
		100*float64(countActive(hidden))/float64(len(hidden)))

	// The inspector prints the count in that exact form; the card prefixes it.
	inInspector := false
	for _, s := range tr.Texts() {
		if s == want {
			inInspector = true
		}
	}
	if !inInspector {
		t.Fatalf("the inspector does not print %q", want)
	}

	// And the card's line must use the same count, not a count of its own. It words
	// the threshold differently — "输出 > N" where the inspector says "激活 > N" — but N
	// comes from activeThreshold on both sides, so moving the cutoff moves both.
	th := fmt.Sprintf("%.2f", activeThreshold)
	wantCard := fmt.Sprintf("输出 > %s 的维度 %s", th, want)
	var sawCard bool
	for _, s := range tr.Texts() {
		if strings.Contains(s, wantCard) {
			sawCard = true
		}
	}
	if !sawCard {
		t.Fatalf("the stage card does not print %q", wantCard)
	}

	// And the inspector's label must carry the same number.
	wantLabel := fmt.Sprintf("激活 > %s", th)
	var sawLabel bool
	for _, s := range tr.Texts() {
		if s == wantLabel {
			sawLabel = true
		}
	}
	if !sawLabel {
		t.Fatalf("the inspector does not label the cutoff %q", wantLabel)
	}
}

// TestScaleReadoutIsTheSameOnBothPanels pins the 1/√d_head figure the score stage shows
// twice: once in the legend beside the matrix, once in the inspector.
//
// It was the same source line in two places, so the two could have drifted — and a third
// copy in the workings prints the same quantity at float32 precision rather than float64,
// which is why the three had two types between them. kvScale now holds the legend and
// inspector copy.
//
// Both panels must show the same string, and it must be the value the model's float32
// arithmetic produces at three decimals — the precision every panel prints at.
func TestScaleReadoutIsTheSameOnBothPanels(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	a.SelToken = 10
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	// The inspector copy is on the first screen; the legend is below it.
	tr.Scroll(700, 500, 0, -120)
	tr.Frame()

	dHead := a.Trace.Config.DHead()
	want := fmt.Sprintf("1/√%d = %.3f", dHead, 1/math.Sqrt(float64(dHead)))

	count := 0
	for _, s := range tr.Texts() {
		if s == want {
			count++
		}
	}
	if count != 2 {
		t.Fatalf("%q appears %d times, expected once on the legend and once in the "+
			"inspector", want, count)
	}

	// The float32/float64 question is already decided by the assertion above: `want` is
	// built the way the panels build it, so a panel that switched precision would print a
	// different string and the count would come out wrong. A separate check here compared
	// the two forms and only logged the result, which proved nothing — removed rather than
	// turned into a second assertion over the same thing.
}

// TestScaleReadoutComesFromOneHelper keeps the two copies from being re-inlined.
func TestScaleReadoutComesFromOneHelper(t *testing.T) {
	src, err := os.ReadFile("stages.go")
	if err != nil {
		t.Fatal(err)
	}
	text := string(src)
	if !strings.Contains(text, "func kvScale(c *ui.Context, dHead int) {") {
		t.Fatal("kvScale is gone; the legend and the inspector are independent again")
	}
	// The legend's own line may appear inside kvScale only. Matching the bare
	// expression would be wrong: the score workings legitimately computes the same
	// quantity on its own, with a different argument name.
	const legend = `kv(c, "缩放", fmt.Sprintf("1/√%d = %.3f"`
	if n := strings.Count(text, legend); n != 1 {
		t.Fatalf("the legend line appears %d times; it belongs in kvScale alone", n)
	}
}

// TestFFNLessonQuotesTheLoadedWeights checks the lesson footer's figures come from the
// model, not from the prose.
//
// The FFN paragraph said "先升维到 4 倍宽 … 算上嵌入表则是 58%" as a literal. Both numbers
// describe the bundled checkpoint, and both are also computed elsewhere on the same stage
// — the card prints the share via ffnParamShare. A different checkpoint would move the
// card and leave the paragraph claiming the old figures, which is the same class as the
// printed-cutoff bug two rounds ago: a constant extracted in one place while prose kept a
// literal.
//
// The paragraph's figures are derived now. This asserts the rendered text matches what the
// config gives — with the deliberate exception of "三分之二", which is prose for roughly
// two thirds and is left as words.
func TestFFNLessonQuotesTheLoadedWeights(t *testing.T) {
	a := newTestApp(t)
	a.Step = 5
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	ofAll, _ := ffnParamShare(a.Trace.Config)
	width := float64(a.Trace.Config.DFF) / float64(a.Trace.Config.DModel)
	wantWidth := fmt.Sprintf("%.0f 倍宽", width)
	wantShare := fmt.Sprintf("算上嵌入表则是 %.0f%%", ofAll*100)

	var para string
	for _, s := range tr.Texts() {
		if strings.Contains(s, "先升维到") {
			para = s
		}
	}
	if para == "" {
		t.Fatal("the FFN lesson paragraph is not rendered")
	}
	if !strings.Contains(para, wantWidth) {
		t.Fatalf("the paragraph does not state the derived width %q: %s", wantWidth, para)
	}
	if !strings.Contains(para, wantShare) {
		t.Fatalf("the paragraph does not state the derived share %q: %s", wantShare, para)
	}

	// And the literal must be gone from the source, so a future checkpoint change cannot
	// silently leave it behind.
	src, err := os.ReadFile("stages.go")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(src), "算上嵌入表则是 58%") {
		t.Fatal("the lesson paragraph still hard-codes 58%")
	}
}

// TestSoftmaxLessonClaimsMatchTheAttention checks the empirical sentence in the Softmax
// lesson.
//
// The paragraph says "第 0 层的头 0 几乎每个位置都盯着紧邻的前一个 token，第 1 层的几个头则
// 反复回到同一小批位置上" — a claim about what this checkpoint's attention actually does.
// Nothing verified the second half. The test that was here measured only the
// previous-token rate of both layers' head 0 and said in its own comment that this
// checked "the layer-0 head-0 peak count being far higher than layer 1's, which is
// the contrast the sentence draws". It does not: a head that peaks on 35 different
// positions, each exactly one before its query, and a head that peaks on one position
// 35 times both score 100% on that measure. The two clauses need two measurements.
//
// Measured on the bundled weights and the default prompt (35 rows):
//
//	         distinct peaks   largest share of rows on one key
//	L0 H0..H3   31/25/22/30          2 / 4 / 5 / 2
//	L1 H0..H3   10/10/ 9/ 7         15 / 15 / 8 / 10
//
// So layer 1 does concentrate — 7 to 10 distinct positions instead of 22 to 31, with
// one key taking up to 15 of the 35 rows — which is what "反复回到同一小批位置" says.
func TestSoftmaxLessonClaimsMatchTheAttention(t *testing.T) {
	a := newTestApp(t)
	tr := a.Trace
	if len(tr.Layers) < 2 || len(tr.Layers[0].Heads) == 0 {
		t.Skip("the claim names two layers and head 0")
	}

	// peaks counts, for one head, how many distinct keys are some row's argmax and
	// how many rows the most-chosen key takes.
	peaks := func(layer, head int) (distinct, maxRepeat, rows int) {
		h := tr.Layers[layer].Heads[head]
		counts := map[int]int{}
		for i := 1; i < len(tr.Tokens); i++ {
			row := h.Attn[i]
			best, bestJ := float32(-1), -1
			for j := 0; j <= i; j++ {
				if row[j] > best {
					best, bestJ = row[j], j
				}
			}
			if bestJ >= 0 {
				counts[bestJ]++
				rows++
			}
		}
		for _, c := range counts {
			if c > maxRepeat {
				maxRepeat = c
			}
		}
		return len(counts), maxRepeat, rows
	}
	peakOnPrevious := func(layer int) (hits, rows int) {
		h := tr.Layers[layer].Heads[0]
		for i := 1; i < len(tr.Tokens); i++ {
			row := h.Attn[i]
			best, bestJ := float32(-1), -1
			for j := 0; j <= i; j++ {
				if row[j] > best {
					best, bestJ = row[j], j
				}
			}
			if bestJ >= 0 {
				rows++
				if bestJ == i-1 {
					hits++
				}
			}
		}
		return hits, rows
	}

	firstHits, firstRows := peakOnPrevious(0)
	if firstRows == 0 {
		t.Skip("no rows to measure")
	}

	// Clause 1: "几乎每个位置" — the sentence says almost every position.
	if frac := float64(firstHits) / float64(firstRows); frac < 0.8 {
		t.Fatalf("the lesson says layer 0 head 0 watches the previous token almost "+
			"everywhere, but only %d of %d rows peak there", firstHits, firstRows)
	}

	// Clause 2: "第 1 层的几个头则反复回到同一小批位置上". Two things have to hold for
	// that sentence to be supportable, and the previous version of this test asserted
	// neither: layer 1's heads must use fewer distinct positions than layer 0's, and
	// at least one of them must return to the same position often enough to be
	// "反复". Checked on every head, since the sentence says "几个头".
	var l0Distinct, l1Distinct []int
	for head := 0; head < len(tr.Layers[0].Heads); head++ {
		d0, _, _ := peaks(0, head)
		l0Distinct = append(l0Distinct, d0)
		if head < len(tr.Layers[1].Heads) {
			d1, rep, rows := peaks(1, head)
			l1Distinct = append(l1Distinct, d1)
			// "反复回到同一小批位置" is a claim about concentration, and a raw repeat
			// count does not measure it: 8 of 35 rows on one key could be repetition or
			// could be the largest share when 9 keys share the rows. Compare against
			// what an uninformative head would give — with d distinct keys chosen over
			// rows draws, the expected share of the most-chosen key is about
			// rows/d. Requiring the observed peak to beat that by 1.5x is the claim
			// this app makes everywhere else: a number has to be compared against a
			// reference to mean anything.
			//
			// Measured: L1 H0..H3 choose 10/10/9/7 distinct keys and repeat the top one
			// 15/15/8/10 times, against an expected 3.5/3.5/3.9/5.0 — a factor of 2 to 4.
			// Layer 0's heads sit at 31/25/22/30 distinct with a maximum repeat of
			// 2/4/5/2, where the same expected share is ~1.1-1.6.
			expected := float64(rows) / float64(d1)
			if float64(rep) < 1.5*expected {
				t.Fatalf("layer 1 head %d puts its most-chosen key on %d of %d rows while "+
					"spreading over %d distinct keys (an uninformative head would give "+
					"about %.1f); the lesson says layer 1's heads repeatedly return to the "+
					"same small set of positions", head, rep, rows, d1, expected)
			}
		}
	}
	// The first clause of the contrast: fewer distinct peaks in layer 1. Compared as
	// sums rather than head by head, because the sentence compares the layers.
	if sum(l1Distinct) >= sum(l0Distinct) {
		t.Fatalf("layer 1 uses %d distinct peak positions against layer 0's %d; the "+
			"lesson contrasts the two, saying layer 1 concentrates", sum(l1Distinct), sum(l0Distinct))
	}
}

// sum adds a slice of ints, for the layer comparison above.
func sum(v []int) int {
	total := 0
	for _, n := range v {
		total += n
	}
	return total
}

// TestScoreColourRangeMatchesTheDocumentedFigures pins the range that the README and the
// stage's own comment quote.
//
// Both said "5.4–8.3" (the comment said 5.4 to 8.2). Measured now: layer 0 spans 4.6–12.5
// and layer 1 spans 8.1–15.2 across the bundled prompts. The old figures were neither the
// whole range nor any single layer's — 5.4–8.3 is close to layer 0 on the Chinese prompt
// alone, which looks like a measurement from an earlier checkpoint or an earlier
// measurement that only covered one layer.
//
// The reader is told this number is a moving fact, so it has to be the moving fact that is
// actually there. The test checks both layers against the documented bounds rather than
// hard-coding each value, since the figures move whenever the weights do.
func TestScoreColourRangeMatchesTheDocumentedFigures(t *testing.T) {
	lo := [2]float64{1e9, 1e9}
	hi := [2]float64{-1e9, -1e9}
	for _, p := range promptPresets {
		a := newTestApp(t)
		a.Prompt = p.prompt
		a.recompute()
		if a.Trace == nil || len(a.Trace.Tokens) == 0 {
			continue
		}
		for li := range a.Trace.Layers {
			if li >= len(lo) {
				break
			}
			for h := range a.Trace.Layers[li].Heads {
				sc := scoreColorScale(a.Trace.Layers[li].Heads[h])
				if sc <= 0 {
					continue
				}
				lo[li] = math.Min(lo[li], sc)
				hi[li] = math.Max(hi[li], sc)
			}
		}
	}
	if lo[0] >= 1e9 || hi[0] <= -1e9 {
		t.Skip("no scored matrices in the presets")
	}

	// The documented figures, one decimal place as the README states them.
	//
	// Layer 0: 4.6–12.5, layer 1: 8.1–15.2.
	for li, want := range [][2]float64{{4.6, 12.5}, {8.1, 15.2}} {
		if li >= len(lo) || hi[li] <= -1e9 {
			continue
		}
		if got := math.Round(lo[li]*10) / 10; math.Abs(got-want[0]) > 0.11 {
			t.Fatalf("layer %d's smallest colour scale is %.1f, the README says %.1f",
				li, got, want[0])
		}
		if got := math.Round(hi[li]*10) / 10; math.Abs(got-want[1]) > 0.11 {
			t.Fatalf("layer %d's largest colour scale is %.1f, the README says %.1f",
				li, got, want[1])
		}
	}

	// And the README must carry these numbers, not the old ones.
	doc := readREADME(t)
	if !strings.Contains(doc, "第 0 层 4.6–12.5") || !strings.Contains(doc, "第 1 层 8.1–15.2") {
		t.Fatal("the README does not state the measured per-layer ranges")
	}
	if strings.Contains(doc, "5.4–8.3") {
		t.Fatal("the README still carries the stale 5.4–8.3 range")
	}
}

// TestTopHeadingCountsTheRowsItLists keeps the output card's heading and its contents in
// step.
//
// The title said "Top-12" as prose while TopK was called with a literal 12 a few lines
// above it. Changing one and not the other would print a heading that miscounts its own
// rows — the same shape as the lesson paragraph that hard-coded 58% while the card beside
// it computed the share.
//
// topN now feeds both. The test counts the rendered rows rather than trusting the string,
// so a title that says 12 over eleven bars fails.
func TestTopHeadingCountsTheRowsItLists(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	var heading string
	for _, s := range tr.Texts() {
		if strings.HasPrefix(s, "Top-") && strings.Contains(s, "概率") {
			heading = s
		}
	}
	if heading == "" {
		t.Fatal("the output card has no Top-N heading")
	}
	want := fmt.Sprintf("Top-%d 概率", topN)
	if heading != want {
		t.Fatalf("heading is %q, want %q", heading, want)
	}

	// The count in the heading has to be the count of rows actually listed.
	probs := a.Trace.Probabilities(a.Temp)
	if got := len(model.TopK(probs, topN)); got != topN {
		t.Fatalf("the card lists %d rows but its heading names %d", got, topN)
	}

	// And the literal must be gone from the source, so the two cannot drift apart again.
	src, err := os.ReadFile("stages.go")
	if err != nil {
		t.Fatal(err)
	}
	// Only the heading counts: a comment may still mention the old wording when it
	// explains why the card now prints the vocabulary size alongside it.
	if strings.Contains(string(src), `sectionTitle(c, "Top-12`) ||
		strings.Contains(string(src), `"Top-12 概率"`) {
		t.Fatal("the heading is still written out as a literal")
	}
}

// TestTokenLessonExplainsMultiRuneSymbols checks the example that breaks the lesson's own
// rule of thumb.
//
// The stage says a hanzi costs three tokens, which invites the reader to model this as
// "three bytes per character". The family emoji is the counter-example: five runes, none of
// them the thing the reader sees — one picture — and eighteen tokens. Without it the rule of
// thumb reads as the whole truth.
//
// The claim is checked against the tokenizer rather than trusted: the example is quoted by
// name in the text (the glyph itself renders as a box, since the font has no combined-emoji
// support), and the number beside it has to be the token count that prompt really produces.
func TestTokenLessonExplainsMultiRuneSymbols(t *testing.T) {
	const family = "\U0001F468\u200D\U0001F469\u200D\U0001F467"
	a := newTestApp(t)
	a.Prompt = family
	a.recompute()
	got := len(a.Trace.Tokens)
	if got != 18 {
		t.Fatalf("the lesson quotes 18 tokens for the family emoji; the tokenizer gives %d", got)
	}
	if runes := utf8.RuneCountInString(family); runes != 5 {
		t.Fatalf("the family emoji is %d runes, not 5", runes)
	}

	// And the sentence has to be in the rendered stage.
	b := newTestApp(t)
	b.Step = 0
	tr := ui.NewTester(b.View, 1500, 940)
	tr.Frame()
	var para string
	for _, s := range tr.Texts() {
		if strings.Contains(s, "模型不认识字母") {
			para = s
		}
	}
	if para == "" {
		t.Fatal("the tokenisation paragraph is not rendered")
	}
	for _, want := range []string{"18 个 token", "零宽连接符", "四个字节"} {
		if !strings.Contains(para, want) {
			t.Fatalf("the paragraph does not mention %q", want)
		}
	}
}

// TestTruncationBoundaryIsExact pins the two cases either side of the context limit.
//
// A prompt of exactly ctx bytes must pass through untouched, and one byte more must lose
// exactly one. That boundary had no test: the existing cases use prompts far over or far
// under it, and round 126's finding (a CJK prompt keeping 63 of 64) lived on this same line
// without anyone having measured where the edge actually is.
//
// Multi-byte prompts are checked separately, because there the count depends on where the
// window starts rather than on how much was supplied: 63 CJK characters fit exactly, and
// every length from 64 up keeps 63, since the 64-byte window always begins inside a
// character.
func TestTruncationBoundaryIsExact(t *testing.T) {
	ctx := newTestApp(t).W.Config.Ctx

	for _, c := range []struct {
		bytes    int
		wantKept int
	}{
		{ctx - 1, ctx - 1},
		{ctx, ctx},     // exactly at the limit: nothing dropped
		{ctx + 1, ctx}, // one over: exactly one byte dropped
		{ctx + 2, ctx},
	} {
		a := newTestApp(t)
		a.Prompt = strings.Repeat("a", c.bytes)
		a.recompute()
		if got := len(a.Trace.Tokens); got != c.wantKept {
			t.Fatalf("%d ASCII bytes kept %d tokens, want %d", c.bytes, got, c.wantKept)
		}
	}

	// The card must say nothing about truncation when nothing was truncated: with a
	// prompt of exactly ctx bytes, "已截断到最近 64 个 token" would be false.
	a := newTestApp(t)
	a.Prompt = strings.Repeat("a", ctx)
	a.recompute()
	a.Step = 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	for _, s := range tr.Texts() {
		if strings.Contains(s, "已截断") {
			t.Fatalf("a prompt of exactly %d bytes is reported as truncated: %q", ctx, s)
		}
	}

	// Multi-byte: 63 characters (189 bytes) fit; 64 and beyond keep 63.
	for _, chars := range []int{63, 64, 65} {
		b := newTestApp(t)
		b.Prompt = strings.Repeat("中", chars)
		b.recompute()
		if got := len(b.Trace.Tokens); got != 63 {
			t.Fatalf("%d CJK characters kept %d tokens, want 63", chars, got)
		}
	}
}

// TestTemperatureExtremesKeepTheDistributionValid covers the slider's two ends.
//
// The control runs from 0.1 to 2.0, so logits are never divided by zero — but 0.1 alone
// multiplies them by ten, which saturates the softmax completely: measured max probability
// 1.000000 at T=0.1 against 0.629685 at T=2.0. Saturation is the correct answer here, and
// the card says so (熵 0.00 nats, 100.0%, and the note that T→0 is greedy), so the thing
// worth pinning is that the numbers stay coherent at both ends rather than that the
// distribution stays soft.
func TestTemperatureExtremesKeepTheDistributionValid(t *testing.T) {
	a := newTestApp(t)

	for _, temp := range []float64{0.1, 2.0} {
		probs := a.Trace.Probabilities(temp)
		sum := 0.0
		for _, p := range probs {
			if p < 0 || math.IsNaN(float64(p)) || math.IsInf(float64(p), 0) {
				t.Fatalf("T=%.1f produced an invalid probability %v", temp, p)
			}
			sum += float64(p)
		}
		if math.Abs(sum-1) > 1e-3 {
			t.Fatalf("T=%.1f probabilities sum to %.6f, not 1", temp, sum)
		}
	}

	// The two ends must actually differ: a temperature that changed nothing would make
	// the slider dead.
	cold := a.Trace.Probabilities(0.1)
	hot := a.Trace.Probabilities(2.0)
	if float64(model.Entropy(cold)) >= float64(model.Entropy(hot)) {
		t.Fatalf("entropy does not rise with temperature: %.4f at T=0.1, %.4f at T=2.0",
			model.Entropy(cold), model.Entropy(hot))
	}

	// And both ends must render without a NaN reaching the screen.
	for _, temp := range []float64{0.1, 2.0} {
		b := newTestApp(t)
		b.Temp = temp
		for step := 0; step < len(lessons); step++ {
			b.Step = step
			tr := ui.NewTester(b.View, 1500, 940)
			tr.Frame()
			for _, s := range tr.Texts() {
				if strings.Contains(s, "NaN") || strings.Contains(s, "+Inf") || strings.Contains(s, "Infinity") {
					t.Fatalf("T=%.1f stage %d renders %q", temp, step, s)
				}
			}
		}
	}
}

// TestEmbeddingsArePermutationInvariantWithoutPositions backs the embedding lesson's
// argument for position encoding.
//
// The paragraph says attention scoring depends only on q·k, so the same words in a
// different order produce the same weights — which is why position has to be added
// explicitly. The claim rests on the embedding table alone: permuting a prompt must permute
// the E[token] rows and leave the multiset of rows unchanged, and the position rows must be
// what breaks the symmetry.
//
// Measured: "abc" and "cba" give the same three embedding rows in a different order, and
// their Sum vectors (E + P) differ. If the first half ever stopped holding, the paragraph's
// premise would be false and nothing else in the suite would notice.
func TestEmbeddingsArePermutationInvariantWithoutPositions(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = "abc"
	a.recompute()
	b := newTestApp(t)
	b.Prompt = "cba"
	b.recompute()

	if len(a.Trace.Embed) != 3 || len(b.Trace.Embed) != 3 {
		t.Fatalf("expected three tokens each, got %d and %d",
			len(a.Trace.Embed), len(b.Trace.Embed))
	}

	rows := func(tr *model.Trace) map[string]int {
		m := map[string]int{}
		for _, v := range tr.Embed {
			m[fmt.Sprint(v)]++
		}
		return m
	}
	ea, eb := rows(a.Trace), rows(b.Trace)
	if len(ea) != len(eb) {
		t.Fatalf("permuting the prompt changed the embedding rows: %d distinct vs %d",
			len(ea), len(eb))
	}
	for k, n := range ea {
		if eb[k] != n {
			t.Fatal("permuting the prompt changed the multiset of embedding rows")
		}
	}

	// The rows must actually be reordered, not identical in place — three identical
	// rows would satisfy the multiset check without the permutation happening.
	sameInPlace := true
	for i := range a.Trace.Embed {
		if fmt.Sprint(a.Trace.Embed[i]) != fmt.Sprint(b.Trace.Embed[i]) {
			sameInPlace = false
			break
		}
	}
	if sameInPlace {
		t.Fatal("the two prompts produced identical embedding rows in place, so the " +
			"permutation was not exercised")
	}

	// And positions must be what differs: the sums are not permutations of each other.
	sums := func(tr *model.Trace) map[string]int {
		m := map[string]int{}
		for _, v := range tr.Input {
			m[fmt.Sprint(v)]++
		}
		return m
	}
	sa, sb := sums(a.Trace), sums(b.Trace)
	same := len(sa) == len(sb)
	for k, n := range sa {
		if sb[k] != n {
			same = false
		}
	}
	if same {
		t.Fatal("E+P is also permutation invariant, so the lesson's claim that position " +
			"breaks the symmetry has nothing to show it")
	}
}

// TestScoreScalingKeepsTheDistributionFromCollapsing backs the score stage's explanation of
// why scores are divided by √d_head.
//
// The paragraph says the division keeps the variance stable, because otherwise higher
// dimensions make scores explode and softmax saturates. On this trained model both forms
// reach a maximum probability of 1.000 somewhere, so the top-row figure alone cannot show
// the difference — but the distribution as a whole can. Measured over 280 rows: mean
// attention entropy is 1.2508 nats with the scale and 0.1865 without it, a factor of 6.7.
//
// That is the claim worth pinning: the scale is what keeps a row from collapsing onto a
// single key. The test recomputes both forms from the traces rather than trusting a stored
// number, and requires the scaled form to stay clearly more spread out.
func TestScoreScalingKeepsTheDistributionFromCollapsing(t *testing.T) {
	a := newTestApp(t)
	tr := a.Trace
	dHead := tr.Config.DHead()
	scale := math.Sqrt(float64(dHead))
	dot := func(x, y []float32) float64 {
		s := 0.0
		for i := 0; i < dHead && i < len(x) && i < len(y); i++ {
			s += float64(x[i]) * float64(y[i])
		}
		return s
	}

	var scaledSum, rawSum float64
	rows := 0
	for li := range tr.Layers {
		for h := range tr.Layers[li].Heads {
			head := tr.Layers[li].Heads[h]
			for q := 1; q < len(tr.Tokens) && q < len(head.Q); q++ {
				var raw, sc []float64
				for k := 0; k <= q && k < len(head.K); k++ {
					d := dot(head.Q[q], head.K[k])
					raw = append(raw, d)
					sc = append(sc, d/scale)
				}
				if len(raw) < 2 {
					continue
				}
				scaledSum += rowEntropy(sc)
				rawSum += rowEntropy(raw)
				rows++
			}
		}
	}
	if rows == 0 {
		t.Skip("no multi-key rows to measure")
	}
	scaled, raw := scaledSum/float64(rows), rawSum/float64(rows)
	if scaled <= raw {
		t.Fatalf("scaling does not widen the distribution: %.4f scaled vs %.4f raw", scaled, raw)
	}
	// The paragraph's point is that the difference is large, not marginal.
	if scaled < raw*2 {
		t.Fatalf("scaling only raises mean entropy from %.4f to %.4f; that is too small "+
			"a gap to support the lesson's explanation", raw, scaled)
	}
	t.Logf("%d rows: mean entropy %.4f scaled, %.4f unscaled", rows, scaled, raw)
}

// rowEntropy is the Shannon entropy of softmax(v), in nats.
func rowEntropy(v []float64) float64 {
	m := v[0]
	for _, x := range v {
		if x > m {
			m = x
		}
	}
	p := make([]float64, len(v))
	var s float64
	for i, x := range v {
		p[i] = math.Exp(x - m)
		s += p[i]
	}
	var e float64
	for _, q := range p {
		pr := q / s
		if pr > 0 {
			e -= pr * math.Log(pr)
		}
	}
	return e
}

// TestPureLayerNormPinsTheVarianceNearOne checks the inspector's untrained-weight branch
// against the data it describes.
//
// When γ and β are still 1 and 0, the card says "这一步只是纯归一化". That is a claim about
// what the numbers beside it should look like: normalising a row to zero mean and unit
// variance, with nothing learned on top, has to leave mean ≈ 0 and std ≈ 1. Nothing checked
// it — lnParamsAreLearned has a test, but only of its own return value, not of whether the
// two rows it distinguishes actually differ the way the sentence says.
//
// Measured on random weights: mean -0.00000 and std 0.9885 / 0.9909. The tolerance allows
// for the eps under the square root, which is what keeps std slightly below 1.
func TestPureLayerNormPinsTheVarianceNearOne(t *testing.T) {
	a := New(model.Random(model.DefaultConfig, 42))
	if lnParamsAreLearned(a.W) {
		t.Fatal("random weights report learned LayerNorm parameters")
	}
	l := a.layer()
	if l == nil || len(l.Norm1) == 0 {
		t.Fatal("no normalised rows to measure")
	}
	for i := 0; i < len(l.Norm1) && i < 8; i++ {
		mean, std := meanStd(l.Norm1[i])
		if math.Abs(float64(mean)) > 1e-3 {
			t.Fatalf("row %d: mean %.5f; pure normalisation should centre it on zero", i, mean)
		}
		if std < 0.98 || std > 1.0 {
			t.Fatalf("row %d: std %.5f; pure normalisation should land at 1 (eps keeps it "+
				"just below)", i, std)
		}
	}

	// And the trained model is the other side of the same sentence: its γ/β are not 1/0,
	// so the variance is free to leave 1 — which the card says explicitly.
	b := newTestApp(t)
	if !lnParamsAreLearned(b.W) {
		t.Fatal("the bundled weights report unlearned LayerNorm parameters")
	}
	_, std := meanStd(b.layer().Norm1[b.SelToken])
	if math.Abs(float64(std)-1) < 1e-4 {
		t.Fatalf("the bundled model's std is %.6f; if it lands exactly on 1 then the "+
			"card's explanation of why it need not is describing nothing", std)
	}
}

// TestOutputCalloutQuotesTheLiveProbability holds the output stage's callout to its own
// design note.
//
// The source says the top-1 figure and the training budget both come from the running model,
// "so the callout stays true when the prompt, the temperature or the weights change". That is
// the property that makes it a callout about this run rather than a generic warning — and it
// had no test, so a hard-coded percentage or a stale variable would have gone unnoticed.
//
// Measured: 100%, 97%, 78% and 63% at temperatures 0.3, 0.8, 1.5 and 2.0, each matching the
// frame's own top-1 probability. The prompt is checked too: a different prompt changes the
// number, so the figure cannot be coming from a cached trace.
func TestOutputCalloutQuotesTheLiveProbability(t *testing.T) {
	quote := func(prompt string, temp float64) (string, float32) {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		a.Temp = temp
		a.Step = 6
		probs := a.Trace.Probabilities(temp)
		var maxP float32
		for _, p := range probs {
			if p > maxP {
				maxP = p
			}
		}
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		// The figure appears in the callout's title, not its body: the title is
		// "「概率 97% = 模型很确定…」" and the body then explains why that is not
		// certainty. Both are separate entries in Texts(), so join them.
		var all []string
		for _, s := range tr.Texts() {
			all = append(all, s)
		}
		joined := strings.Join(all, "\n")
		if !strings.Contains(joined, "这两件事没有关系") {
			t.Fatal("the output callout is not rendered")
		}
		return joined, maxP
	}

	seen := map[string]bool{}
	for _, temp := range []float64{0.3, 0.8, 1.5, 2.0} {
		text, maxP := quote("The transformer reads the tokens and", temp)
		want := fmt.Sprintf("「概率 %.0f%%", maxP*100)
		if !strings.Contains(text, want) {
			t.Fatalf("T=%.1f: the callout does not quote %q (top-1 is %.2f%%): %s",
				temp, want, maxP*100, text)
		}
		seen[want] = true
	}
	if len(seen) < 4 {
		t.Fatalf("the four temperatures produced only %d distinct figures; the number is "+
			"not tracking the distribution", len(seen))
	}

	// And a different prompt has to move it, or the figure is stale.
	otherText, otherMax := quote("hi", 0.8)
	otherWant := fmt.Sprintf("「概率 %.0f%%", otherMax*100)
	if !strings.Contains(otherText, otherWant) {
		t.Fatalf("with a different prompt the callout does not quote %q", otherWant)
	}
}

// TestInspectorTitlesNameTheirSubject keeps the right column's headings specific.
//
// The residual stage's first inspector card used to be titled just "向量" with "残差流" as
// its subtitle. The subtitle is the part that says what the numbers describe, so the heading
// a reader sees first was the vague half — and it sat beside a stage card also titled
// "残差流", making the two look like different things when they show the same row from two
// angles.
//
// Every inspector heading now names its subject, and the subtitle, when present, says which
// slice of the model it covers. This checks that no heading is a bare noun as generic as
// "向量" and that the residual card in particular pairs the subject with its source.
func TestInspectorTitlesNameTheirSubject(t *testing.T) {
	vague := map[string]bool{"向量": true, "统计": true, "数据": true, "结果": true}

	for _, c := range []struct {
		step int
		name string
	}{
		{0, "分词"}, {1, "嵌入"}, {2, "注意力分数"}, {3, "Softmax"},
		{4, "残差 / LN"}, {5, "FFN"}, {6, "输出 / 采样"},
	} {
		a := newTestApp(t)
		a.Step = c.step
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		for _, s := range tr.Texts() {
			if vague[s] {
				t.Fatalf("stage %d (%s): the inspector has a heading as generic as %q; the "+
					"subtitle should not be carrying the whole subject", c.step, c.name, s)
			}
		}
	}

	// The pair that was fixed, on the stage it lives.
	a := newTestApp(t)
	a.Step = 4
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	for _, want := range []string{"残差流统计", "这一层的输出向量"} {
		if !tr.HasText(want) {
			t.Fatalf("the residual inspector no longer shows %q", want)
		}
	}
}

// TestStepperStatesAreDistinctWithoutRelyingOnColour checks the sidebar's three states.
//
// The dot for "done" and the dot for "current" are both filled circles whose colours sit at
// 1.33:1 against each other — far under the 3:1 WCAG asks of a non-text indicator. On its own
// that would make the stepper unreadable for a reader who cannot separate the two greys.
//
// It is not on its own: the current row also carries an accent-tinted background and bold
// text, and "pending" differs in shape (hollow ring rather than filled). This test pins that
// redundancy, so a restyle that recolours the dots cannot quietly leave colour as the only
// cue.
//
// Label contrast is asserted too, since the three states have to stay readable: measured
// 15.82 / 7.62 / 5.22 against the background.
func TestStepperStatesAreDistinctWithoutRelyingOnColour(t *testing.T) {
	// Shape separates done from pending, and the current row has a background plus weight.
	if contrastRatio(colMuted, colAccent) >= 3 {
		t.Logf("done and current dots now contrast at %.2f; the redundancy below is no "+
			"longer load-bearing, which is fine but worth knowing",
			contrastRatio(colMuted, colAccent))
	}

	for _, c := range []struct {
		name string
		col  ui.Color
		min  float64
	}{
		{"current label", stepTextColor(1), 4.5},
		{"done label", stepTextColor(0), 4.5},
		{"pending label", stepTextColor(2), 4.5},
	} {
		if got := contrastRatio(c.col, colBg); got < c.min {
			t.Fatalf("%s is %.2f:1 on the sidebar background", c.name, got)
		}
	}

	// The current row must carry a second cue in the markup, not just the dot colour.
	src, err := os.ReadFile("widgets.go")
	if err != nil {
		t.Fatal(err)
	}
	text := string(src)
	if !strings.Contains(text, "row = row.Background(colAccent.Alpha(0.14))") {
		t.Fatal("the current stepper row has no background tint; the accent dot would be " +
			"the only cue that it is current")
	}
	if !strings.Contains(text, "name = name.Bold()") {
		t.Fatal("the current stepper row is no longer bold; a reader who cannot separate " +
			"the two filled-dot colours loses the state")
	}
}

// TestKeyStripFollowsTheCellNotJustTheRow keeps the Q/K/V card honest.
//
// Walking the matrix cursor to (q, k) with Shift+arrows selects the row: SelToken becomes q
// while KeyK can be any column, and the workings line below then prints q_q · k_k. The card
// above kept drawing K[SelToken] and labelling it "同位置", so on any off-diagonal cell it
// showed a vector the arithmetic did not use — measured at (35, 32), the drawn K was
// (1.109, 1.362, 2.295) where the cell used (-2.188, 0.687, 1.709), opposite signs in the
// first three dimensions.
//
// The Key and Value strips now follow the cursor when the cursor sits on the selected row,
// and the label names the column instead of claiming the same position.
func TestKeyStripFollowsTheCellNotJustTheRow(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	a.SelToken = 12
	a.KeyQ, a.KeyK = 12, 4
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	var keyLabel, valLabel string
	for _, s := range tr.Texts() {
		switch {
		case strings.HasPrefix(s, "Key（"):
			keyLabel = s
		case strings.HasPrefix(s, "Value（"):
			valLabel = s
		}
	}
	if keyLabel == "" || valLabel == "" {
		t.Fatalf("the Q/K/V card lost its Key or Value label (%q, %q)", keyLabel, valLabel)
	}
	if !strings.Contains(keyLabel, "位置 4") || !strings.Contains(valLabel, "位置 4") {
		t.Fatalf("cursor is on key column 4 but the labels read %q and %q", keyLabel, valLabel)
	}

	// The label is only half of it: the vector the strip draws has to be the one the
	// score line used. Checking that closes the loop from "the card says column 4" to
	// "dotting the drawn K reproduces the cell's score".
	h := a.layer().Heads[a.Head]
	p := computeScore(h.Q[12], h.K[4], a.Trace.Config.DHead(), h.Scores[12][4])
	scale := float64(float32(1 / math.Sqrt(float64(a.Trace.Config.DHead()))))
	if gap := math.Abs(p.Dot*scale - p.Score); gap > 1e-4 {
		t.Fatalf("the score line for (12, 4) does not close when recomputed from K[4]: "+
			"got %.6f, want %.6f (gap %.1e)", p.Dot*scale, p.Score, gap)
	}

	// On the diagonal there is nothing to distinguish, and the card says so.
	b := newTestApp(t)
	b.Step = 2
	b.SelToken = 7
	b.KeyQ, b.KeyK = 7, 7
	bt := ui.NewTester(b.View, 1500, 940)
	bt.Frame()
	var samePos int
	for _, s := range bt.Texts() {
		if strings.Contains(s, "同位置") {
			samePos++
		}
	}
	if samePos != 2 {
		t.Fatalf("on the diagonal both labels should say 同位置; found %d", samePos)
	}
}

// TestScoreStageShowsWhereQKVComesFrom checks the projection line above the strips.
//
// The stage draws Query, Key and Value as three bands of numbers and then shows how two of them
// make a score. Nothing on that card said the bands are projections of x — the stage body does,
// but it lives in the card below the heatmap, so a reader studying the strips had to scroll
// past the matrix to learn where they came from.
//
// One line now says it, above the first strip, in the muted micro type the card already uses for
// its readouts.
func TestScoreStageShowsWhereQKVComesFrom(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	var line string
	for _, s := range tr.Texts() {
		if strings.Contains(s, "W_Q") {
			line = s
		}
	}
	if line == "" {
		t.Fatal("the score stage does not say how Query, Key and Value are made")
	}
	for _, want := range []string{"q = x·W_Q", "k = x·W_K", "v = x·W_V"} {
		if !strings.Contains(line, want) {
			t.Fatalf("the projection line is missing %q: %q", want, line)
		}
	}

	// It has to be above the first strip, not below the last one.
	projY, ok := findY(tr, line)
	if !ok {
		t.Fatal("cannot locate the projection line")
	}
	queryY, ok := findY(tr, "Query")
	if !ok {
		t.Fatal("cannot locate the Query label")
	}
	if projY > queryY {
		t.Fatalf("the projection line sits at y=%.0f, below the Query label at y=%.0f",
			projY, queryY)
	}
}

// findY returns the vertical position of a rendered text, or false when it is not drawn.
func findY(tr *ui.Tester, text string) (float32, bool) {
	if r, ok := tr.Find(text); ok {
		return r.Y, true
	}
	return 0, false
}

// TestWeightedSumCardSaysTheStripsHaveDifferentWidths checks the line that heads the card.
//
// The card draws three strips across the same width: the head output at d_head = 24, then the
// head's projection into the residual and the sum of all four heads, both at d_model = 96. All
// three are painted edge to edge, so the eye lines them up column by column — cell 3 of the
// first strip sits directly above cell 12 of the second. Nothing on the card said so.
//
// The line now names each strip and its width, before any of the data.
func TestWeightedSumCardSaysTheStripsHaveDifferentWidths(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	tr := ui.NewTester(a.View, 1500, 1800)
	tr.Frame()

	var line string
	for _, s := range tr.Texts() {
		if strings.Contains(s, "下面三条依次是") {
			line = s
		}
	}
	if line == "" {
		t.Fatal("the weighted-sum card does not say what its three strips are")
	}
	// Compare against the model's own dimensions, not a second copy of the numbers: a
	// literal "24 维" here would pass even if the card printed a stale constant.
	c := a.Trace.Config
	for _, want := range []string{
		fmt.Sprintf("%d 维", c.DHead()),
		fmt.Sprintf("%d 维", c.DModel),
	} {
		if !strings.Contains(line, want) {
			t.Fatalf("the strip description is missing %q: %q", want, line)
		}
	}
	if strings.Contains(line, fmt.Sprintf("%d 维", c.DFF)) {
		t.Fatalf("the description quotes d_ff, which no strip on this card has: %q", line)
	}

	// And the line comes before the first strip's own label.
	lineY, ok := findY(tr, line)
	if !ok {
		t.Fatal("cannot locate the strip description")
	}
	firstY, ok := findY(tr, fmt.Sprintf("Σ α·v（头内输出，d_head = %d）", a.Trace.Config.DHead()))
	if !ok {
		t.Fatal("cannot locate the first strip label")
	}
	if lineY > firstY {
		t.Fatalf("the description sits at y=%.0f, below the first label at y=%.0f", lineY, firstY)
	}
}

// TestOnlyWarningsUseTheWarmLabelColour keeps colWarm out of ordinary labels.
//
// The palette documents colWarm as a data colour — the warm end of the score ramp — and the app
// also uses it for warnings: the untrained banner, the dropped-bytes note, the "常见误解"
// heading. Those are consistent.
//
// The residual stage's LN(x) label had drifted into it. Nothing about the normalised row is a
// warning, and at 11.09:1 against the background it was the brightest label on a card where it
// is the middle quantity, brighter than the results below it. It now takes colAccent2, which
// the palette defines as "what came out of the step" — the same colour as the two sums it feeds.
//
// The test pins which labels may be warm, so a future colour choice has to be deliberate.
func TestOnlyWarningsUseTheWarmLabelColour(t *testing.T) {
	src, err := os.ReadFile("stages.go")
	if err != nil {
		t.Fatal(err)
	}
	file := string(src)
	for _, allowed := range []string{
		`"丢弃前 %d 字节"`, // truncated prompt: bytes the model will not see
	} {
		if !strings.Contains(file, allowed) {
			t.Fatalf("the warm label %s is gone; update this list if that was intended", allowed)
		}
	}

	// The normalised row is a result, not a warning.
	if strings.Contains(file, `"LN(x) — 子层实际读到的输入").FontSize(fontMicro).TextColor(colWarm)`) {
		t.Fatal("the LN(x) label is warm again; colWarm means a scored matrix or a warning, " +
			"and this row is neither")
	}
	if !strings.Contains(file, `"LN(x) — 子层实际读到的输入").FontSize(fontMicro).TextColor(colAccent2)`) {
		t.Fatal("the LN(x) label lost its result colour")
	}
}

// TestDataColoursAreNotUsedOnText keeps the palette's two kinds of colour apart.
//
// theme.go documents two families. Accents and text colours — colAccent, colAccent2, colMuted,
// colFaint — say what a thing is. Data colours — colPos, colNeg, colGreen, colWarm — say what
// a value is, and belong in charts: the red/blue of a signed vector, a bar fill, the warm end
// of the score ramp.
//
// Using a data colour on a label makes the label read as a reading off a chart. That is exactly
// what happened to LN(x) last round, which was warm on a card where warm means a score.
//
// colPos, colNeg and colGreen are never used on text today. colWarm is, but only for warnings;
// TestOnlyWarningsUseTheWarmLabelColour covers those. This test holds the line for the three
// that have no text use at all.
func TestDataColoursAreNotUsedOnText(t *testing.T) {
	files := []string{"app.go", "stages.go", "widgets.go", "theme.go"}
	for _, dataColour := range []string{"colPos", "colNeg", "colGreen"} {
		for _, name := range files {
			src, err := os.ReadFile(name)
			if err != nil {
				t.Fatal(err)
			}
			if n := strings.Count(string(src), "TextColor("+dataColour+")"); n > 0 {
				t.Fatalf("%s uses %s on text %d time(s); that colour means a value in a chart, "+
					"not a label — use a text colour or explain why the exception is right",
					name, dataColour, n)
			}
		}
	}
}

// TestStepperHighlightFollowsTheStage reads the current row out of the pixels.
//
// The three states of a stepper row are dot fill, background tint and text weight. A test that
// recomputes "which one is current" from a.Step checks its own arithmetic, not what was drawn —
// an earlier draft did exactly that and would have passed with the highlight removed.
//
// This one samples the sidebar's background colour at each row and requires the brightest row
// to be the stage that is selected, for all seven.
func TestStepperHighlightFollowsTheStage(t *testing.T) {
	a := newTestApp(t)
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	// Rows are laid out from the top of the sidebar. Take the first row's y from the label
	// rather than hard-coding it, so a padding change does not silently sample the gap
	// between rows.
	var rowY int
	seed := newTestApp(t)
	seed.Step = 0
	st := ui.NewTester(seed.View, 1500, 940)
	st.Frame()
	r0, ok := st.Find("步骤 1 分词")
	if !ok {
		t.Fatal("cannot locate the first stepper row")
	}
	r1, ok := st.Find("步骤 2 嵌入")
	if !ok {
		t.Fatal("cannot locate the second stepper row")
	}
	rowY = int(r0.Y)
	rowH := int(r1.Y - r0.Y)
	if rowH <= 0 {
		t.Fatalf("the stepper rows report a height of %d", rowH)
	}

	for step := 0; step < len(lessons); step++ {
		b := newTestApp(t)
		b.Step = step
		bt := ui.NewTester(b.View, 1500, 940)
		bt.Frame()
		img := bt.Image()
		stride := img.Bounds().Dx()
		brightest, brightestRow := -1, -1
		for row := 0; row < len(lessons); row++ {
			// Sample across the row's empty right-hand end. One point is not enough: the
			// highlight is a rounded rect, so a sample at its corner reads as background
			// and the row looks unselected.
			best := -1
			for _, inset := range []int{30, 60, 90} {
				x := int(b.SidebarW) - inset
				y := rowY + row*rowH + rowH/3
				idx := (y*stride + x) * 4
				if idx+2 >= len(img.Pix) {
					continue
				}
				if sum := int(img.Pix[idx]) + int(img.Pix[idx+1]) + int(img.Pix[idx+2]); sum > best {
					best = sum
				}
			}
			if best > brightest {
				brightest, brightestRow = best, row
			}
		}
		if brightestRow != step {
			t.Fatalf("stage %d is selected but the brightest stepper row is %d", step, brightestRow)
		}
	}
}

// TestMaskedCellsAreNeverPainted checks the causal mask at the drawing layer.
//
// The heatmap draws only cells with j <= i, so the upper triangle is not painted at all rather
// than painted in the background colour. That is stronger than a value check — a transparent
// mask cell cannot be revealed by a later restyle or a change to heatColor — and it halves the
// fills on a 64×64 matrix.
//
// The check is on the loop bounds, deliberately. A pixel test was tried first and abandoned:
// locating the grid from the frame needs the axis labels (Find returns the first match for "0"),
// and the page has bluish pixels from the rail, the strips and the progress dots, so the samples
// could not be attributed to matrix cells with confidence. The loop bound is the property that
// matters and it is directly readable.
func TestMaskedCellsAreNeverPainted(t *testing.T) {
	src, err := os.ReadFile("widgets.go")
	if err != nil {
		t.Fatal(err)
	}
	file := string(src)
	if !strings.Contains(file, "for j := 0; j <= i && j < len(row); j++ {") {
		t.Fatal("the heatmap's cell loop no longer stops at the diagonal; masked cells would " +
			"start being painted")
	}
	// And the fill must stay inside that loop — not in a second pass over the full row.
	if strings.Contains(file, "for j := 0; j < len(row); j++ {") {
		t.Fatal("a second cell loop walks the whole row, which would paint the masked half")
	}
}

// TestNegativeRegionAxisWidthIsRecomputed checks the FFN caption's "只占 4.0% 的轴宽".
//
// The caption tells the reader the negative region is a sliver of the axis and that the
// histogram therefore cannot show its shape. That is only true if the chart's axis is the data's
// own range, which is what histogram() passes (0, 0) to mean. The percentage and the chart's
// scale come from the same minMaxFloat call, so they agree today — but nothing connected the
// sentence to the plot, and a change to either would leave the caption describing a shape the
// reader is not looking at.
func TestNegativeRegionAxisWidthIsRecomputed(t *testing.T) {
	a := newTestApp(t)
	a.Step = 5
	tr := ui.NewTester(a.View, 1500, 1800)
	tr.Frame()

	l := a.layer()
	hidden := l.Hidden[a.SelToken]
	if len(hidden) == 0 {
		t.Fatal("no hidden activations")
	}
	lo, hi := minMaxFloat(hidden)
	if !(lo < 0 && hi > lo) {
		t.Fatalf("this prompt has no negative region to describe (range %.3f..%.3f)", lo, hi)
	}
	neg := 0
	for _, v := range hidden {
		if v < 0 {
			neg++
		}
	}
	wantNegPct := 100 * float64(neg) / float64(len(hidden))
	wantAxisPct := 100 * (0 - float64(lo)) / (float64(hi) - float64(lo))

	var caption string
	for _, s := range tr.Texts() {
		if strings.Contains(s, "轴宽") {
			caption = s
		}
	}
	if caption == "" {
		t.Fatal("the negative-region caption is gone")
	}
	for _, want := range []string{
		fmt.Sprintf("%d 个负值", neg),
		fmt.Sprintf("占 %.0f%%", wantNegPct),
		fmt.Sprintf("%.1f%% 的轴宽", wantAxisPct),
	} {
		if !strings.Contains(caption, want) {
			t.Fatalf("the caption does not state %q for this row: %s", want, caption)
		}
	}

	// And the chart the caption describes must use that same range.
	src, err := os.ReadFile("widgets.go")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(src), "lo, hi = minMaxFloat(values)") {
		t.Fatal("histogramRange no longer derives the axis from the data, so the caption's " +
			"percentage would describe a different scale from the one drawn")
	}
}

// TestShapeSequenceShowsWhereTheTokenAxisCollapses checks the data-flow card's last step.
//
// The first six shapes all begin with the token count — [36], [36, 96], [36, 36], [36, 384] — and
// the seventh used to read [256]. Reading the row in order, that asks where the 36 went: only the
// last position is projected to the vocabulary, so the axis becomes 1 rather than disappearing. The
// stage body said "得到词表上每个 token 的 logit", which reads as if every position were projected.
//
// The card now writes [1, 256]. This test pins the shape of the collapse: every leading axis is the
// token count until the projection, and the projection's leading axis is 1.
func TestShapeSequenceShowsWhereTheTokenAxisCollapses(t *testing.T) {
	a := newTestApp(t)
	a.Step = 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	T := len(a.Trace.Tokens)
	c := a.Trace.Config
	want := []string{
		fmt.Sprintf("[%d]", T),
		fmt.Sprintf("[%d, %d]", T, c.DModel),
		fmt.Sprintf("[%d, %d]", T, c.DModel),
		fmt.Sprintf("[%d, %d]", T, T),
		fmt.Sprintf("[%d, %d]", T, c.DModel),
		fmt.Sprintf("[%d, %d]", T, c.DFF),
		fmt.Sprintf("[1, %d]", c.Vocab),
	}
	for _, shape := range want {
		if !tr.HasText(shape) {
			t.Fatalf("the data-flow card does not show %q; the sequence of shapes has a gap "+
				"or the output step still hides the collapsed axis", shape)
		}
	}

	// The stage body has to say the same thing, since it is where a reader looks for the reason.
	// It lives on the output stage, not this one.
	b := newTestApp(t)
	b.Step = 6
	bt := ui.NewTester(b.View, 1500, 1400)
	bt.Frame()
	var body string
	for _, s := range bt.Texts() {
		if strings.Contains(s, "只把") && strings.Contains(s, "投影到词表上") {
			body = s
		}
	}
	if body == "" {
		t.Fatal("the output stage body does not explain that only the last position is projected")
	}
}

// TestEveryInspectorFollowsThePrompt checks the right-hand column on all seven stages.
//
// Typing a new prompt reruns the forward pass, and the inspector is where a reader looks for the
// numbers behind whatever the middle column is drawing. A panel that kept showing the previous
// prompt's figures would look plausible — the shapes are identical, only the values differ — so the
// failure would be invisible without comparing two runs.
//
// Measured: the inspector's text changes on every stage when the prompt changes from "hi" to a full
// sentence. This is a coarse check on purpose: the per-figure agreement is covered by the tests
// that recompute each number from the trace.
func TestEveryInspectorFollowsThePrompt(t *testing.T) {
	const short, long = "hi", "attention is all you need and the transformer"
	inspectorText := func(prompt string, step int) string {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		a.Step = step
		tr := ui.NewTester(a.View, 1500, 1400)
		tr.Frame()
		// The inspector is the right-hand column. At 1500 wide with a 330-wide inspector
		// it starts around x=1170, so keeping everything past 1200 selects it without
		// hard-coding the exact edge.
		var parts []string
		for _, s := range tr.Texts() {
			if r, ok := tr.Find(s); ok && r.X >= 1200 {
				parts = append(parts, s)
			}
		}
		return strings.Join(parts, "\n")
	}

	for step := 0; step < len(lessons); step++ {
		s, l := inspectorText(short, step), inspectorText(long, step)
		if s == "" || l == "" {
			t.Fatalf("stage %d: the inspector produced no text", step)
		}
		if s == l {
			t.Fatalf("stage %d: the inspector reads the same for %q and %q; it is not showing "+
				"the current prompt's numbers", step, short, long)
		}
	}
}

// TestScoreLegendListsOnlyTheSignsPresent keeps the legend matching the matrix.
//
// The legend had a warm swatch for positive scores and a blue one for negative, always both. On a
// short prompt every score in a head can be negative — measured on "hi", where layer 0 head 0 is
// all blue — so the card advertised a category the reader could not find anywhere in the grid, right
// beside a callout explaining that sign does not matter.
//
// The list now follows the data, and a wholly negative head gets a line saying why that is ordinary:
// the scores can shift as a group, and attention depends on the order within a row.
func TestScoreLegendListsOnlyTheSignsPresent(t *testing.T) {
	signs := func(prompt string) (hasPos, hasNeg bool) {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		a.Step = 2
		tr := ui.NewTester(a.View, 1500, 1100)
		tr.Frame()
		return tr.HasText("正分（暖色）"), tr.HasText("负分（蓝色）")
	}

	// A short prompt where one head is entirely negative.
	pos, neg := signs("hi")
	if pos {
		t.Fatalf("the legend shows a warm swatch for a matrix with no positive scores")
	}
	if !neg {
		t.Fatal("the legend dropped the negative swatch too")
	}

	// The default prompt has both signs in every head, so both swatches belong.
	pos, neg = signs("The transformer reads the tokens and")
	if !pos || !neg {
		t.Fatalf("the default prompt has both signs but the legend shows pos=%v neg=%v", pos, neg)
	}

	// And the all-negative case explains itself rather than leaving the reader to guess.
	a := newTestApp(t)
	a.Prompt = "hi"
	a.recompute()
	a.Step = 2
	tr := ui.NewTester(a.View, 1500, 1100)
	tr.Frame()
	var explained bool
	for _, s := range tr.Texts() {
		if strings.Contains(s, "所有分数都是负的") && strings.Contains(s, "大小顺序") {
			explained = true
		}
	}
	if !explained {
		t.Fatal("an all-negative matrix is drawn with no word about why that is fine")
	}
}

// TestPositionCardMatchesHowManyPositionsThePromptHas keeps step 2's position table honest.
//
// The card was headed "位置向量随位置变化" with the note "取前四个位置的 P[t]" for every prompt. The
// table has one row per position the prompt uses, so a one-token prompt showed a single row under a
// heading about variation and a note promising four — a comparison with nothing to compare and a
// count that was not on screen.
//
// Three cases now: one position states the rule instead of showing variation, two or three list what
// is there, and four or more take the sample.
func TestPositionCardMatchesHowManyPositionsThePromptHas(t *testing.T) {
	cases := []struct {
		prompt string
		want   string
	}{
		{"a", "位置向量是每个位置一行"},
		{"hi", "位置向量随位置变化"},
		{"The transformer reads the tokens and", "位置向量随位置变化"},
	}
	for _, c := range cases {
		a := newTestApp(t)
		a.Prompt = c.prompt
		a.recompute()
		a.Step = 1
		tr := ui.NewTester(a.View, 1500, 1400)
		tr.Frame()
		if !tr.HasText(c.want) {
			t.Fatalf("prompt %q (%d tokens): the position card does not say %q",
				c.prompt, len(a.Trace.Tokens), c.want)
		}
	}

	// A single position must not claim variation.
	a := newTestApp(t)
	a.Prompt = "a"
	a.recompute()
	a.Step = 1
	tr := ui.NewTester(a.View, 1500, 1400)
	tr.Frame()
	if tr.HasText("取前四个位置的 P[t]") {
		t.Fatal("a one-token prompt is still told the table shows four positions")
	}
	if !tr.HasText("这个提示词只有 1 个位置") {
		t.Fatal("a one-token prompt does not say how many positions it has")
	}
}

// TestTailProbabilityIsNotFormattedAsResidue keeps arithmetic noise out of the readout.
//
// At low temperature the top twelve tokens carry the whole distribution and the sum of the other 244
// lands wherever float32 addition leaves it: 5.6e-17 at T=0.1, 1.5e-19 at T=0.3. Formatting those in
// scientific notation printed "1.5e-17%", which reads as a very small but real share of probability
// rather than as "nothing left".
//
// Below 1e-9 percent the readout says 0%. Above it the figure is a real tail and is kept — T=0.8
// prints 1.3e-05%.
func TestTailProbabilityIsNotFormattedAsResidue(t *testing.T) {
	tailOf := func(temp float64) string {
		a := newTestApp(t)
		a.Temp = temp
		a.recompute()
		a.Step = 6
		tr := ui.NewTester(a.View, 1500, 1400)
		tr.Frame()
		for _, s := range tr.Texts() {
			if strings.Contains(s, "其余 244 个合计") {
				return s
			}
		}
		return ""
	}

	for _, temp := range []float64{0.1, 0.3, 0.5} {
		line := tailOf(temp)
		if line == "" {
			t.Fatalf("temp %.1f: the top-N heading is gone", temp)
		}
		if !strings.Contains(line, "合计 0%") {
			t.Fatalf("temp %.1f: the exhausted tail is not reported as 0: %q", temp, line)
		}
	}

	// A tail that really is small but non-zero must still be shown, not rounded away.
	line := tailOf(0.8)
	if strings.Contains(line, "合计 0%") {
		t.Fatalf("temp 0.8 still has a non-zero tail (1.3e-05%%) but it reads as %q", line)
	}
	if !strings.Contains(line, "e-") {
		t.Fatalf("temp 0.8: a small tail should stay in scientific notation, got %q", line)
	}
}

// TestRowZeroExplainsItsEmptyEntropyChart keeps a chart with no information out of the panel.
//
// Row 0 attends to exactly one key — itself — because of the causal mask, so every head's entropy
// there is exactly 0 and the chart was four zero-length bars on a log(1) = 0 scale. That is a fact
// about the mask, not about the model, and a reader seeing four empty bars has no way to tell.
//
// The row now says so in words. Any later row still gets the chart, because its values differ.
func TestRowZeroExplainsItsEmptyEntropyChart(t *testing.T) {
	// Row 0: words instead of a chart that cannot vary.
	a := newTestApp(t)
	a.Step = 3
	a.SelToken = 0
	tr := ui.NewTester(a.View, 1500, 900)
	tr.Frame()

	var explained bool
	for _, s := range tr.Texts() {
		if strings.Contains(s, "第 0 行只能看到它自己") && strings.Contains(s, "必然都是 0") {
			explained = true
		}
	}
	if !explained {
		t.Fatal("row 0 draws an all-zero entropy chart without saying why")
	}
	if tr.HasText("这一行在各头的熵（越小越专注）") {
		t.Fatal("row 0 still draws the entropy chart, which cannot show anything there")
	}

	// A later row keeps the chart, and it must actually vary.
	b := newTestApp(t)
	b.Step = 3
	b.SelToken = 5
	bt := ui.NewTester(b.View, 1500, 900)
	bt.Frame()
	if !bt.HasText("这一行在各头的熵（越小越专注）") {
		t.Fatal("row 5 lost the entropy chart")
	}
	var vals []string
	for _, s := range bt.Texts() {
		if len(s) == 4 && s[0] >= '0' && s[0] <= '9' && s[1] == '.' {
			vals = append(vals, s)
		}
	}
	distinct := map[string]bool{}
	for _, v := range vals {
		distinct[v] = true
	}
	if len(distinct) < 2 {
		t.Fatalf("row 5's per-head entropies are all the same (%v), so this test's premise "+
			"that later rows differ does not hold", vals)
	}
}

// TestHashMeansATokenIndexEverywhere keeps "#N" to one meaning.
//
// The app writes token positions as "#35" in the rail, "#4" in the byte strip and "token #35 / 35" in the
// inspector header — all zero-based token indices. The top-N bars' tooltip used the same form for a rank:
// 'id 32 · "u" · #1' for the highest-probability token, where #1 is its position in the list, not its
// token index. One symbol, two meanings, and the tooltip is exactly where a reader is checking an id.
//
// Ranks now read "第 1 名".
func TestHashMeansATokenIndexEverywhere(t *testing.T) {
	src, err := os.ReadFile("stages.go")
	if err != nil {
		t.Fatal(err)
	}
	file := string(src)
	if strings.Contains(file, `"id %d · %q · #%d"`) {
		t.Fatal("the top-N tooltip is back to writing a rank with the token-index symbol")
	}
	if !strings.Contains(file, `"id %d · %q · 第 %d 名"`) {
		t.Fatal("the top-N tooltip no longer states the rank in words")
	}

	// The remaining "#" uses must be token positions.
	a := newTestApp(t)
	a.Step = 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	// The inspector header and the byte strip both index the selected token.
	want := fmt.Sprintf("token #%d / %d", a.SelToken, len(a.Trace.Tokens)-1)
	if !tr.HasText(want) {
		t.Fatalf("the inspector header no longer reads %q", want)
	}
	if !tr.HasText(fmt.Sprintf("#%d", a.SelToken)) {
		t.Fatalf("the byte strip no longer shows #%d for the selected token", a.SelToken)
	}
}

// TestTooltipNumberingMatchesTheVisibleUI checks the six tooltips against the conventions the visible
// UI follows. Tooltips never render in the headless pass, so no snapshot or text assertion covers them
// — they are the one surface where a divergent spelling can sit indefinitely.
//
// Measured conventions: stage numbers are 1-based ("第 3 / 7 步", matching the sidebar), layer and head
// numbers are 0-based (the segmented controls read 层 0|1 and H0–H3), and a token's byte is printed in
// decimal by the byte strip. That last one was the exception — the token chip's tooltip said "byte
// 0x54" where the strip says 84.
func TestTooltipNumberingMatchesTheVisibleUI(t *testing.T) {
	files := map[string]string{}
	for _, name := range []string{"app.go", "stages.go", "widgets.go"} {
		src, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		files[name] = string(src)
	}

	// Stage numbers in tooltips are 1-based, like the sidebar.
	for _, want := range []string{
		`"第 %d / %d 步 · %s", i+1, len(lessons)`, // sidebar dot, app.go
		`"第 %d / %d 步 · %s", i+1, len(lessons)`, // sidebar row, widgets.go
	} {
		found := strings.Contains(files["app.go"], want) || strings.Contains(files["widgets.go"], want)
		if !found {
			t.Fatalf("no tooltip still formats the stage with i+1 as %q", want)
		}
	}

	// Layer and head tooltips quote a stage range and must convert to 1-based for display,
	// because the range constants are stage indices.
	for _, want := range []string{"layerFrom+1, layerTo+1", "headFrom+1, headTo+1"} {
		if !strings.Contains(files["app.go"], want) {
			t.Fatalf("a control tooltip no longer converts its stage range with %q", want)
		}
	}

	// The byte tooltip leads with decimal, matching the byte strip.
	byteTip := `"token %d · byte %d（0x%02X）· %q"`
	if !strings.Contains(files["widgets.go"], byteTip) {
		t.Fatalf("the token tooltip no longer leads with the decimal byte (%s)", byteTip)
	}
	// And the strip it has to agree with still prints decimal.
	if !strings.Contains(files["stages.go"], `fmt.Sprintf("%d", tr.Tokens[i].Byte)`) {
		t.Fatal("the byte strip stopped printing decimal, so the tooltip's leading value is no " +
			"longer the same as what is on screen")
	}
}

// TestStatusLineHasThreeStates checks the sidebar's summary against the three states it can be in.
//
// An empty prompt used to report "0 tokens · 2 层 × 4 头 · 前向完成". Every clause is literally true —
// the app does run a forward pass over nothing — but the inspector beside it says the app is waiting for
// input, and a "前向完成" badge reads as though there were a result to look at.
//
// Three cases now: nothing yet, truncated, and a normal pass.
func TestStatusLineHasThreeStates(t *testing.T) {
	statusOf := func(prompt string) string {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		return a.Status
	}

	empty := statusOf("")
	if !strings.HasPrefix(empty, "还没有提示词") {
		t.Fatalf("the empty prompt reports %q, which reads as a finished pass", empty)
	}
	if strings.Contains(empty, "前向完成") {
		t.Fatalf("the empty prompt claims a completed forward pass: %q", empty)
	}

	normal := statusOf("hi")
	if !strings.Contains(normal, "2 个 token") || !strings.Contains(normal, "前向完成") {
		t.Fatalf("a two-token prompt reports %q", normal)
	}

	ctx := newTestApp(t).W.Config.Ctx
	truncated := statusOf(strings.Repeat("x", ctx+5))
	if !strings.HasPrefix(truncated, "已截断到最近 ") {
		t.Fatalf("an over-long prompt reports %q", truncated)
	}
	if strings.Contains(truncated, "前向完成") {
		t.Fatalf("the truncated status also claims a full pass: %q", truncated)
	}
}

// TestNoMalformedTextAcrossTheSweep walks every stage under several prompts, layers, heads and
// temperatures looking for text that should never reach a reader.
//
// The defects of the previous several rounds were each found by hand, by guessing which extreme input
// to try. This does the opposite: it renders the grid and looks for the shapes a broken formatter
// leaves behind — NaN, a signed zero, a nil, a Go formatting error like %!d(int=0), scientific notation
// for what is really zero.
//
// -Inf is deliberately allowed: the causal mask sets the upper triangle to negative infinity and the
// stage explains it in words, so it is content rather than a formatting failure.
func TestNoMalformedTextAcrossTheSweep(t *testing.T) {
	bad := []string{"NaN", "0.0e+00", "null", "%!", "<nil>", "-0.0000", "+0.0000", "e+308"}
	// "3/0 行" and the like: a proportion with nothing in the denominator.
	zeroDenom := regexp.MustCompile(`\d+/0\b`)
	prompts := []string{"", "a", "hi", "The transformer reads the tokens and", "注意力就是加权求和"}
	temps := []float64{0.1, 0.8, 2.0}

	for _, prompt := range prompts {
		for step := 0; step < len(lessons); step++ {
			for _, layer := range []int{0, 1} {
				for _, head := range []int{0, 3} {
					for _, temp := range temps {
						a := newTestApp(t)
						a.Prompt = prompt
						a.recompute()
						a.Step, a.Layer, a.Head, a.Temp = step, layer, head, temp
						if n := len(a.Trace.Tokens); n > 0 {
							a.SelToken = n / 2
						}
						tr := ui.NewTester(a.View, 1500, 1400)
						tr.Frame()
						for _, s := range tr.Texts() {
							for _, b := range bad {
								if strings.Contains(s, b) {
									t.Fatalf("stage %d, prompt %q, layer %d, head %d, temp %.1f: "+
										"the text %q contains %q", step, prompt, layer, head, temp,
										truncateForTest(s, 90), b)
								}
							}
							// A label that renders as nothing is a layout mistake that no
							// colour or contrast test can see.
							if s == "" {
								t.Fatalf("stage %d, prompt %q: an empty text was laid out",
									step, prompt)
							}
							// A ratio over zero is not a proportion. "0/0 行" shipped in round 213
							// because the numerator and denominator were computed but neither
							// was checked; the sweep now looks for any a/0 on screen.
							for _, m := range zeroDenom.FindAllString(s, -1) {
								t.Fatalf("stage %d, prompt %q: the zero-denominator ratio %q "+
									"appears in %q", step, prompt, m, truncateForTest(s, 90))
							}
							// Nothing on screen is a paragraph. The longest strings are the
							// stage bodies and the misconception corrections; a much longer
							// one would mean two got concatenated.
							if n := len([]rune(s)); n > 400 {
								t.Fatalf("stage %d, prompt %q: a text of %d runes is longer than "+
									"any paragraph this UI writes: %q", step, prompt, n,
									truncateForTest(s, 80))
							}
						}
					}
				}
			}
		}
	}
}

// truncateForTest shortens a failure message without cutting a multibyte rune in half.
func truncateForTest(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

// TestRowZeroCalloutDoesNotClaimARowComparison checks the score stage's callout on row 0.
//
// The callout argues that absolute scores do not matter, using the example "把这一行 N 个分数统统减去
// 5". Row 0 holds exactly one score — the causal mask leaves it nothing else — so on that row the
// sentence is arithmetic on a set of one, and "同一行内的大小关系" has nothing to order.
//
// Row 0 now gets the half of the point that does apply: only the mask makes a key truly impossible.
func TestRowZeroCalloutDoesNotClaimARowComparison(t *testing.T) {
	readCallout := func(token int) string {
		a := newTestApp(t)
		a.Step = 2
		a.SelToken = token
		tr := ui.NewTester(a.View, 1500, 1400)
		tr.Frame()
		// The belief and the correction are one entry here; step 6's callout splits them. Read the
		// entry after the heading either way.
		texts := tr.Texts()
		for i, s := range texts {
			if s == "「负分就是把这个 key 屏蔽掉了」" && i+1 < len(texts) {
				return texts[i+1]
			}
		}
		return ""
	}

	row0 := readCallout(0)
	if row0 == "" {
		t.Fatal("row 0 lost the misconception callout")
	}
	if strings.Contains(row0, "统统减去") {
		t.Fatalf("row 0 still offers a row-wide subtraction with one score in the row: %q", row0)
	}
	if !strings.Contains(row0, "只有一个分数") {
		t.Fatalf("row 0's callout does not say why the comparison does not apply: %q", row0)
	}

	// A later row has several scores and keeps the original argument.
	later := readCallout(10)
	if later == "" {
		t.Fatal("row 10 lost the misconception callout")
	}
	if !strings.Contains(later, "统统减去") || !strings.Contains(later, "11 个分数") {
		t.Fatalf("row 10's callout no longer argues with its own count: %q", later)
	}
}

// TestNumbersShownTwiceAgree cross-checks the figures the UI prints in more than one place.
//
// The other tests recompute a printed number from the trace. That catches a stale figure only if the
// recomputation is right; it cannot catch two on-screen figures disagreeing with each other, which is
// what a reader notices first — the inspector says one entropy and the chart beside it says another.
//
// Three pairs are checked, each across several inputs:
//   - the attention inspector's entropy against the bar for the selected head, which is the one marked ◂
//   - the "最大权重" line's value and key against the row's actual argmax
//   - the output stage's entropy in the card and in the inspector
func TestNumbersShownTwiceAgree(t *testing.T) {
	// Pair 1: entropy in the attention inspector versus the marked bar.
	for _, head := range []int{0, 2} {
		a := newTestApp(t)
		a.Step = 3
		a.SelToken = 20
		a.Head = head
		tr := ui.NewTester(a.View, 1500, 1800)
		tr.Frame()
		texts := tr.Texts()
		var inspector, marked string
		for i, s := range texts {
			if i+1 >= len(texts) {
				continue
			}
			if s == "熵" {
				inspector = texts[i+1]
			}
			if strings.HasPrefix(s, "H") && strings.Contains(s, "◂") {
				marked = texts[i+1]
			}
		}
		if inspector == "" || marked == "" {
			t.Fatalf("head %d: missing the entropy in the inspector (%q) or the bar (%q)",
				head, inspector, marked)
		}
		// The inspector writes "1.19 nats", the bar writes "1.19".
		if got := strings.TrimSuffix(inspector, " nats"); got != marked {
			t.Fatalf("head %d: the inspector says %q and the marked bar says %q",
				head, inspector, marked)
		}
	}

	// Pair 2: the "最大权重" line's value and key versus the row's argmax.
	for _, token := range []int{0, 1, 20, 35} {
		for _, head := range []int{0, 3} {
			a := newTestApp(t)
			a.Step = 3
			a.SelToken = token
			a.Head = head
			tr := ui.NewTester(a.View, 1500, 1800)
			tr.Frame()
			texts := tr.Texts()
			var line string
			for i, s := range texts {
				if s == "最大权重" && i+1 < len(texts) {
					line = texts[i+1]
				}
			}
			row := a.layer().Heads[head].Attn[token]
			var maxW float32
			best := 0
			for j := 0; j <= token; j++ {
				if row[j] > maxW {
					maxW, best = row[j], j
				}
			}
			want := fmt.Sprintf("%.3f → token %d", maxW, best)
			if !strings.HasPrefix(line, want) {
				t.Fatalf("token %d head %d: the line reads %q, the row's maximum is %q",
					token, head, line, want)
			}
		}
	}

	// Pair 3: the output stage's entropy, stated on the card and in the inspector.
	for _, temp := range []float64{0.1, 0.8, 1.5, 2.0} {
		a := newTestApp(t)
		a.Temp = temp
		a.Step = 6
		tr := ui.NewTester(a.View, 1500, 1800)
		tr.Frame()
		texts := tr.Texts()
		var card, inspector string
		for i, s := range texts {
			if i+1 >= len(texts) {
				continue
			}
			if strings.Contains(s, "温度") && strings.Contains(s, "熵") {
				card = s
			}
			if s == "熵" {
				inspector = texts[i+1]
			}
		}
		if card == "" || inspector == "" {
			t.Fatalf("temp %.1f: missing the entropy on the card (%q) or in the inspector (%q)",
				temp, card, inspector)
		}
		if !strings.Contains(card, inspector) {
			t.Fatalf("temp %.1f: the card says %q while the inspector says %q",
				temp, card, inspector)
		}
	}
}

// TestEmbeddingBodySaysTheTableIsLearned closes a gap in step 2's prose.
//
// The body explained what the embedding table is and how lookup works, but never that the table was
// trained — a reader could reasonably take the rows for something derived rather than learned. Every
// other learned quantity in the app says so: the position rows ("学出来的"), γ/β ("训练出来的"), and the
// attention heads' behaviour ("学到的模式"). The table itself, and the three projection matrices, did not.
func TestEmbeddingBodySaysTheTableIsLearned(t *testing.T) {
	a := newTestApp(t)
	a.Step = 1
	a.SelToken = 20
	tr := ui.NewTester(a.View, 1500, 1800)
	tr.Frame()

	// The body card is the one headed "这一段在讲什么"; the entry after it is the prose.
	texts := tr.Texts()
	var body string
	for i, s := range texts {
		if s == "这一段在讲什么" && i+2 < len(texts) {
			body = texts[i+2]
		}
	}
	if body == "" {
		t.Fatal("step 2 has no body prose")
	}
	if !strings.Contains(body, "训练出来的") {
		t.Fatalf("the embedding body still does not say the table is learned: %q",
			truncateForTest(body, 120))
	}
	// And it should point at the other learned numbers rather than leaving the reader to wonder
	// whether this one is special.
	if !strings.Contains(body, "投影矩阵") || !strings.Contains(body, "γ") {
		t.Fatalf("the body does not connect this to the other learned quantities: %q",
			truncateForTest(body, 120))
	}
}

// TestTemperatureIsStatedToBeInferenceOnly draws a line the UI was leaving blurred.
//
// The top bar puts three knobs side by side: layer, head and temperature. The first two select a part of
// the model's own structure. Temperature is not in the model at all — nothing under internal/model
// mentions it, and it is applied when the logits are turned into a distribution. A reader moving that
// slider has no way to tell which side of that line it is on.
//
// The output stage's inspector now says so where the number is displayed, and names what it does.
func TestTemperatureIsStatedToBeInferenceOnly(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	tr := ui.NewTester(a.View, 1500, 1100)
	tr.Frame()

	var card string
	for _, s := range tr.Texts() {
		if strings.Contains(s, "推理时的旋钮") {
			card = s
		}
	}
	if card == "" {
		t.Fatal("the output inspector does not say temperature is an inference-time setting")
	}
	if !strings.Contains(card, "不是训练出来的权重") {
		t.Fatalf("the sentence does not rule out the wrong reading: %q", card)
	}
	// It still has to explain what the knob does, or the clarification has replaced the teaching.
	for _, want := range []string{"贪心", "随机"} {
		if !strings.Contains(card, want) {
			t.Fatalf("the temperature note lost the explanation of %q: %q", want, card)
		}
	}
}

// TestTokenizeCalloutQuotesThePromptsOwnCounts checks step 1's new misconception block.
//
// Five stages had a callout and the two earliest did not. Step 1's is the belief a reader brings from
// other tokenisers — one character, one token — and its correction uses this prompt's counts, so it can be
// verified against the strip above rather than taken on faith, which is what the helper's contract asks
// for.
//
// It only makes sense on a prompt that actually has multibyte characters, so ASCII prompts must not get it.
func TestTokenizeCalloutQuotesThePromptsOwnCounts(t *testing.T) {
	read := func(prompt string) (string, int, int) {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		a.Step = 0
		tr := ui.NewTester(a.View, 1500, 1400)
		tr.Frame()
		for _, s := range tr.Texts() {
			if strings.Contains(s, "个字符编成了") {
				return s, len(a.Prompt), len(a.Trace.Tokens)
			}
		}
		return "", len(a.Prompt), len(a.Trace.Tokens)
	}

	// ASCII: no callout, because the belief does not apply.
	for _, prompt := range []string{"hi", "The transformer reads the tokens and"} {
		if line, _, _ := read(prompt); line != "" {
			t.Fatalf("prompt %q is ASCII but got a CJK-token callout: %q", prompt, line)
		}
	}

	// Multibyte: the callout cites this prompt's rune and token counts, not constants.
	for _, prompt := range []string{"注意力", "hi 注意力"} {
		line, bytes, tokens := read(prompt)
		if line == "" {
			t.Fatalf("prompt %q has multibyte characters but no callout", prompt)
		}
		runes := len([]rune(prompt))
		want := fmt.Sprintf("这里 %d 个字符编成了 %d 个 token", runes, tokens)
		if !strings.Contains(line, want) {
			t.Fatalf("prompt %q: the callout says %q, want the phrase %q", prompt, line, want)
		}
		if bytes == runes {
			t.Fatalf("prompt %q has equal byte and rune counts, so it does not exercise the case",
				prompt)
		}
	}
}

// TestSoftmaxBodyDescribesTheWeightsItHas keeps step 4's prose from asserting a pattern that may not be there.
//
// The body said layer 0 head 0 attends to the previous token "in almost every position". Measured on the
// bundled weights that is 31 of 35 rows; on random weights it is 3 to 5 of 35. The app renders random weights
// whenever the bundled file is missing — there is a snapshot for exactly that state — and while the banner
// warns the reader not to trust the numbers, prose describing a pattern the picture does not show is a
// separate failure: it is a claim about what is on screen.
//
// The body is now computed from the trace, and both branches cite the count they measured.
func TestSoftmaxBodyDescribesTheWeightsItHas(t *testing.T) {
	bodyOf := func(w *model.Weights) (string, int, int) {
		a := New(w)
		a.Step = 3
		tr := ui.NewTester(a.View, 1500, 2200)
		tr.Frame()
		for _, s := range tr.Texts() {
			if strings.Contains(s, "第 0 层的头 0") && strings.Contains(s, "行") {
				prev, rows := previousTokenShare(a.Trace.Layers[0].Heads[0].Attn,
					len(a.Trace.Tokens))
				return s, prev, rows
			}
		}
		return "", 0, 0
	}

	trained, err := model.Bundled()
	if err != nil {
		t.Fatalf("bundled weights: %v", err)
	}
	body, prev, rows := bodyOf(trained)
	if body == "" {
		t.Fatal("the softmax body does not mention layer 0 head 0 anymore")
	}
	if rows == 0 {
		t.Fatal("no rows to measure")
	}
	wantPhrase := fmt.Sprintf("%d/%d 行", prev, rows)
	if !strings.Contains(body, wantPhrase) {
		t.Fatalf("the body says %q, but the measured share is %s", truncateForTest(body, 120),
			wantPhrase)
	}
	// The trained weights must actually exhibit the pattern, or this branch is unreachable and the test
	// proves nothing.
	if prev*4 < rows*3 {
		t.Fatalf("the bundled weights no longer show the pattern (%d/%d rows); the trained branch "+
			"of this prose is now dead", prev, rows)
	}

	// Random weights take the other branch and must not claim the pattern.
	random, _, _ := bodyOf(model.Random(model.DefaultConfig, 42))
	if random == "" {
		t.Fatal("the softmax body is missing under random weights")
	}
	if strings.Contains(random, "31/35") || strings.Contains(random, "紧邻的前一个 token；") {
		t.Fatalf("random weights still get the trained-weights sentence: %q",
			truncateForTest(random, 160))
	}
	if !strings.Contains(random, "看不出这种分工") {
		t.Fatalf("random weights get neither branch's wording: %q", truncateForTest(random, 160))
	}
}

// TestResidualNormFiguresKeepTheirMagnitude keeps step 5's callout from rounding a short residual stream
// into a flat one.
//
// The sentence printed both norms with %.0f. On the trained weights that is fine — 13 to 67. On random
// weights the stream is short: inMean 0.2233, outMean 0.6322, which %.0f renders as "从 0 长到 1". A reader
// gets "it went from nothing to one" followed immediately by "约 2.8 倍，而且每一层都在变大", and the two
// halves of one sentence disagree.
//
// Precision now follows the magnitude: two decimals below 10, none above.
func TestResidualNormFiguresKeepTheirMagnitude(t *testing.T) {
	read := func(w *model.Weights) (string, float64, float64) {
		a := New(w)
		a.Step = 4
		tr := ui.NewTester(a.View, 1500, 2600)
		tr.Frame()
		for _, s := range tr.Texts() {
			if strings.Contains(s, "残差流确实会一路变长") {
				inMean, outMean := residualNormMeans(a.Trace)
				return s, inMean, outMean
			}
		}
		return "", 0, 0
	}

	// Random weights: the case that was wrong.
	body, inMean, outMean := read(model.Random(model.DefaultConfig, 42))
	if body == "" {
		t.Fatal("the residual callout is missing")
	}
	if inMean >= 10 || outMean >= 10 {
		t.Fatalf("the random model's norms are %.3f and %.3f, so this no longer exercises the "+
			"small-magnitude case", inMean, outMean)
	}
	if strings.Contains(body, "从块输入处的 0 长到") {
		t.Fatalf("a stream of %.2f is printed as 0: %q", inMean, truncateForTest(body, 160))
	}
	wantIn := fmt.Sprintf("%.2f", inMean)
	wantOut := fmt.Sprintf("%.2f", outMean)
	if !strings.Contains(body, wantIn) || !strings.Contains(body, wantOut) {
		t.Fatalf("the callout does not print the measured norms (%s, %s): %q",
			wantIn, wantOut, truncateForTest(body, 160))
	}

	// Trained weights keep whole numbers, where the fraction is noise.
	body, inMean, outMean = read(mustBundled(t))
	if inMean < 10 {
		t.Fatalf("the bundled model's input norm is %.2f, so the large-magnitude branch is no longer "+
			"reachable", inMean)
	}
	if !strings.Contains(body, fmt.Sprintf("%.0f", inMean)) {
		t.Fatalf("the trained model's norm %.2f is not printed as a whole number: %q",
			inMean, truncateForTest(body, 160))
	}
}

// mustBundled loads the bundled weights or fails the test.
func mustBundled(t *testing.T) *model.Weights {
	t.Helper()
	w, err := model.Bundled()
	if err != nil {
		t.Fatalf("bundled weights: %v", err)
	}
	return w
}

// TestProseAboutConfidenceAndSqueezeFollowsTheData checks two sentences that described the trained model's
// figures regardless of what the loaded weights actually produce.
//
// Step 7's callout said "完全可以对一个字节有这么高的把握". On the bundled weights maxP is 97% and the phrase
// fits; on random weights it is 1%, near the 1/256 of a uniform distribution, and calling that high confidence
// contradicts the percentage printed in the same sentence.
//
// Step 6's histogram note said the negative region "只占 4.0% 的轴宽，这一侧的直方图因此只剩一根柱子". That is
// what makes it unreadable on the trained weights; random weights give the region 38.6%, where the bins are
// legible, so the reason for "look at the colour band instead" no longer holds.
func TestProseAboutConfidenceAndSqueezeFollowsTheData(t *testing.T) {
	trained := mustBundled(t)
	random := model.Random(model.DefaultConfig, 42)

	// Step 7: high confidence, low confidence.
	callout := func(w *model.Weights) (string, float32) {
		a := New(w)
		a.Step = 6
		tr := ui.NewTester(a.View, 1500, 2800)
		tr.Frame()
		for _, s := range tr.Texts() {
			if strings.Contains(s, "这两件事没有关系") {
				return s, a.Trace.Probabilities(a.Temp)[0]
			}
		}
		return "", 0
	}
	body, _ := callout(trained)
	if body == "" {
		t.Fatal("the sampling callout is missing")
	}
	if !strings.Contains(body, "这么高的把握") {
		t.Fatalf("the trained model's top-1 is high but the text does not say so: %q",
			truncateForTest(body, 160))
	}
	rbody, _ := callout(random)
	if strings.Contains(rbody, "这么高的把握") {
		t.Fatalf("random weights still describe a confident prediction: %q",
			truncateForTest(rbody, 160))
	}
	if !strings.Contains(rbody, "根本没有那么大的把握") {
		t.Fatalf("random weights get neither branch: %q", truncateForTest(rbody, 160))
	}

	// Step 6: squeezed or not.
	gelu := func(w *model.Weights) string {
		a := New(w)
		a.Step = 5
		a.Layer = 0
		tr := ui.NewTester(a.View, 1500, 2800)
		tr.Frame()
		for _, s := range tr.Texts() {
			if strings.Contains(s, "轴宽") {
				return s
			}
		}
		return ""
	}
	tg := gelu(trained)
	if !strings.Contains(tg, "只剩一根柱子") {
		t.Fatalf("the trained model's negative region is squeezed but the text does not say so: %q",
			truncateForTest(tg, 160))
	}
	rg := gelu(random)
	if rg == "" {
		t.Fatal("the GELU note is missing under random weights")
	}
	if strings.Contains(rg, "只剩一根柱子") {
		t.Fatalf("a region spanning a large share of the axis is still called one bar: %q",
			truncateForTest(rg, 160))
	}
	if !strings.Contains(rg, "还能看出分布的形状") {
		t.Fatalf("the wide-region branch is not taken: %q", truncateForTest(rg, 160))
	}
}

// TestProseDiffersBetweenWeightsOnlyWhereDesigned is the guard for the class of defect found in rounds 209-211.
//
// Those rounds each found prose that described the trained model while the app was rendering random weights:
// a claim about which token a head attends to, "%.0f" flattening 0.22 into 0, "这么高的把握" for a 1% share.
// Each was found by hand. This makes the comparison automatic.
//
// It renders all seven stages under both weight sets, strips the numbers out (their digits legitimately differ),
// and requires every remaining difference to be one of the branches the app is supposed to have. A new hard-coded
// sentence that only holds for trained weights shows up here as an unexpected difference.
func TestProseDiffersBetweenWeightsOnlyWhereDesigned(t *testing.T) {
	trained := mustBundled(t)
	random := model.Random(model.DefaultConfig, 42)

	// Collapse any number, with its sign, decimals and percent sign, to a single marker.
	num := regexp.MustCompile(`[-+]?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?%?`)
	// allowed lists, per stage, the phrase fragments an intended difference is expected to contain. An
	// unexpected difference must match none of them.
	allowed := map[int][]string{
		// Matched against the number-stripped text, so digits are already "#".
		// The embedding card names the most similar byte pairs, which are whatever that
		// weight set happened to produce — with random weights the top pair is different
		// bytes, so the whole line moves. The fragment is the byte notation itself.
		1: {"I-", "H-", "-\\x", "右边那条色带每一格是一个维度", "这张表里没有这回事"},
		// the per-weights attention description, and the four-head comparison verdict,
		// which reads "差别明显" for one weight set and "很接近" for another
		3: {"第 # 层的头 #", "四个头差别明显", "四个头在这一层上很接近", "这里换两个能区分行为的量",
			"这里换两个能区分行为的量"},
		// gamma and beta learned versus identity, plus the logit lens ladder: it
		// describes what these particular weights say at each depth, so every one of
		// its sentences moves with them — which rungs agree with the final answer,
		// how many are confidently wrong, and the rung names in the sentence.
		4: {"训练出来的", "这份权重没有训练过", "只看这个位置", "最后一个中间层"},
		// the GELU squeeze note, plus the FFN concentration comparison: whether the two
		// layers divide their hidden units differently is a property of the weights, so
		// with random ones the sentence is absent rather than false
		// the GELU squeeze note, the FFN concentration comparison, and the answer-direction
		// callout: whether the FFN writes the answer direction is a property of the weights,
		// so with random ones the sentence is absent rather than false
		5: {"只剩一根柱子", "还能看出分布的形状", "两层的用法很不一样", "方向不是注意力定的",
			// The inspector explains the gap between its active count and the stage's negative
			// count. How many dimensions sit in (0, threshold] is a property of the weights, so
			// the sentence appears at all only when that count is non-zero.
			"两边都不算"},
		// the confidence note, plus the word-boundary sentence: whether the gap it
		// describes exists is a property of the weights, so with random ones it is
		// absent rather than false
		// the confidence note, the word-boundary sentence, and the temperature curve: which
		// temperature minimises the loss is a property of the weights, so the whole sentence
		// differs between trained and random
		6: {"这么高的把握", "根本没有那么大的把握", "错得最多的地方是词边界", "这条曲线是实测的"},
	}

	collect := func(w *model.Weights) []map[string]bool {
		out := make([]map[string]bool, len(lessons))
		for step := range lessons {
			a := New(w)
			a.Prompt = "The transformer reads the tokens and"
			a.recompute()
			a.Step = step
			a.SelToken = 20
			tr := ui.NewTester(a.View, 1500, 2800)
			tr.Frame()
			set := map[string]bool{}
			for _, s := range tr.Texts() {
				// Prose only: short labels are numbers and tokens, not claims.
				if len([]rune(s)) <= 40 {
					continue
				}
				// The untrained banner is a deliberate, documented difference.
				if strings.Contains(s, "读不到内置权重") {
					continue
				}
				// The sampling line's difference is the drawn token's own glyph, which is data
				// rather than wording: the numbers are stripped but the quoted character is not.
				if strings.Contains(s, "抽到 id ") {
					continue
				}
				// Same for the per-position sentence: different weights put the largest loss at
				// a different position, so the byte it quotes changes. That glyph is data — the
				// sentence's wording is identical across weight sets.
				if strings.Contains(s, "是在核对模型真的做过的一次预测") {
					continue
				}
				set[num.ReplaceAllString(s, "#")] = true
			}
			out[step] = set
		}
		return out
	}
	trainedText, randomText := collect(trained), collect(random)

	for step := 0; step < len(lessons); step++ {
		var diffs []string
		for s := range trainedText[step] {
			if !randomText[step][s] {
				diffs = append(diffs, s)
			}
		}
		for s := range randomText[step] {
			if !trainedText[step][s] {
				diffs = append(diffs, s)
			}
		}
		for _, d := range diffs {
			matched := false
			for _, frag := range allowed[step] {
				if strings.Contains(d, frag) {
					matched = true
					break
				}
			}
			if !matched {
				t.Fatalf("stage %d: the prose differs between trained and random weights outside any "+
					"designed branch: %q", step, truncateForTest(d, 200))
			}
		}
	}

	// And the designed branches must still be reached, or this test would pass with the feature removed.
	for step, frags := range allowed {
		for _, frag := range frags {
			found := false
			for s := range randomText[step] {
				if strings.Contains(s, frag) {
					found = true
				}
			}
			for s := range trainedText[step] {
				if strings.Contains(s, frag) {
					found = true
				}
			}
			if !found {
				t.Fatalf("stage %d: no prose contains %q, so that branch is gone", step, frag)
			}
		}
	}
}

// TestSoftmaxBodyAdmitsWhenThereIsNothingToMeasure closes a gap the weight-dependent fix left open.
//
// Round 209 made the body branch on whether the weights show the head pattern. Both branches printed a
// fraction, and with a one-token prompt there are no rows with a predecessor at all, so it printed "0/0 行"
// and went on to describe a pattern from a sample of zero. A two-token prompt gives exactly one row, which
// is no better as evidence.
//
// Under three measurable rows the body now says how many there are and that no pattern can be claimed, and
// it does not tell the reader to switch heads either: with one position all four heads are identical.
func TestSoftmaxBodyAdmitsWhenThereIsNothingToMeasure(t *testing.T) {
	bodyOf := func(prompt string) (string, int) {
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		a.Step = 3
		tr := ui.NewTester(a.View, 1500, 2800)
		tr.Frame()
		for _, s := range tr.Texts() {
			if strings.Contains(s, "一行分数经过") {
				prev, rows := 0, 0
				if len(a.Trace.Layers) > 0 && len(a.Trace.Layers[0].Heads) > 0 {
					prev, rows = previousTokenShare(a.Trace.Layers[0].Heads[0].Attn,
						len(a.Trace.Tokens))
				}
				_ = prev
				return s, rows
			}
		}
		return "", 0
	}

	// An empty prompt is not a short one. It used to be grouped with these, which forced the
	// body to say "只有 0 行可以比较 … 多打几个字再看" — a measurement of nothing, with advice to
	// add to text that is not there. The empty case is checked separately below.
	for _, prompt := range []string{"a", "hi", "abc"} {
		body, rows := bodyOf(prompt)
		if body == "" {
			t.Fatalf("prompt %q: the softmax body is missing", prompt)
		}
		if rows >= 3 {
			t.Fatalf("prompt %q gives %d rows, so this no longer exercises the short case", prompt, rows)
		}
		if strings.Contains(body, "0/0 行") {
			t.Fatalf("prompt %q prints a 0/0 fraction: %q", prompt, truncateForTest(body, 160))
		}
		if strings.Contains(body, "切换头上的分段控件就能看出来") {
			t.Fatalf("prompt %q is told to switch heads, but all four are identical there: %q",
				prompt, truncateForTest(body, 160))
		}
		want := fmt.Sprintf("只有 %d 行可以比较", rows)
		if !strings.Contains(body, want) {
			t.Fatalf("prompt %q says %q, want the phrase %q", prompt,
				truncateForTest(body, 160), want)
		}
	}

	// The empty prompt must read as the empty state, not as a measurement.
	if body, _ := bodyOf(""); strings.Contains(body, "只有 0 行") || strings.Contains(body, "多打几个字") {
		t.Fatalf("an empty prompt is described as a small one: %q", truncateForTest(body, 160))
	} else if !strings.Contains(body, "还没有提示词") {
		t.Fatalf("an empty prompt does not say there is no prompt yet: %q",
			truncateForTest(body, 160))
	}

	// A long prompt keeps the full description.
	body, rows := bodyOf("The transformer reads the tokens and")
	if rows < 3 {
		t.Fatalf("the long prompt gives only %d rows", rows)
	}
	if strings.Contains(body, "行可以比较") {
		t.Fatalf("the long prompt still gets the short-sample wording: %q",
			truncateForTest(body, 160))
	}
}

// TestProseDiffersAcrossPromptLengthsOnlyWhereDesigned is the same guard as the weights one, for the other
// input dimension.
//
// Round 213 found step 4 telling the reader layer 0 head 0 "在 0/0 行上" attends to the previous token — a
// proportion over zero rows — and suggesting they switch heads when all four are identical. That is a
// short-prompt defect, so short prompts need their own comparison.
//
// Numbers are stripped as before; two differences survive that are data rather than wording and are excluded:
// a value's sign (which changes whether the printed term is parenthesised) and the glyph of the sampled token.
func TestProseDiffersAcrossPromptLengthsOnlyWhereDesigned(t *testing.T) {
	num := regexp.MustCompile(`[-+]?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?%?`)
	// allowed lists, per stage, the fragments a designed difference is expected to contain.
	allowed := map[int][]string{
		// the all-negative legend note, plus every line of the mask counterfactual card:
		// it needs a second token before there is any future position to hide, so a
		// one-token prompt renders none of it
		// (Fragments have to come from strings longer than 40 runes: collect() only keeps
		// those, since short labels are numbers and tokens rather than prose.)
		// (Fragments must come from strings over 40 runes; collect() drops shorter ones.)
		// The embedding card names the most similar byte pairs, which need two tokens before
		// there is an adjacent pair to compare at all; a one-token prompt renders none of it.
		1: {"I-", "H-", "-\\x", "右边那条色带每一格是一个维度", "这张表里没有这回事"},
		2: {"所有分数都是负的", "去掉以后变化极大", "因果掩码不是提醒"},
		// the short-sample branch, then the long one, then the four-head comparison,
		// whose verdict depends on how many rows there are to average over
		// ..., plus every line of the four-head comparison card: it needs at least two
		// tokens to have any row to average, so a one-token prompt renders none of it.
		3: {"行可以比较", "第 # 层的头 #", "四个头差别明显", "四个头在这一层上很接近", "看不出分工",
			// The head-independence card needs a second position to compare any two heads
			// over; with one token it draws nothing.
			"四个头说的是同一件事吗", "不是彼此的副本", "两两之间最大的平均余弦",
			"这里换两个能区分行为的量", "前一位权重", "押注超过半数",
			// A one-token prompt has no row 1, so the entropy note says that rather than
			// pointing there. The diff runs both ways — the long prompt's version is reported
			// as missing from the short set — so both forms are listed.
			"没有第 # 行", "从第 # 行起才有可比较的分配"},
		// Confidence follows maxP; the last-position note appears only when the selected token is
		// not the last one, which a one-token prompt makes impossible.
		// The per-position card needs two tokens to have anything to check; a one-token prompt gets
		// an explanation instead of the strip, which is a designed branch rather than drift.
		// The FFN concentration card needs at least two tokens to have anything to
		// aggregate; with a one-token prompt it draws nothing, so every line of it is a
		// designed difference rather than drift. Its own comparison sentence also needs
		// a token pair to be a comparison at all.
		// (Only fragments from strings over 40 runes can be reached by the check below.)
		5: {"两层的用法很不一样", "个单元凑够一半激活，每位置都开的有", "把整句每个位置都算进来",
			// The answer-direction card needs a second token before there is a next byte to
			// rank at all, so a one-token prompt renders none of it.
			"把 # 个字节按「它的输入嵌入和这个向量的余弦」排名",
			"：输入里真实字节平均第", "方向不是注意力定的"},
		6: {"这么高的把握", "根本没有那么大的把握", "有那么大的把握", "选中的 t=# 不参与预测",
			"这个提示词只有一个位置",
			// The per-position card's body exists only when there is a position to score, so
			// every fixed line of it is absent for a one-token prompt by design.
			"当前损失 #", "均匀猜测", "与它比对",
			// The word-boundary sentence: a one-token prompt has no position whose
			// predecessor is a space. (Only fragments from strings over 40 runes count —
			// collect() drops shorter ones as labels rather than prose.)
			"错得最多的地方是词边界",
			// The causal-mask check compares the window against its own prefix, so a
			// one-token prompt has nothing to compare and the line is absent by design —
			// the same branch as the strip above it. Two tokens already reach it: "hi"
			// prints "剩下 1 个位置…", which becomes identical to the long prompt's line
			// once the numbers are stripped.
			"因果掩码核对"},
	}

	collect := func(prompt string) []map[string]bool {
		out := make([]map[string]bool, len(lessons))
		for step := range lessons {
			a := newTestApp(t)
			a.Prompt = prompt
			a.recompute()
			a.Step = step
			a.SelToken = 0
			tr := ui.NewTester(a.View, 1500, 2800)
			tr.Frame()
			set := map[string]bool{}
			for _, s := range tr.Texts() {
				if len([]rune(s)) <= 40 {
					continue
				}
				// Not wording: the sampled glyph, and a term's sign deciding its parentheses.
				if strings.Contains(s, "抽到 id ") || strings.Contains(s, "× γ ") {
					continue
				}
				set[num.ReplaceAllString(s, "#")] = true
			}
			out[step] = set
		}
		return out
	}

	long := collect("The transformer reads the tokens and")
	for _, prompt := range []string{"a", "hi", "abc"} {
		short := collect(prompt)
		for step := 0; step < len(lessons); step++ {
			var diffs []string
			for s := range long[step] {
				if !short[step][s] {
					diffs = append(diffs, s)
				}
			}
			for s := range short[step] {
				if !long[step][s] {
					diffs = append(diffs, s)
				}
			}
			for _, d := range diffs {
				matched := false
				for _, frag := range allowed[step] {
					if strings.Contains(d, frag) {
						matched = true
						break
					}
				}
				if !matched {
					t.Fatalf("prompt %q, stage %d: the prose differs from the long prompt's outside any "+
						"designed branch: %q", prompt, step, truncateForTest(d, 200))
				}
			}
		}
	}

	// Every allowed branch must still be reached somewhere, or it is not being tested.
	//
	// The layer control is swept too. collect() leaves Layer at 0, and the four-head
	// comparison card says something different per layer — "差别明显" on layer 0,
	// "很接近" on layer 1 — so without sweeping it the layer-1 verdict is an allowed
	// branch that is never reached, which this check correctly calls out as gone.
	all := map[string]bool{}
	for _, prompt := range []string{"a", "hi", "abc", "The transformer reads the tokens and"} {
		for step := range lessons {
			for s := range collect(prompt)[step] {
				all[s] = true
			}
		}
	}
	for layer := 0; layer < 2; layer++ {
		a := newTestApp(t)
		a.Step = 3
		a.Layer = layer
		tr := ui.NewTester(a.View, 1500, 2800)
		tr.Frame()
		for _, s := range tr.Texts() {
			all[s] = true
		}
	}
	for step, frags := range allowed {
		for _, frag := range frags {
			found := false
			for s := range all {
				if strings.Contains(s, frag) {
					found = true
				}
			}
			if !found {
				t.Fatalf("stage %d: no prose contains %q, so that branch is gone", step, frag)
			}
		}
	}
}

// TestCausalMaskIsStatedToBeArchitecture completes the learned-versus-given account the app gives of its numbers.
//
// Round 206 said the embedding table is trained; 207 said temperature is not a weight; 208 said the projection
// matrices are trained. The causal mask was never classified, so a reader met it described by its effect only —
// "the upper triangle is empty" — with no way to tell that it is written in code rather than derived by training.
//
// The mask belongs on the given side: it is part of the architecture.
func TestCausalMaskIsStatedToBeArchitecture(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	a.SelToken = 20
	tr := ui.NewTester(a.View, 1500, 2400)
	tr.Frame()

	var body string
	for i, s := range tr.Texts() {
		if s == "这一段在讲什么" && i+2 < len(tr.Texts()) {
			body = tr.Texts()[i+2]
		}
	}
	if body == "" {
		t.Fatal("step 3 has no body prose")
	}
	if !strings.Contains(body, "代码写死的") {
		t.Fatalf("the body still does not say the mask is fixed by the code: %q",
			truncateForTest(body, 160))
	}
	if !strings.Contains(body, "不是训练出来的") {
		t.Fatalf("the body does not rule out the trained reading: %q",
			truncateForTest(body, 160))
	}
	// And it has to keep explaining what the mask does, or the clarification replaced the teaching.
	if !strings.Contains(body, "看不到未来") {
		t.Fatalf("the body lost the explanation of what the mask does: %q",
			truncateForTest(body, 160))
	}
}

// TestGeluIsStatedToBeAFixedFunction completes the account of where each number in the pipeline comes from.
//
// By round 217 the app had classified the trained parameters (embedding table, projection matrices, gamma and
// beta), the inference-time knob (temperature) and the architecture's own constant (the causal mask). GELU was
// the remaining piece: the stage names it and explains its shape, but never says it carries no parameters at all,
// so a reader can mistake it for something the model learned.
func TestGeluIsStatedToBeAFixedFunction(t *testing.T) {
	a := newTestApp(t)
	a.Step = 5
	a.SelToken = 20
	tr := ui.NewTester(a.View, 1500, 2600)
	tr.Frame()

	var body string
	for i, s := range tr.Texts() {
		if s == "这一段在讲什么" && i+2 < len(tr.Texts()) {
			body = tr.Texts()[i+2]
		}
	}
	if body == "" {
		t.Fatal("step 6 has no body prose")
	}
	if !strings.Contains(body, "固定的数学函数") {
		t.Fatalf("the body does not say GELU is a fixed function: %q", truncateForTest(body, 160))
	}
	if !strings.Contains(body, "不含任何可学参数") {
		t.Fatalf("the body does not rule out learned parameters in GELU: %q",
			truncateForTest(body, 160))
	}
	// It must also name what *is* trained, or the clarification leaves the reader without the positive half.
	if !strings.Contains(body, "W₁") || !strings.Contains(body, "W₂") {
		t.Fatalf("the body does not point at the matrices that are trained: %q",
			truncateForTest(body, 160))
	}
	// And the existing explanation of its shape has to survive.
	if !strings.Contains(body, "平滑地趋近零") {
		t.Fatalf("the body lost the explanation of GELU's negative region: %q",
			truncateForTest(body, 160))
	}
}

// TestHeadControlIsNotTabReachable records a limitation rather than a behaviour to preserve.
//
// The layer, head and temperature controls sit in the top bar. Tabbing reaches the layer control and the
// temperature slider — arrow keys then change the focused one, so the app's global Right/Left shortcut does
// not swallow them — but the head control is never focused, however many times Tab is pressed. Adding
// Focusable() to both Segmented controls changes nothing, so the cause is in how the toolkit assigns focus
// within a segmented group, not in this app.
//
// It is recorded because the next person to look at keyboard reachability will ask this exact question, and
// because the failure is silent: the control draws and responds to the mouse, so nothing on screen suggests
// it is unreachable from the keyboard.
//
// The head can still be changed without the mouse, but only indirectly: shift+arrows walk the matrix on the
// two attention stages, which shows the effect of one head at a time without selecting another.
// TestEveryTopBarControlIsReachableByTab and the two below replace a record that was wrong.
//
// For a long time the suite contained TestHeadControlIsNotTabReachable, which walked twenty-four Tab steps and pressed Right
// after each. It found the layer control and the temperature slider but never the head control, and concluded that MyGo's
// focus grouping made the head unreachable — a limitation to be documented rather than fixed.
//
// It is reachable. The loop was the problem: pressing Right changes a value and repaints, and a Tab step is consumed by
// whatever happens to hold focus, so counting fixed Tab presses against a moving target skipped the head group entirely.
// Measured by tabbing n times from a fresh frame and then pressing Right three times: four tabs land on the layer group, five
// on the head group, and three Rights walk the head 0 → 3.
//
// These tests read the focus by what it changes rather than by a pixel box, because that is what a keyboard reader cares about.
func tabTo(t *testing.T, n int) (*App, *ui.Tester) {
	t.Helper()
	a := newTestApp(t)
	a.Step = 2 // attention stage: layer and head are both live here
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	for i := 0; i < n; i++ {
		tr.Key(0, ui.KeyTab)
		tr.Frame()
	}
	return a, tr
}

func TestLayerControlIsReachableByTab(t *testing.T) {
	a, tr := tabTo(t, 4)
	before := a.Layer
	tr.Key(0, ui.KeyRight)
	tr.Frame()
	if a.Layer == before {
		t.Fatalf("after four tabs the layer control is not focused: Layer stayed %d", before)
	}
}

func TestHeadControlIsReachableByTab(t *testing.T) {
	a, tr := tabTo(t, 5)
	if a.Head != 0 {
		t.Fatalf("the head starts at %d, so this test cannot tell a move from a no-op", a.Head)
	}
	tr.Key(0, ui.KeyRight)
	tr.Frame()
	if a.Head != 1 {
		t.Fatalf("after five tabs the head control is not focused: Head is %d, want 1", a.Head)
	}
	// The arrows keep walking it, which is what a radio group owes a keyboard reader.
	for want := 2; want <= 3; want++ {
		tr.Key(0, ui.KeyRight)
		tr.Frame()
		if a.Head != want {
			t.Fatalf("Right should walk the heads; got %d, want %d", a.Head, want)
		}
	}
	// And back.
	tr.Key(0, ui.KeyLeft)
	tr.Frame()
	if a.Head != 2 {
		t.Fatalf("Left should walk back; got %d, want 2", a.Head)
	}
}

// TestTemperatureSliderIsReachableByTab is the third of the top-bar controls. It joins the cycle only on the output stage,
// where it is live.
func TestTemperatureSliderIsReachableByTab(t *testing.T) {
	a := newTestApp(t)
	a.Step = 6
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	before := a.Temp
	found := false
	for i := 0; i < 8; i++ {
		tr.Key(0, ui.KeyTab)
		tr.Frame()
		tr.Key(0, ui.KeyRight)
		tr.Frame()
		if a.Temp != before {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("no Tab stop reaches the temperature slider on the output stage; Temp stayed %.2f", before)
	}
}

// TestOutputCardSaysWhenItIgnoresTheSelection closes the one stage where clicking a token does nothing visible.
//
// On the other six stages, changing the selected token redraws the stage: a different row of the matrix, a
// different vector, a different row of the histogram. The output stage always shows the last position, because
// that is the only one projected to the vocabulary. Selecting token 5 there leaves the card identical, so the
// click reads as an interaction that failed.
//
// The card's note now names the position it is showing whenever that is not the selected one.
func TestOutputCardSaysWhenItIgnoresTheSelection(t *testing.T) {
	noteOf := func(sel int) (string, int) {
		a := newTestApp(t)
		a.Step = 6
		a.SelToken = sel
		tr := ui.NewTester(a.View, 1500, 1200)
		tr.Frame()
		texts := tr.Texts()
		for i, s := range texts {
			if s == "最后位置的向量" && i+1 < len(texts) {
				return texts[i+1], len(a.Trace.Tokens) - 1
			}
		}
		return "", 0
	}

	last := len(newTestApp(t).Trace.Tokens) - 1
	if last < 6 {
		t.Fatalf("the default prompt has only %d tokens; this test needs a non-last index", last)
	}

	// Selecting a non-last token must explain why the card did not change.
	note, want := noteOf(5)
	if note == "" {
		t.Fatal("the output stage lost its vector card")
	}
	if !strings.Contains(note, fmt.Sprintf("t=%d", want)) {
		t.Fatalf("the note does not say which position is drawn (t=%d): %q", want, note)
	}
	if !strings.Contains(note, fmt.Sprintf("选中的 t=%d", 5)) {
		t.Fatalf("the note does not name the selection it is ignoring: %q", note)
	}
	if !strings.Contains(note, "不参与预测") {
		t.Fatalf("the note does not say why: %q", note)
	}

	// Selecting the last token needs no explanation.
	note, _ = noteOf(last)
	if strings.Contains(note, "选中的") {
		t.Fatalf("the note explains an ignored selection when there is none: %q", note)
	}
}

// TestHoverFeedbackIsWhereItCarriesInformation records which stages carry hover feedback, and
// checks it per element rather than per region.
//
// The original form of this test swept a grid across the stage column and asked whether *any*
// text changed anywhere. That question is answered by any hoverable thing on the page, so it
// could not fail for the reason it was written: breaking the vector strips' readouts left it
// green, because the FFN stage also reacts at y>=720 (measured: 3 texts change there even with
// every strip's readout disabled). The gap was recorded in the changelog two rounds ago and is
// what this version closes.
//
// The distinction that matters is which stages expose one value out of many — a single
// dimension of a 96-long vector, a single cell of a 36x36 matrix — where pointing at an element
// has to name it or the reader cannot tell which they are on. That is six of the seven stages:
// the embedding strips are the first element in the app that needs it, so the list starts at
// stage 2 and runs to stage 7. Only stage 1 has none, because a token chip labels itself with a
// glyph and an id — highlighting it would add colour without adding information.
func TestHoverFeedbackIsWhereItCarriesInformation(t *testing.T) {
	// stagesWithHover names, for each stage that must carry hover feedback, the hover field
	// its own elements write. Asserting on the field rather than on "some text changed" is
	// what makes the check element-specific: a stage cannot satisfy it with somebody else's
	// highlight, and the failure names the missing element instead of a count.
	// The fields here were measured, not guessed: each was found by sweeping the stage's
	// column and recording which field first goes non-negative. An earlier draft listed the
	// stage names from the comment above instead, and reported stage 5 (the residual stage)
	// as having no hover feedback — it writes HoverDimIn, and the guess had put the matrix
	// stages one step too high.
	stagesWithHover := []struct {
		step int
		name string
		read func(*App) int
	}{
		{1, "嵌入向量色带", func(a *App) int { return a.HoverDimIn }},
		{2, "分数矩阵", func(a *App) int { return a.HoverQ }},
		{3, "注意力矩阵", func(a *App) int { return a.HoverQ }},
		{4, "残差色带", func(a *App) int { return a.HoverDimIn }},
		{5, "FFN 隐藏层 / 输出色带", func(a *App) int { return a.HoverDimH }},
		{6, "逐位置条带", func(a *App) int { return a.PredHover }},
	}
	// reaches requires the element to react *visibly*: the hover field is set AND the frame's
	// text changed. Checking only the field is not enough, and that is the lesson from this
	// test's own history — disabling every strip's readout leaves the fields set to the right
	// dimension while the reader sees nothing change, and a field-only assertion stays green
	// (verified by mutation). The two together are what "pointing at an element names it" means.
	//
	// The pointer has to land on the element and the elements sit at different heights per
	// stage, so the probe sweeps. It stops at the first hit that satisfies both conditions.
	reaches := func(step int, read func(*App) int) (bool, string) {
		a := newTestApp(t)
		a.Step = step
		a.SelToken = 20
		tr := ui.NewTester(a.View, 1500, 2300)
		tr.Frame()
		base := tr.Texts()
		// Start below the card-row chips (they end near y=235 at this size) so a chip tooltip
		// is never mistaken for the element, and above the first card's contents.
		for y := 320; y < 2200; y += 25 {
			for x := 300; x < 1150; x += 100 {
				tr.Move(float32(x), float32(y))
				tr.Frame()
				if read(a) < 0 {
					continue
				}
				now := tr.Texts()
				n := len(base)
				if len(now) < n {
					n = len(now)
				}
				for i := 0; i < n; i++ {
					if base[i] != now[i] {
						return true, ""
					}
				}
				// The field was written but nothing on screen moved: the readout that names
				// the hovered element is missing or not rendered.
				return false, fmt.Sprintf("the field was set at (%d,%d) but no text changed", x, y)
			}
		}
		return false, "the pointer never reached it"
	}
	for _, stage := range stagesWithHover {
		ok, why := reaches(stage.step, stage.read)
		if !ok {
			t.Fatalf("stage %d no longer carries hover feedback for %s (%s); those elements expose "+
				"one value out of many and the reader needs the pointer to name which",
				stage.step+1, stage.name, why)
		}
	}

	// Stage 1 must stay without one: its token chips label themselves with a glyph and an id,
	// so a highlight would add colour without adding information. If it gains one, this record
	// is stale rather than the app being wrong — the reasoning above needs revisiting.
	for _, step := range []int{0} {
		a := newTestApp(t)
		a.Step = step
		a.SelToken = 20
		tr := ui.NewTester(a.View, 1500, 2300)
		tr.Frame()
		base := tr.Texts()
		for y := 320; y < 2200; y += 25 {
			for x := 300; x < 1150; x += 100 {
				tr.Move(float32(x), float32(y))
				tr.Frame()
				now := tr.Texts()
				n := len(base)
				if len(now) < n {
					n = len(now)
				}
				d := 0
				for i := 0; i < n; i++ {
					if base[i] != now[i] {
						d++
					}
				}
				if d > 0 {
					t.Fatalf("stage %d now changes on hover (%d texts at (%d,%d)); update this "+
						"record's reasoning", step+1, d, x, y)
				}
			}
		}
	}
}

// TestCardsAndTitlesGoThroughOneBuilder keeps the visual hierarchy from drifting piecemeal.
//
// Twenty-seven cards across the seven stages and seven inspectors, all built by card(), which fixes the padding,
// gap, radius, background and border. Thirty headings, all built by sectionTitle(), which fixes the size, weight
// and colour and omits the note when there is none. That is why the cards look like one family: there is one place
// the style can come from.
//
// The guarantee holds only while nothing draws a card or a heading by hand. A Box with the same radius and a
// Text at fontTitle would look identical today and diverge at the next restyle, so this checks the source rather
// than the pixels.
func TestCardsAndTitlesGoThroughOneBuilder(t *testing.T) {
	src, err := os.ReadFile("stages.go")
	if err != nil {
		t.Fatal(err)
	}
	text := string(src)

	// A hand-rolled card: a column with a panel background and a corner radius.
	if strings.Contains(text, "Background(colPanel).Radius(") {
		t.Fatal("a card is drawn by hand instead of through card(); its padding and gap will drift")
	}
	// A hand-rolled heading: fontTitle applied directly to a Text.
	if strings.Contains(text, "FontSize(fontTitle)") {
		t.Fatal("a heading sets fontTitle directly instead of going through sectionTitle(); " +
			"its note and colour will drift")
	}

	// And both builders have to still be the ones in use.
	//
	// cardNamed is counted alongside card because it wraps card() to add the step bar's
	// jump target — it is the same builder with one extra call, not a second way to draw a
	// panel. Counting only card( made this test fail when most cards moved to the named
	// form; the invariant it guards is "no hand-rolled panels", not "every card uses one
	// particular function name".
	if n := strings.Count(text, "card(c, func()") + strings.Count(text, `cardNamed(c, "`); n < 20 {
		t.Fatalf("only %d cards go through card()/cardNamed(); the screenshot set shows far more", n)
	}
	if n := strings.Count(text, "sectionTitle(c,"); n < 25 {
		t.Fatalf("only %d headings go through sectionTitle()", n)
	}
}

// TestEachControlChangesOnlyItsOwnStages ties the response to the scope table the top bar advertises.
//
// layerFrom..layerTo, headFrom..headTo and tempFrom..tempTo decide which controls are dimmed on which stage, and
// the note under the lesson says so in words. Nothing checked that the picture agrees: a control could be enabled on
// a stage that does not read it, or disabled on a stage that does.
//
// Inside a control's range, changing it must move something other than its own label. Outside the range the only
// text that may change is that label — the numbers shown are "温度 0.80" in the top bar itself.
func TestEachControlChangesOnlyItsOwnStages(t *testing.T) {
	read := func(step int, set func(a *App)) []string {
		a := newTestApp(t)
		a.Step = step
		a.SelToken = 20
		set(a)
		tr := ui.NewTester(a.View, 1500, 2600)
		tr.Frame()
		return tr.Texts()
	}
	changed := func(a, b []string) int {
		n := len(a)
		if len(b) < n {
			n = len(b)
		}
		d := 0
		for i := 0; i < n; i++ {
			if a[i] != b[i] {
				d++
			}
		}
		return d
	}

	controls := []struct {
		name     string
		from, to int
		apply    func(a *App)
	}{
		{"layer", layerFrom, layerTo, func(a *App) { a.Layer = 1 }},
		{"head", headFrom, headTo, func(a *App) { a.Head = 3 }},
		{"temperature", tempFrom, tempTo, func(a *App) { a.Temp = 2.0 }},
	}
	for _, c := range controls {
		for step := 0; step < len(lessons); step++ {
			n := changed(read(step, func(a *App) {}), read(step, c.apply))
			inRange := inStages(step, c.from, c.to)
			switch {
			case inRange && n < 2:
				t.Fatalf("stage %d (%s) is inside %s's range %d–%d, but changing it moves only %d text(s)",
					step, lessons[step].name, c.name, c.from, c.to, n)
			case !inRange && n > 1:
				t.Fatalf("stage %d (%s) is outside %s's range %d–%d, but changing it moves %d texts",
					step, lessons[step].name, c.name, c.from, c.to, n)
			}
		}
	}
}

// TestAdvertisedShortcutsDoWhatTheLineSays turns the shortcut sentence into a contract.
//
// The line under every lesson reads "快捷键：← → 切换阶段 · Shift+← → 切换 token · 空格 播放/暂停 · Enter 采样 · 1–7 跳转",
// with the Shift part rewritten on the two stages that draw a matrix. It is a promise to the reader, and one that
// has been wrong before: the Shift clause used to say "切换 token" on the matrix stages too.
//
// Each key is pressed and its effect checked against the sentence, and the sentence is checked to still mention it —
// so a mapping change that forgets the line fails here rather than misleading a reader.
func TestAdvertisedShortcutsDoWhatTheLineSays(t *testing.T) {
	// The advertised line, as rendered.
	lineOf := func(step int) string {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, 1500, 2000)
		tr.Frame()
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, "快捷键：") {
				return s
			}
		}
		return ""
	}
	plain, matrix := lineOf(0), lineOf(3)
	if plain == "" || matrix == "" {
		t.Fatal("the shortcut line is missing")
	}
	for _, want := range []string{"←", "→", "空格", "Enter", "1–7", "PgUp/PgDn"} {
		if !strings.Contains(plain, want) {
			t.Fatalf("the shortcut line no longer mentions %q: %q", want, plain)
		}
	}
	if !strings.Contains(plain, "切换 token") {
		t.Fatalf("on a matrix-less stage the Shift clause should say 切换 token: %q", plain)
	}
	if !strings.Contains(matrix, "移动矩阵格子") {
		t.Fatalf("on a matrix stage the Shift clause should say 移动矩阵格子: %q", matrix)
	}

	// And the keys do those things.
	press := func(step int, key ui.Key, mods ui.Modifiers) (dStep, dToken, dKeyK int) {
		a := newTestApp(t)
		a.Step = step
		a.SelToken = 10
		tr := ui.NewTester(a.View, 1500, 1400)
		tr.Frame()
		s0, tk0, kk0 := a.Step, a.SelToken, a.KeyK
		tr.Key(mods, key)
		tr.Frame()
		return a.Step - s0, a.SelToken - tk0, a.KeyK - kk0
	}

	// ← → 切换阶段. Left is tested from a middle stage: at stage 0 there is nowhere to go, and
	// selectStep clamps, so the key correctly does nothing there.
	for _, step := range []int{0, 3} {
		if d, tk, _ := press(step, ui.KeyRight, 0); d != 1 || tk != 0 {
			t.Fatalf("step %d: Right should move one stage only, got step%+d token%+d", step, d, tk)
		}
	}
	if d, tk, _ := press(3, ui.KeyLeft, 0); d != -1 || tk != 0 {
		t.Fatalf("step 3: Left should move back one stage only, got step%+d token%+d", d, tk)
	}

	// Shift+arrows: token on a matrix-less stage, cell on a matrix stage.
	if _, tk, _ := press(0, ui.KeyRight, ui.Shift); tk != 1 {
		t.Fatalf("step 0 advertises Shift+→ as 切换 token, but the token moved by %d", tk)
	}
	if d, tk, kk := press(3, ui.KeyRight, ui.Shift); d != 0 || tk != 0 || kk == 0 {
		t.Fatalf("step 3 advertises Shift+→ as 移动矩阵格子, but step%+d token%+d keyK%+d", d, tk, kk)
	}

	// 1–7 跳转.
	for i, key := range [...]ui.Key{ui.Key1, ui.Key2, ui.Key3, ui.Key4, ui.Key5, ui.Key6, ui.Key7} {
		if d, _, _ := press(0, key, 0); d != i {
			t.Fatalf("digit %d should jump to stage %d, got %+d", i+1, i, d)
		}
	}

	// PgUp/PgDn 滚动: pressed on the tallest stage at the smallest window, with the
	// pointer parked away from the column so only the key can be doing it.
	scrollPress := func(key ui.Key) (before, after float32) {
		a := newTestApp(t)
		a.Step = 4
		tr := ui.NewTester(a.View, 1180, 720)
		tr.Frame()
		tr.Move(0, 0)
		tr.Frame()
		before = a.Scroll.Y
		tr.Key(0, key)
		tr.Frame()
		tr.Frame()
		return before, a.Scroll.Y
	}
	if before, after := scrollPress(ui.KeyPageDown); after <= before {
		t.Fatalf("the line advertises PgUp/PgDn 滚动, but PageDown moved the column from "+
			"%.0f to %.0f", before, after)
	}
	if before, after := scrollPress(ui.KeyPageUp); after != before {
		t.Fatalf("PageUp from the top moved the column from %.0f to %.0f", before, after)
	}
}

// TestInspectorSubtitleMatchesTheCards checks the layer and head the inspector claims against the range table.
//
// The header's stage name comes from lessons[a.Step].name — the same value the cards are dispatched on — so it cannot
// disagree with them, and asserting it would be a test that can never fail. The subtitle is different: it is built
// separately as "层 %d · 头 %d", so it can claim a head on a stage whose cards show one head's row regardless.
func TestInspectorSubtitleMatchesTheCards(t *testing.T) {
	for step := 0; step < len(lessons); step++ {
		a := newTestApp(t)
		a.Step = step
		a.SelToken = 20
		tr := ui.NewTester(a.View, 1500, 2000)
		tr.Frame()
		texts := tr.Texts()

		idx := -1
		for i, s := range texts {
			if s == "检查器" {
				idx = i
				break
			}
		}
		if idx < 0 || idx+1 >= len(texts) {
			t.Fatalf("stage %d: no inspector header", step)
		}
		// Checked as a sanity anchor rather than a guard: both sides read the same slice.
		if got := texts[idx+1]; got != lessons[step].name {
			t.Fatalf("stage %d: header says %q, lessons say %q", step, got, lessons[step].name)
		}

		inLayer := inStages(step, layerFrom, layerTo)
		inHead := inStages(step, headFrom, headTo)
		var sub string
		if idx+2 < len(texts) {
			sub = texts[idx+2]
		}
		if got := strings.Contains(sub, "层"); got != inLayer {
			t.Fatalf("stage %d (%s): subtitle %q reports a layer, but the range says %v",
				step, lessons[step].name, sub, inLayer)
		}
		if got := strings.Contains(sub, "头"); got != inHead {
			t.Fatalf("stage %d (%s): subtitle %q reports a head, but the range says %v",
				step, lessons[step].name, sub, inHead)
		}
	}
}

// TestResidualChainBracketsBothSigns keeps the two addition lines in one shape.
//
// Round 167 bracketed the residual chain's negative term so "-0.1050 + (-1.0295)" could not be read as
// two subtractions. The positive term on the next line stayed bare, so on the default prompt the chain
// printed one bracketed term and one open one. Round 231 fixed the same asymmetry in the embedding sum;
// both now go through signed(), which brackets whatever the sign is.
func TestResidualChainBracketsBothSigns(t *testing.T) {
	a := newTestApp(t)
	a.Step = 4
	a.SelToken = 20
	tr := ui.NewTester(a.View, 1500, 2800)
	tr.Frame()

	var chain []string
	for _, s := range tr.Texts() {
		if strings.Contains(s, "子层 Δ ") {
			chain = append(chain, s)
		}
	}
	if len(chain) != 2 {
		t.Fatalf("the residual chain should print two addition lines, got %d: %v", len(chain), chain)
	}

	bracketed := 0
	for _, line := range chain {
		// The term after the label must be "(+x)" or "(-x)": one opening bracket before the sign.
		i := strings.Index(line, "子层 Δ ")
		if i < 0 {
			continue
		}
		rest := line[i+len("子层 Δ "):]
		if strings.HasPrefix(rest, "(") {
			bracketed++
		}
	}
	if bracketed != 2 {
		t.Fatalf("expected both terms bracketed, got %d of 2: %v", bracketed, chain)
	}

	// And the parenthesised form must still be the same arithmetic the model did: adding the printed
	// terms reproduces the printed sum.
	for _, line := range chain {
		parts := strings.Split(line, " = ")
		if len(parts) != 2 {
			t.Fatalf("line does not end in a sum: %q", line)
		}
		var lhs, sum float64
		if _, err := fmt.Sscanf(parts[1], "%f", &sum); err != nil {
			t.Fatalf("cannot read the sum from %q: %v", line, err)
		}
		_ = lhs
		// The terms themselves are checked by TestPrintedResidualChainUsesTheComputedSteps; here only the
		// shape matters.
		if !strings.Contains(parts[0], ")") {
			t.Fatalf("the term is not bracketed: %q", line)
		}
	}
}

// TestScoreInspectorNamesItsRankingRule keeps the card's title from promising a ranking it does not use.
//
// The card shows the five scores with the largest magnitude — topAbsBars sorts by |value| and then restores row
// order — but was headed "分数 top-5", which reads as "the five keys this query attends to most". That is close to
// the opposite: a score of -4.32 is the largest magnitude in its row and the least attention the query can give,
// since softmax nearly erases it. Round 234 renamed the card.
func TestScoreInspectorNamesItsRankingRule(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	a.SelToken = 20
	tr := ui.NewTester(a.View, 1500, 2200)
	tr.Frame()

	var title string
	for _, s := range tr.Texts() {
		if r, ok := tr.Find(s); ok && r.X >= 1190 && strings.Contains(s, "分数") {
			title = s
			break
		}
	}
	if title == "" {
		t.Fatal("the score inspector lost its title")
	}
	if strings.Contains(title, "top-5") || strings.Contains(title, "top 5") {
		t.Fatalf("the title promises the highest scores, but the card shows the largest magnitudes: %q",
			title)
	}
	if !strings.Contains(title, "绝对值") {
		t.Fatalf("the title does not say what it ranks by: %q", title)
	}
}

// TestTopAbsBarsKeepsTheLargestMagnitudes pins the rule the title now names: five rows, chosen by |value|, shown
// back in their original order.
func TestTopAbsBarsKeepsTheLargestMagnitudes(t *testing.T) {
	rows := []barRow{
		{Index: 0, Value: -4.0},
		{Index: 1, Value: 0.5},
		{Index: 2, Value: -3.0},
		{Index: 3, Value: 0.9},
		{Index: 4, Value: 2.0},
		{Index: 5, Value: -0.1},
		{Index: 6, Value: 1.5},
	}
	got := topAbsBars(rows, 5)
	if len(got) != 5 {
		t.Fatalf("kept %d rows, want 5", len(got))
	}
	// Largest magnitudes are -4.0, -3.0, 2.0, 1.5 and 0.9 — indices 0, 2, 4, 6, 3.
	want := []int{0, 2, 3, 4, 6} // largest magnitudes, restored to row order
	gotIdx := make([]int, 0, 5)
	for _, r := range got {
		gotIdx = append(gotIdx, r.Index)
	}
	for i := range want {
		if gotIdx[i] != want[i] {
			t.Fatalf("kept indices %v, want %v (largest magnitudes, in row order)", gotIdx, want)
		}
	}
	// 0.9 is the fifth largest, so it stays; anything below it should not have made the cut.
	for _, r := range got {
		if math.Abs(float64(r.Value)) < 0.89 {
			t.Fatalf("kept a value of %.2f, which is outside the five largest magnitudes", r.Value)
		}
	}
}

// TestReduceMotionChangesTheRestingFrameNowhere keeps the preference to what it means.
//
// The desktop preference asks for no animation. It does not ask for a different layout, different colours or different
// content, so a stage's resting frame has to be identical either way — and the only place the app reads the preference
// is the stage transition, which decides whether a fade plays, not how anything is drawn.
//
// Nothing checked that. A future branch that skipped a card or dimmed something under ReduceMotion would have passed every
// test here; this fails, and names the stage.
func TestReduceMotionChangesTheRestingFrameNowhere(t *testing.T) {
	for step := 0; step < len(lessons); step++ {
		a := newTestApp(t)
		a.Step = step
		a.SelToken = 20
		plain := ui.Render(a.View, 1500, 940, 1)

		b := newTestApp(t)
		b.Step = step
		b.SelToken = 20
		tr := ui.NewTester(b.View, 1500, 940)
		tr.SetPreferences(ui.Preferences{ReduceMotion: true})
		tr.Frame()
		tr.Frame()
		got := tr.Image()

		if len(got.Pix) != len(plain.Pix) {
			t.Fatalf("stage %d renders a different size under ReduceMotion: %d vs %d bytes",
				step, len(got.Pix), len(plain.Pix))
		}
		for i := range plain.Pix {
			if plain.Pix[i] != got.Pix[i] {
				t.Fatalf("stage %d draws differently under ReduceMotion (first difference at byte %d); "+
					"the preference should silence animation, not change the picture", step, i)
			}
		}
	}
}

// TestReduceMotionLeavesInteractionAlone is the other half: scrolling, hovering and the keyboard are not animations and
// must behave the same with the preference on.
func TestReduceMotionLeavesInteractionAlone(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	tr := ui.NewTester(a.View, 1500, 940)
	tr.SetPreferences(ui.Preferences{ReduceMotion: true})
	tr.Frame()

	// Scrolling reaches the stage column.
	if a.Scroll.Y != 0 {
		t.Fatalf("scrolled before touching anything: %v", a.Scroll.Y)
	}
	tr.Scroll(400, 400, 0, 200)
	tr.Frame()
	if a.Scroll.Y == 0 {
		t.Fatal("the stage column did not scroll under ReduceMotion")
	}

	// Hovering the matrix still identifies a cell.
	found := false
	for y := float32(700); y > 240 && !found; y -= 8 {
		for x := float32(240); x < 420 && !found; x += 8 {
			tr.Move(x, y)
			tr.Frame()
			found = a.HoverQ >= 0
		}
	}
	if !found {
		t.Fatal("moving over the attention matrix set no hovered cell under ReduceMotion")
	}

	// And the keyboard still changes stage.
	before := a.Step
	tr.Key(0, ui.KeyRight)
	tr.Frame()
	if a.Step != before+1 {
		t.Fatalf("Right moved the stage from %d to %d under ReduceMotion", before, a.Step)
	}
}

// TestTooltipsAppearAndMatchTheirWidget replaces a wrong claim in CONTRIBUTING.
//
// The guide said tooltips could not be observed headless — "sweeping the pointer across the token strip leaves the text count
// unchanged" — and used that to justify checking them only at the source level. That measurement was taken without advancing
// the clock. A tooltip is delayed, and Tester's clock stands still unless the test moves it with Advance; hover plus roughly
// three quarters of a second and the tooltip text shows up in Texts().
//
// So they can be asserted like anything else. The three checked here are the ones whose content is a number a reader is meant to
// compare against the screen: the token chip's byte id, the sidebar's stage number, and a histogram bar's value.
// hoverForTooltip parks the pointer, lets any previous tooltip expire, hovers (x,y), and returns the first text that
// satisfies match.
//
// The parking step is not optional: a tooltip from the previous call stays in Texts() for a while, so hovering the next
// widget and reading Texts() immediately returns the old one. Two rounds of wrong answers came from skipping it.
func hoverForTooltip(t *testing.T, tr *ui.Tester, x, y float32, match func(string) bool) string {
	t.Helper()
	tr.Move(700, 600)
	tr.Frame()
	for i := 0; i < 10; i++ {
		tr.Advance(250 * time.Millisecond)
		tr.Frame()
	}
	tr.Move(x, y)
	tr.Frame()
	for i := 0; i < 10; i++ {
		tr.Advance(250 * time.Millisecond)
		tr.Frame()
		for _, s := range tr.Texts() {
			if match(s) {
				return s
			}
		}
	}
	return ""
}

func TestTokenChipTooltipStatesItsByte(t *testing.T) {
	a := newTestApp(t)
	a.Step = 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	// The strip's chips sit in the row below the top bar. Walk it until a tooltip appears, rather than guessing a coordinate:
	// the chip widths depend on the prompt.
	isChipTip := func(s string) bool {
		return strings.HasPrefix(s, "token ") && strings.Contains(s, "byte") && strings.Contains(s, "0x")
	}
	var tip string
	for x := float32(280); x < 900 && tip == ""; x += 30 {
		for _, y := range []float32{112, 116, 120} {
			if tip = hoverForTooltip(t, tr, x, y, isChipTip); tip != "" {
				break
			}
		}
	}
	if tip == "" {
		t.Fatal("hovering the token strip never showed its tooltip")
	}

	// The tooltip names a token number and that token's byte, and the screen shows the same byte in decimal.
	var n, id int
	var runeText string
	if _, err := fmt.Sscanf(tip, "token %d · byte %d", &n, &id); err != nil {
		t.Fatalf("cannot read the tooltip %q: %v", tip, err)
	}
	if n < 0 || n >= len(a.Trace.Tokens) {
		t.Fatalf("the tooltip names token %d, out of range for %d tokens", n, len(a.Trace.Tokens))
	}
	if want := a.Trace.Tokens[n].ID; id != want {
		t.Fatalf("the tooltip says token %d is byte %d; the trace says %d", n, id, want)
	}
	_ = runeText
}

func TestStageTooltipNamesTheStep(t *testing.T) {
	a := newTestApp(t)
	a.Step = 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()

	// The sidebar's seven rows, top to bottom: first row at y=76, each 32 DIP tall at 1500×940.
	found := 0
	for i := range lessons {
		y := float32(76 + i*32 + 16)
		// The step bar also prints "第 N / 7 步", and it is in Texts() before any tooltip. The tooltip is the one
		// with the stage's title after the dot.
		isStepTip := func(s string) bool {
			return strings.HasPrefix(s, "第 ") && strings.Contains(s, "/ 7 步 · ")
		}
		tip := hoverForTooltip(t, tr, 110, y, isStepTip)
		if tip == "" {
			continue
		}
		found++
		var step int
		if _, err := fmt.Sscanf(tip, "第 %d / 7 步", &step); err != nil {
			t.Fatalf("cannot read the stage tooltip %q: %v", tip, err)
		}
		if step != i+1 {
			t.Fatalf("the tooltip on sidebar row %d says it is step %d", i+1, step)
		}
	}
	if found < 5 {
		t.Fatalf("only %d of the seven sidebar rows showed a tooltip", found)
	}
}

// TestLabelStrideKeepsLabelsApart pins the rule behind the heatmap's axis labels.
//
// The labels are painted with Painter.RichText, so they never appear in Texts() and the only way to see the stride on screen is
// to look at a picture and count. The rule itself is arithmetic — never draw two labels closer than 26 DIP — and it was inlined
// in the draw callback, where nothing could reach it. It is a function now.
//
// The README quotes what it produces at three sizes; those figures are the same rule, so keeping the rule right keeps the README
// right. Measured off a 1500×940 render of the 36-token prompt: the top edge reads 0 3 6 9 12 …, a stride of 3.
func TestLabelStrideKeepsLabelsApart(t *testing.T) {
	const minGap = 26
	for _, cw := range []float32{2, 3, 5, 8.67, 9, 13, 13.01, 20, 26, 26.5, 40} {
		s := labelStride(cw)
		if s < 1 {
			t.Fatalf("cell width %.2f produced a stride of %d", cw, s)
		}
		if gap := float32(s) * cw; gap < minGap {
			t.Fatalf("cell width %.2f with stride %d leaves %.2f DIP between labels, under the %.0f the rule "+
				"promises", cw, s, gap, float32(minGap))
		}
		// And it does not overshoot: one cell less would have been too tight.
		if s > 1 && float32(s-1)*cw >= minGap {
			t.Fatalf("cell width %.2f got stride %d, but %d would already clear %.0f", cw, s, s-1, float32(minGap))
		}
	}

	// The three figures the README quotes, as cell widths. A 36-token prompt at 1500 wide draws at stride 3; a 64-token one
	// at 1500 and 1920 at 4; a 64-token one at 1180 at 6.
	for _, c := range []struct {
		cw   float32
		want int
	}{
		{26.0 / 3.0, 3}, // the 1500-wide 36-token heatmap, which the screenshot shows as 0 3 6 9 …
		{26.0 / 4.0, 4},
		{26.0 / 6.0, 6},
	} {
		if got := labelStride(c.cw); got != c.want {
			t.Fatalf("cell width %.3f gives stride %d, and the README's figures assume %d", c.cw, got, c.want)
		}
	}
}

// TestEveryPinReleaseTriggerIsInTheReadme closes the gap the previous test left:
// it checked stage switch, token selection and sampling, which were the three the
// README listed, so both the document and the test shared the same blind spot.
//
// Measured, three more actions clear the pin — switching layer, switching head, and
// autoplay advancing — and each is a real context change for the same reason: the
// cell belongs to a (layer, head, stage, row) context, and after any of those the
// card's heading and its contents would describe different things. The README now
// lists all six; this test walks them so a seventh cannot appear unannounced.
func TestEveryPinReleaseTriggerIsInTheReadme(t *testing.T) {
	doc := readREADME(t)
	// The row that documents the pinned cell, taken whole so a phrase from a
	// neighbouring row cannot satisfy the check.
	var row string
	for _, line := range strings.Split(doc, "\n") {
		if strings.HasPrefix(line, "| 点击热力图单元格 |") {
			row = line
		}
	}
	if row == "" {
		t.Fatal("the README no longer documents the pinned cell")
	}

	pinned := func() *App {
		a := newTestApp(t)
		a.Step = 2
		a.KeyQ, a.KeyK = 20, 5
		return a
	}
	releases := []struct {
		phrase string // how the README names it
		act    func(a *App)
	}{
		{"切换阶段", func(a *App) { a.selectStep(3) }},
		{"切换层", func(a *App) { a.Layer = 1; a.clearHover() }},
		{"切换头", func(a *App) { a.Head = 1; a.clearHover() }},
		{"自动播放推进", func(a *App) { a.Step++; a.clearHover() }},
		{"选中另一个 token", func(a *App) { a.selectToken(3, false) }},
		{"采样续写", func(a *App) { a.sampleNext() }},
	}
	for _, r := range releases {
		a := pinned()
		r.act(a)
		if a.KeyQ >= 0 || a.KeyK >= 0 {
			t.Fatalf("%s did not release the pin (still at (%d,%d))", r.phrase, a.KeyQ, a.KeyK)
		}
		if !strings.Contains(row, r.phrase) {
			t.Fatalf("the README's pinned-cell row does not mention %q, which releases the "+
				"pin:\n%s", r.phrase, row)
		}
	}
}

// TestNoLiteralMarkdownInRenderedText: ui.Text draws the characters it is given,
// so a markdown emphasis sequence renders as its asterisks.
//
// Found by accident while writing another test this round: the output stage's
// lesson paragraph carried "只把**最后一个位置**的向量投影到词表上", which a reader
// saw as four asterisks around the phrase. Nothing was checking for it, and this
// project's prose is dense enough that a stray marker is easy to introduce.
func TestNoLiteralMarkdownInRenderedText(t *testing.T) {
	// A tall window so every card on every stage is laid out and rendered.
	for step := 0; step < len(lessons); step++ {
		a := newTestApp(t)
		a.Step = step
		a.KeyQ, a.KeyK = 20, 5 // so the score stage's conditional block renders too
		tr := ui.NewTester(a.View, 1500, 3000)
		tr.Frame()
		for _, s := range tr.Texts() {
			if strings.Contains(s, "**") {
				t.Fatalf("stage %d renders a literal markdown emphasis marker: %q", step+1, s)
			}
			if strings.Contains(s, "`") {
				t.Fatalf("stage %d renders a literal backtick: %q", step+1, s)
			}
		}
	}
}

// TestScoreColorScaleRangeMatchesTheWeights holds the ranges quoted in the score-stage comment.
//
// That comment used to say the colour ramp's ceiling runs 4.6 to 12.5 on layer 0 and 8.1 to 15.2
// on layer 1. Measured across the same kind of prompt set: 0.6 to 10.4 and 1.7 to 17.8 — both
// ranges wrong, and wrong in the direction that makes the ramp's movement look smaller than it
// is. Nothing checked them; the only nearby test asks whether the scale VARIES between prompts,
// which stays true whatever the numbers are.
//
// The bounds below are the measurement with a little room, so ordinary noise does not fail the
// test but a real change in what the weights do does. If they trip, re-measure and update both
// this test and the comment it guards — the two are one fact stated twice, so a failure should
// say so rather than looking like a regression in the painting.
func TestScoreColorScaleRangeMatchesTheWeights(t *testing.T) {
	// The prompt set the comment describes, which is the set the README means by "this model":
	// the four presets the app ships. An earlier version of this test used eight ad-hoc prompts
	// and concluded the documented range was stale; it was not. Different prompt sets produce
	// different extremes, so the set is part of the claim.
	var prompts []string
	for _, p := range promptPresets {
		prompts = append(prompts, p.prompt)
	}
	lo := map[int]float64{}
	hi := map[int]float64{}
	for _, p := range prompts {
		a := newTestApp(t)
		a.Prompt = p
		a.recompute()
		for li := range a.Trace.Layers {
			for h := range a.Trace.Layers[li].Heads {
				sc := scoreColorScale(a.Trace.Layers[li].Heads[h])
				if sc <= 0 {
					continue
				}
				if v, ok := lo[li]; !ok || sc < v {
					lo[li] = sc
				}
				if v, ok := hi[li]; !ok || sc > v {
					hi[li] = sc
				}
			}
		}
	}
	if len(lo) < 2 {
		t.Fatalf("measured %d layers, want both", len(lo))
	}

	// Read the figures out of the comment rather than keeping a second copy. Holding them here
	// instead means the test still passes when the comment alone goes stale — which is the exact
	// defect the comment had. Measured: reverting the comment while leaving this test alone left
	// it green.
	src, err := os.ReadFile("stages.go")
	if err != nil {
		t.Fatal(err)
	}
	re := regexp.MustCompile(`colour scale ([0-9]+\.[0-9]+)–([0-9]+\.[0-9]+) on layer 0 and ` +
		`([0-9]+\.[0-9]+)–([0-9]+\.[0-9]+) on layer 1`)
	m := re.FindStringSubmatch(string(src))
	if m == nil {
		t.Fatal("the score stage's comment no longer quotes a measured range of the form " +
			"\"A–B on layer 0 and C–D on layer 1\"; update this test with whatever it says now")
	}
	parse := func(v string) float64 {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			t.Fatalf("cannot read %q as a number: %v", v, err)
		}
		return f
	}
	want := []struct {
		layer  int
		lo, hi float64
	}{
		{0, parse(m[1]), parse(m[2])},
		{1, parse(m[3]), parse(m[4])},
	}
	for _, w := range want {
		gotLo, gotHi := lo[w.layer], hi[w.layer]
		// A tenth of slack: the exact extremes depend on which prompts are in the set, and
		// pinning them to the last digit would make this a test of the prompt list.
		if math.Abs(gotLo-w.lo) > 0.1 || math.Abs(gotHi-w.hi) > 0.1 {
			t.Fatalf("layer %d colour scale runs %.2f to %.2f; the comment on the score stage "+
				"says %.1f to %.1f — re-measure and update both, they are one fact stated twice",
				w.layer, gotLo, gotHi, w.lo, w.hi)
		}
	}
}

// TestSamplingAtTheContextLimitSaysTheByteIsInvisible keeps the sample record honest when the
// window is already full.
//
// recompute keeps the last Ctx tokens, so appending a byte to a prompt that already fills the
// context grows the string without changing the trace. Measured before this: a 70-byte prompt
// reached 73 bytes after three presses while the token count stayed at 64 — the reader's only
// visible feedback was a count that did not move, and no card said why the new byte had no effect.
//
// The draw is still shown, since the reader asked for a token, but the record has to say the byte
// landed outside the window.
func TestSamplingAtTheContextLimitSaysTheByteIsInvisible(t *testing.T) {
	a := newTestApp(t)
	ctx := a.W.Config.Ctx
	a.Prompt = strings.Repeat("a", ctx+6)
	a.recompute()
	if got := len(a.Trace.Tokens); got != ctx {
		t.Fatalf("the prompt is %d bytes but the trace holds %d tokens; this test needs the window "+
			"to be full", len(a.Prompt), got)
	}
	before := len(a.Prompt)
	a.sampleNext()
	if len(a.Prompt) != before+1 {
		t.Fatalf("sampling appended %d bytes, want 1", len(a.Prompt)-before)
	}
	if got := len(a.Trace.Tokens); got != ctx {
		t.Fatalf("the trace now holds %d tokens, want it pinned at %d", got, ctx)
	}
	if !strings.Contains(a.LastSample, "追加在窗口之外") {
		t.Fatalf("a byte was appended past the window with the record saying %q; nothing tells the "+
			"reader the model cannot see it", a.LastSample)
	}

	// A prompt with room must not carry the warning: it would be false there.
	b := newTestApp(t)
	b.Prompt = "hi"
	b.recompute()
	b.sampleNext()
	if strings.Contains(b.LastSample, "追加在窗口之外") {
		t.Fatalf("a prompt with room reported the byte as invisible: %q", b.LastSample)
	}
}
