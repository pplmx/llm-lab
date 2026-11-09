package app

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"testing"

	"llmlab/internal/model"

	"github.com/egoist/mygo/ui"
)

// The app accepts any bytes a reader can paste into the prompt box, and it can
// be started with random weights when the bundled file is missing. Neither path
// had a test, and both are the kind that fail only for the one person who hits
// them.

// TestEveryStageRendersAnyInput walks the inputs a paste can produce, through all
// seven stages. A panic here would take down the whole window.
func TestEveryStageRendersAnyInput(t *testing.T) {
	inputs := []struct {
		name   string
		prompt string
	}{
		{"single ascii", "a"},
		{"spaces only", "     "},
		{"newlines only", "\n\n\n"},
		{"control bytes", "\x01\x02\x03"},
		{"nul in the middle", "a\x00b"},
		{"emoji", "🙂🙂"},
		{"astral codepoint", "𝄞"},
		{"lone invalid byte", "\xff"},
		{"invalid run mid-string", "a\xffb"},
		{"repeated byte", strings.Repeat("a", 43)},
		{"mixed scripts and junk", "a注🙂\xffb"},
		{"at the context ceiling", strings.Repeat("b", 64)},
		{"past the context ceiling", strings.Repeat("c", 200)},
	}
	for _, in := range inputs {
		in := in
		t.Run(in.name, func(t *testing.T) {
			a := newTestApp(t)
			a.Prompt = in.prompt
			a.recompute()
			if len(a.Trace.Tokens) == 0 {
				t.Fatalf("%q produced no tokens", in.prompt)
			}
			for step := 0; step < 7; step++ {
				a.Step = step
				tr := ui.NewTester(a.View, 1180, 720)
				tr.Frame()
				// A NaN or an infinity in a formatted figure means some
				// quantity went wrong. The one legitimate "+Inf"-like string
				// is the softmax callout explaining the causal mask.
				for _, s := range tr.Texts() {
					if strings.Contains(s, "因果掩码") {
						continue
					}
					if strings.Contains(s, "NaN") || strings.Contains(s, "+Inf") {
						t.Fatalf("step %d shows %q for prompt %q", step+1, s, in.prompt)
					}
				}
			}
		})
	}
}

// TestUnprintableBytesKeepTheirEscape checks control and invalid bytes are always
// shown as escapes rather than dropped or rendered as blanks: the whole point of
// the byte-level tokeniser is that a reader can see every byte.
func TestUnprintableBytesKeepTheirEscape(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = "a\x01\x7f"
	a.recompute()
	if n := len(a.Trace.Tokens); n != 3 {
		t.Fatalf("%d tokens for three bytes", n)
	}
	for i, want := range []string{"a", `\x01`, `\x7f`} {
		if got := a.Trace.Tokens[i].Text; got != want {
			t.Fatalf("token %d renders as %q, want %q", i, got, want)
		}
	}
}

// TestRandomWeightsRenderWithoutPoisoning walks every stage with random weights,
// which is what main.go falls back to when the bundled file cannot be read. The
// banner warns that the numbers are meaningless; it must still be possible to
// look at them.
func TestRandomWeightsRenderWithoutPoisoning(t *testing.T) {
	a := New(model.Random(model.DefaultConfig, 42))
	if a.paramCount() == 0 {
		t.Fatal("random weights report no parameters")
	}
	for step := 0; step < 7; step++ {
		a.Step = step
		tr := ui.NewTester(a.View, 1180, 720)
		tr.Frame()
		for _, s := range tr.Texts() {
			if strings.Contains(s, "因果掩码") {
				continue
			}
			if strings.Contains(s, "NaN") || strings.Contains(s, "%!") {
				t.Fatalf("step %d shows %q with random weights", step+1, s)
			}
		}
	}
	// The distribution is still a distribution, whatever it means.
	probs := a.Trace.Probabilities(a.Temp)
	var sum float64
	for _, p := range probs {
		if p < 0 || p != p { // negative or NaN
			t.Fatalf("probability %.4f out of range with random weights", p)
		}
		sum += float64(p)
	}
	if sum < 0.999 || sum > 1.001 {
		t.Fatalf("probabilities sum to %v", sum)
	}
	// And the banner is up, so a reader cannot mistake them for trained output.
	// Matched on the banner's own wording: "未训练" alone also appears in the
	// sidebar's weight line, so it would still match with the banner removed.
	if !tr2HasText(t, a, "别从输出里读出意义") {
		t.Fatal("random weights do not raise the untrained banner")
	}
}

// tr2HasText reports whether any frame of the app contains s.
func tr2HasText(t *testing.T, a *App, s string) bool {
	t.Helper()
	for step := 0; step < 7; step++ {
		a.Step = step
		tr := ui.NewTester(a.View, 1180, 720)
		tr.Frame()
		if tr.HasText(s) {
			return true
		}
	}
	return false
}

// TestStartingOnALaterStageRendersThatStage checks the -step flag's path, which
// assigns a.Step directly instead of going through selectStep.
func TestStartingOnALaterStageRendersThatStage(t *testing.T) {
	// Step 1's card adapts to the prompt; the default is ASCII, so it explains why bytes and
	// runes coincide. TestByteCardFollowsThePrompt covers the CJK branch.
	want := map[int]string{0: "ASCII 下字节就是字符", 3: "这一行的统计", 6: "分布统计"}
	for step, marker := range want {
		a := newTestApp(t)
		a.Step = step // what main.go does for -step
		tr := ui.NewTester(a.View, 1180, 720)
		tr.Frame()
		if !tr.HasText(marker) {
			t.Fatalf("booting on step %d does not show its inspector card", step+1)
		}
	}
}

// narrowWeights returns a copy of w with d_model halved, every tensor that depends on it sliced
// accordingly, so the app can be run against a shape the bundle does not have.
//
// The point is not that such a checkpoint is realistic. It is that d_model is only required to be
// positive, so these files load, and a sentence or title that quotes the width has to follow the
// file rather than the bundle.
func narrowWeights(base *model.Weights) *model.Weights {
	d := base.Config.DModel / 2
	out := *base
	out.Config = base.Config
	out.Config.DModel = d
	out.TokEmbed = base.TokEmbed[:base.Config.Vocab*d]
	out.PosEmbed = base.PosEmbed[:base.Config.Ctx*d]
	out.LNFG = base.LNFG[:d]
	out.LNFB = base.LNFB[:d]
	out.Layers = nil
	for _, l := range base.Layers {
		out.Layers = append(out.Layers, model.LayerWeights{
			LN1G: l.LN1G[:d], LN1B: l.LN1B[:d], LN2G: l.LN2G[:d], LN2B: l.LN2B[:d],
			WQ: l.WQ[:d*d], WK: l.WK[:d*d], WV: l.WV[:d*d], WO: l.WO[:d*d],
			W1: l.W1[:d*base.Config.DFF], W2: l.W2[:base.Config.DFF*d],
		})
	}
	return &out
}

// stageTexts collects every string the seven stages render for one weight set.
func stageTexts(t *testing.T, w *model.Weights) map[string]bool {
	t.Helper()
	seen := make(map[string]bool)
	for step := 0; step < 7; step++ {
		a := New(w)
		a.Step = step
		tr := ui.NewTester(a.View, 1500, 4000)
		tr.Frame()
		for _, s := range tr.Texts() {
			seen[s] = true
		}
	}
	return seen
}

// standaloneNumber matches n as a number in its own right: not a digit inside a longer number,
// and not the fractional part of a decimal.
//
// The first version of these checks used a plain word-boundary-ish pattern and reported
// "min -2.64 · max +2.90" as naming the context window, because 64 appears inside 2.64. A guard
// that cries wolf over a cosine is worse than no guard: it trains the reader to skim past its
// failures, which is exactly how a real one gets missed. Byte values are legitimately 64 too —
// "byte 0x64 · rune 'd'" and "'d' = U+0064，占 1 个字节：64 01100100" both do name 64 and neither
// says anything about the context window.
//
// So a match has to be preceded by something that is neither a digit, a letter nor a decimal
// point, and followed by something that is not a digit or a decimal point.
func standaloneNumber(n string) *regexp.Regexp {
	return regexp.MustCompile(`(^|[^0-9A-Za-z.])` + regexp.QuoteMeta(n) + `([^0-9.]|$)`)
}

// TestNoStageNamesTheBundleConfigAtAnotherWidth is the general form of three separate defects
// found one at a time: the positional card's "64 个位置", the output lesson's "[T, 96]", and the
// embedding card's title "这 96 个维度".
//
// Each was a number from the loaded weights written as a constant. Each was correct for the
// bundled file and wrong for any other, and each survived because every test loaded the bundle.
// Rather than looking for the fourth by reading code, this renders the whole app twice — once on
// the bundle, once on a halved d_model — and requires that no string survives unchanged while
// naming the bundle's width.
//
// The check is on unchanged strings specifically. A string that mentions 96 and changed has
// clearly been rewritten to follow the config; one that mentions 96 and did not change is still
// a constant, whether or not it happens to be right for this checkpoint. Coincidental matches are
// the reason the assertion is about the *pair* (mentions the width, did not change) rather than
// about the number alone: a byte value, a cosine or a position index can legitimately be 96.
func TestNoStageNamesTheBundleConfigAtAnotherWidth(t *testing.T) {
	base, err := model.Bundled()
	if err != nil {
		t.Skipf("no bundled weights: %v", err)
	}
	narrow := narrowWeights(base)
	if narrow.Config.DModel == base.Config.DModel {
		t.Fatal("the narrow copy has the bundle's width, so this test proves nothing")
	}

	bundled := stageTexts(t, base)
	halved := stageTexts(t, narrow)
	if len(halved) == 0 {
		t.Fatal("the narrow copy rendered nothing")
	}

	// Every string that mentions the bundle's width and is unchanged in the halved render.
	width := fmt.Sprintf("%d", base.Config.DModel)
	re := standaloneNumber(width)
	var stale []string
	for s := range bundled {
		if !re.MatchString(s) {
			continue
		}
		if halved[s] {
			stale = append(stale, s)
		}
	}
	if len(stale) > 0 {
		sort.Strings(stale)
		var b strings.Builder
		for _, s := range stale {
			b.WriteString("\n  " + s)
		}
		t.Fatalf("%d string(s) name the bundle's d_model %s and survive unchanged when the model "+
			"is %d wide, so they are constants rather than derived:%s",
			len(stale), width, narrow.Config.DModel, b.String())
	}
}

// windowUnchanged reports whether the text immediately around every match of re in s appears,
// at the same offsets, in one of the other strings from the same render.
//
// The point is to separate "this number did not follow the config" from "this sentence happened
// to differ somewhere else". A stale constant sits in prose that is otherwise fixed, so its
// neighbourhood repeats verbatim in the other render even when the sentence as a whole does not.
func windowUnchanged(a, b map[string]bool, s string, re *regexp.Regexp, pad int) bool {
	locs := re.FindAllStringIndex(s, -1)
	if len(locs) == 0 {
		return false
	}
	for _, loc := range locs {
		start := loc[0] - pad
		if start < 0 {
			start = 0
		}
		end := loc[1] + pad
		if end > len(s) {
			end = len(s)
		}
		// Run lengths in bytes are fine here: the pad is only a heuristic for "the words next
		// to it", and a multi-byte character straddling the boundary would just make the window
		// slightly shorter on both sides, equally.
		around := s[start:end]
		found := false
		for other := range b {
			if strings.Contains(other, around) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// halfDFFWeights returns a copy of w with d_ff halved. d_ff is only required to be positive, so
// such a file loads; the FFN tensors are the only ones shaped by it.
func halfDFFWeights(base *model.Weights) *model.Weights {
	dff := base.Config.DFF / 2
	out := *base
	out.Config = base.Config
	out.Config.DFF = dff
	out.Layers = nil
	for _, l := range base.Layers {
		c := l
		c.W1 = l.W1[:base.Config.DModel*dff]
		c.W2 = l.W2[:dff*base.Config.DModel]
		out.Layers = append(out.Layers, c)
	}
	return &out
}

// dropLayerWeights returns a copy of w with one fewer block.
func dropLayerWeights(base *model.Weights) *model.Weights {
	out := *base
	out.Config = base.Config
	out.Config.NLayer = base.Config.NLayer - 1
	out.Layers = base.Layers[:out.Config.NLayer]
	return &out
}

// TestNoStageNamesTheBundleFeedForwardWidth is the d_ff member of the family.
//
// The FFN-concentration card said "上面那条色带只画了选中位置的 384 个单元" and "两层都是 384 个
// 单元" — 384 typed in, while the row struct already carried Units and the same card printed
// per-layer unit counts from it. Measured with d_ff halved: both lines still said 384.
//
// The second line also hardcoded "两层". The card renders only with at least two rows, so naming
// the two layers it is comparing is both correct and clearer than counting them in prose.
func TestNoStageNamesTheBundleFeedForwardWidth(t *testing.T) {
	base, err := model.Bundled()
	if err != nil {
		t.Skipf("no bundled weights: %v", err)
	}
	narrow := halfDFFWeights(base)
	if narrow.Config.DFF == base.Config.DFF {
		t.Fatal("the copy has the bundle's d_ff, so this test proves nothing")
	}
	bundled := stageTexts(t, base)
	smaller := stageTexts(t, narrow)

	width := fmt.Sprintf("%d", base.Config.DFF)
	re := standaloneNumber(width)
	var stale []string
	for s := range bundled {
		if !re.MatchString(s) || !strings.Contains(s, "单元") {
			continue
		}
		if windowUnchanged(bundled, smaller, s, re, 24) {
			stale = append(stale, s)
		}
	}
	if len(stale) > 0 {
		sort.Strings(stale)
		var b strings.Builder
		for _, s := range stale {
			b.WriteString("\n  " + s)
		}
		t.Fatalf("%d string(s) name the bundle's d_ff %s and survive unchanged at d_ff %d:%s",
			len(stale), width, narrow.Config.DFF, b.String())
	}
}

// TestCardsThatCompareLayersNeedTwoOfThem records why n_layer gets no "constant prose" guard
// like the d_model, d_ff and ctx ones.
//
// It was written as one, and the check was vacuous: it looked for "两层" surviving unchanged on a
// one-layer model, and no such string exists because every card that compares layers returns
// early when there are fewer than two. Measured — a one-layer copy renders zero strings
// containing "两层", and the guard passed before any fix was made.
//
// So the property worth pinning is the early return itself, which is what keeps the hardcoded
// layer indices in those cards from ever reaching a reader. The cards in question name layer 0
// and layer 1 explicitly — "预测在层 0 的 FFN 之后就已经成形，层 1 是在把它磨尖" — which is
// correct prose for a two-layer model and would be nonsense on a one-layer one. The guard is the
// `len(...) < 2` check, not the wording.
func TestCardsThatCompareLayersNeedTwoOfThem(t *testing.T) {
	base, err := model.Bundled()
	if err != nil {
		t.Skipf("no bundled weights: %v", err)
	}
	one := dropLayerWeights(base)
	if one.Config.NLayer >= 2 {
		t.Fatal("the copy still has two layers, so this test proves nothing")
	}

	// Every card that talks about layers relative to each other. Each is keyed by a string
	// unique to its heading, so this fails loudly if a card is renamed rather than passing
	// vacuously on a missing card.
	compared := []struct {
		heading string
		step    int
	}{
		{"每一层改写了多少", 4},
		{"单元是怎么分工的", 5},
	}
	for _, c := range compared {
		a := New(one)
		a.Step = c.step
		tr := ui.NewTester(a.View, 1500, 4000)
		tr.Frame()
		joined := strings.Join(tr.Texts(), "\n")
		if strings.Contains(joined, c.heading) {
			t.Fatalf("step %d renders %q on a %d-layer model; that card hardcodes layer 0 and "+
				"layer 1, so it must not draw when there is only one", c.step+1, c.heading, one.Config.NLayer)
		}
	}

	// And the two-layer model still draws them, so this is not passing because the headings
	// were renamed.
	for _, c := range compared {
		a := New(base)
		a.Step = c.step
		tr := ui.NewTester(a.View, 1500, 4000)
		tr.Frame()
		if !strings.Contains(strings.Join(tr.Texts(), "\n"), c.heading) {
			t.Fatalf("step %d does not render %q on the bundled weights either, so the check "+
				"above proves nothing", c.step+1, c.heading)
		}
	}
}

// TestNoStageNamesTheBundleContextAtAnotherWindow is the same check for the context window, which
// has a defect of its own on record: the positional-encoding card said "它把 64 个位置放成了 64
// 个几乎互不相干的地址" with Dims derived on the same line, and validate only requires ctx > 0,
// so a ctx=32 checkpoint loads and the sentence named 64.
//
// Separate from the width test because the copy is made differently — the positional table has one
// row per position, so a smaller window just drops rows, where a narrower d_model reslices every
// tensor in the model.
func TestNoStageNamesTheBundleContextAtAnotherWindow(t *testing.T) {
	base, err := model.Bundled()
	if err != nil {
		t.Skipf("no bundled weights: %v", err)
	}
	small := *base
	small.Config = base.Config
	small.Config.Ctx = base.Config.Ctx / 2
	small.PosEmbed = base.PosEmbed[:small.Config.Ctx*base.Config.DModel]
	if small.Config.Ctx == base.Config.Ctx {
		t.Fatal("the copy has the bundle's context window, so this test proves nothing")
	}

	bundled := stageTexts(t, base)
	smaller := stageTexts(t, &small)

	// Only sentences that are *about* the context window. The first version of this check
	// matched any string containing the number, and reported two false positives: the
	// tokenising stage prints a byte's own value as decimal 64, and byte 64 is 'd'. Those
	// strings name 64 and say nothing about how many positions the model has.
	//
	// Filtering on subject rather than trying to exclude coincidences by their spelling is the
	// honest way round: a guard that cries wolf over a byte value trains the reader to skim its
	// failures, which is how a real one gets missed.
	ctx := fmt.Sprintf("%d", base.Config.Ctx)
	re := standaloneNumber(ctx)
	var stale []string
	for s := range bundled {
		if !re.MatchString(s) || !strings.Contains(s, "位置") {
			continue
		}
		// Comparing whole strings misses a constant in a sentence that also quotes measured
		// figures: those figures differ between the two renders, so the sentence differs and
		// the stale number rides along unnoticed. Measured — putting "64 个位置" back into the
		// positional card left this test green, because the cosines in the same sentence had
		// changed too.
		//
		// So the unit is the neighbourhood of the number rather than the sentence: if the text
		// on both sides of it is identical in both renders, that number is not following the
		// config.
		if windowUnchanged(bundled, smaller, s, re, 24) {
			stale = append(stale, s)
		}
	}
	if len(stale) > 0 {
		sort.Strings(stale)
		var b strings.Builder
		for _, s := range stale {
			b.WriteString("\n  " + s)
		}
		t.Fatalf("%d string(s) name the bundle's ctx %s and survive unchanged at ctx %d, so they "+
			"are constants rather than derived:%s",
			len(stale), ctx, small.Config.Ctx, b.String())
	}
}

// headsWeights returns a copy of w declaring a different n_head. The tensors are unchanged: the
// head count only decides how the existing d_model is split, which is what the app reads for its
// labels and what the per-head views index into.
func headsWeights(base *model.Weights, n int) *model.Weights {
	out := *base
	out.Config = base.Config
	out.Config.NHead = n
	return &out
}

// TestEveryHeadCountRendersAndCountsItself covers n_head, which is the one config field that
// crashed the whole window rather than merely misdescribing it.
//
// validate requires only that d_model divides by n_head, so a two-head checkpoint loads. The
// head-profile card then read profiles[0]..profiles[3] unconditionally and panicked with "index
// out of range [2] with length 2" — measured on n_head = 1, 2 and 3, all on the softmax stage,
// while 4 and 8 were fine. Because the bundle has four heads, no test ever rendered anything
// else.
//
// Two things are asserted, because they fail independently: every divisor of d_model renders all
// seven stages without panicking, and the prose counts the heads it actually has. The second is
// why headsWord exists — eleven strings said 四个头 and were correct only for the bundle, and an
// earlier version of the helper returned the numeral alone, rendering "四头" at the default.
func TestEveryHeadCountRendersAndCountsItself(t *testing.T) {
	base, err := model.Bundled()
	if err != nil {
		t.Skipf("no bundled weights: %v", err)
	}
	// The chip label in stageCards is a fixed table of contents entry and cannot vary with the
	// checkpoint; it is the one place allowed to say 四个头 regardless.
	allowed := map[string]bool{"四个头": true}

	for n := 1; n <= base.Config.DModel; n++ {
		if base.Config.DModel%n != 0 {
			continue
		}
		w := headsWeights(base, n)
		a := New(w)
		for step := 0; step < 7; step++ {
			step := step
			// A panic would take down the window, so it is fatal here rather than a t.Fatalf
			// inside a recover — the failure has to name the stage either way.
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("n_head=%d (d_head=%d) step %d panicked: %v",
							n, w.Config.DHead(), step+1, r)
					}
				}()
				a.Step = step
				tr := ui.NewTester(a.View, 1500, 4000)
				tr.Frame()
				// Only wrong when the model does not have four heads. At n_head=4 these
				// strings are correct, and a guard that flagged them would be testing the
				// bundle rather than the derivation.
				if n == 4 {
					return
				}
				for _, s := range tr.Texts() {
					if allowed[s] {
						continue
					}
					if strings.Contains(s, "四个头") || strings.Contains(s, "四头") {
						t.Fatalf("n_head=%d step %d still says four heads: %q", n, step+1, s)
					}
				}
			}()
		}

	}
	// The phrase itself has to be well formed: "四头" is what a numeral helper without a measure
	// word produces, and it reads as broken Chinese. This is checked directly rather than
	// through the render, because at n_head=4 the corpus happens to read correctly either way.

	for n, want := range map[int]string{1: "一个", 2: "两个", 4: "四个", 8: "八个", 0: "0 个", 12: "12 个"} {
		if got := headsWord(n); got != want {
			t.Fatalf("headsWord(%d) = %q, want %q", n, got, want)
		}
	}
}

// configVariant is one checkpoint shape the app has to survive, built by reslicing or padding
// the bundled weights rather than by training anything.
type configVariant struct {
	name string
	w    *model.Weights
}

// configVariants enumerates the shapes a weights file can legally have.
//
// Each field here is only checked for its own invariant in validate — ctx and d_ff and d_model
// positive, d_model divisible by n_head — so all of these load. Two rounds of this work found real
// defects in that space: a n_head that is not four crashed the window outright, and a d_ff that is
// not 384 made a card describe units it did not have. Enumerating the field values centrally is
// what turns "test the shapes we happen to think of" into "test the shapes the loader admits".
func configVariants(t *testing.T, base *model.Weights) []configVariant {
	t.Helper()
	var out []configVariant

	for _, c := range []int{8, 16, 32, 64} {
		w := *base
		w.Config = base.Config
		w.Config.Ctx = c
		if c < base.Config.Ctx {
			w.PosEmbed = base.PosEmbed[:c*base.Config.DModel]
		}
		out = append(out, configVariant{fmt.Sprintf("ctx=%d", c), &w})
	}

	for _, f := range []int{2, 96, 192, 384} {
		w := *base
		w.Config = base.Config
		w.Config.DFF = f
		w.Layers = nil
		for _, l := range base.Layers {
			c := l
			c.W1 = make([]float32, base.Config.DModel*f)
			c.W2 = make([]float32, f*base.Config.DModel)
			copy(c.W1, l.W1)
			copy(c.W2, l.W2)
			w.Layers = append(w.Layers, c)
		}
		out = append(out, configVariant{fmt.Sprintf("dff=%d", f), &w})
	}

	for _, n := range []int{1, 2, 3} {
		w := *base
		w.Config = base.Config
		w.Config.NLayer = n
		w.Layers = nil
		for i := 0; i < n; i++ {
			w.Layers = append(w.Layers, base.Layers[i%len(base.Layers)])
		}
		out = append(out, configVariant{fmt.Sprintf("nlayer=%d", n), &w})
	}

	// Every divisor of d_model, since n_head may be any of them.
	for nh := 1; nh <= base.Config.DModel; nh++ {
		if base.Config.DModel%nh != 0 {
			continue
		}
		w := *base
		w.Config = base.Config
		w.Config.NHead = nh
		out = append(out, configVariant{fmt.Sprintf("nhead=%d", nh), &w})
	}

	// A narrower d_model with a single head, so the width itself varies too. The tensors are
	// zero-filled rather than resliced: at d_model 8 a zero patch of the 96-wide weights would
	// be meaningless, and this variant is about the shape surviving, not about the numbers.
	for _, d := range []int{8, 16, 48} {
		w := *base
		w.Config = base.Config
		w.Config.DModel = d
		w.Config.NHead = 1
		w.TokEmbed = make([]float32, base.Config.Vocab*d)
		w.PosEmbed = make([]float32, base.Config.Ctx*d)
		w.LNFG = make([]float32, d)
		w.LNFB = make([]float32, d)
		for i := range w.LNFG {
			w.LNFG[i] = 1
		}
		w.Layers = nil
		for range base.Layers {
			c := model.LayerWeights{
				LN1G: make([]float32, d), LN1B: make([]float32, d),
				LN2G: make([]float32, d), LN2B: make([]float32, d),
				WQ: make([]float32, d*d), WK: make([]float32, d*d),
				WV: make([]float32, d*d), WO: make([]float32, d*d),
				W1: make([]float32, d*base.Config.DFF), W2: make([]float32, base.Config.DFF*d),
			}
			for i := range c.LN1G {
				c.LN1G[i] = 1
				c.LN2G[i] = 1
			}
			w.Layers = append(w.Layers, c)
		}
		out = append(out, configVariant{fmt.Sprintf("dmodel=%d", d), &w})
	}

	if len(out) < 10 {
		t.Fatalf("only %d variants; the enumeration has shrunk", len(out))
	}
	return out
}

// TestEveryLegalConfigRendersEveryStage is the sweep that the last two rounds were working toward
// one field at a time.
//
// n_head=2 crashed the window; d_ff=192 made a card lie about its unit count. Both were reachable
// from a file that validate accepts. Rather than wait for the next one, this walks the shapes the
// loader admits — every divisor for n_head, several values for each other free field, three
// widths — against several prompts and all seven stages, and requires that nothing panics and
// nothing renders a malformed figure.
//
// Panics are fatal rather than collected: a panic in this app takes down the whole window, so one
// is enough to fail. The recover is per (variant, prompt, stage) so the message can name all
// three instead of printing a bare stack.
func TestEveryLegalConfigRendersEveryStage(t *testing.T) {
	base, err := model.Bundled()
	if err != nil {
		t.Skipf("no bundled weights: %v", err)
	}
	// Two prompts, not four. The dimension being swept is the config, and prompt shape is already
	// covered by TestEveryStageRendersAnyInput; what these two add is a multi-token ASCII string
	// and a short multi-byte one, which are the extremes of token count and of bytes-per-token.
	// Four prompts made this test 22 seconds of the 190-second suite for no extra config coverage.
	prompts := []struct{ name, text string }{
		{"default", "The transformer reads the tokens and"},
		{"cjk", "注意力就是加权求和"},
	}

	for _, v := range configVariants(t, base) {
		for _, p := range prompts {
			for step := 0; step < 7; step++ {
				step := step
				v, p := v, p
				func() {
					defer func() {
						if r := recover(); r != nil {
							t.Fatalf("%s, prompt %q, stage %d panicked: %v",
								v.name, p.name, step+1, r)
						}
					}()
					a := New(v.w)
					a.Prompt = p.text
					a.recompute()
					a.Step = step
					tr := ui.NewTester(a.View, 1500, 3000)
					tr.Frame()
					for _, s := range tr.Texts() {
						// The softmax callout legitimately mentions the mask producing -Inf.
						if strings.Contains(s, "因果掩码") {
							continue
						}
						if strings.Contains(s, "NaN") || strings.Contains(s, "+Inf") ||
							strings.Contains(s, "-Inf") || strings.Contains(s, "%!") {
							t.Fatalf("%s, prompt %q, stage %d renders %q",
								v.name, p.name, step+1, s)
						}
					}
				}()
			}
		}
	}
}

// TestNoLessonBodyDescribesAnEmptyTraceAsASmallOne keeps the empty state from being measured.
//
// The softmax lesson's body had one branch for "fewer than three comparable rows", which covered
// zero rows as well as one and two. On an empty prompt it rendered "这个提示词只有 0 行可以比较 …
// 样本太少 … 多打几个字再看" — a sentence that reports a measurement of nothing and tells the
// reader to add to text that is not there.
//
// The stage area already short-circuits to the empty-state card when there are no tokens, so this
// is the one surface where the empty prompt still reached a body written for a real one. The
// check walks every bodyFn with an empty trace and looks for the phrasings that presuppose tokens.
func TestNoLessonBodyDescribesAnEmptyTraceAsASmallOne(t *testing.T) {
	empty := &model.Trace{Config: model.DefaultConfig}
	presupposes := []string{"0 行", "0 个位置", "0/0", "0 对", "多打几个字", "这个提示词只有 0"}
	checked := 0
	for i, l := range lessons {
		if l.bodyFn == nil {
			continue
		}
		checked++
		body := l.bodyFn(empty)
		for _, bad := range presupposes {
			if strings.Contains(body, bad) {
				t.Fatalf("stage %d's body says %q when the prompt is empty, which reads as a "+
					"measurement of something rather than as the empty state:\n%s", i+1, bad, body)
			}
		}
	}
	if checked == 0 {
		t.Fatal("no lesson body was exercised; the check is verifying nothing")
	}
}

// TestPromptTooWideForTheWindowIsNotReportedAsEmpty covers the second way to reach zero tokens.
//
// Truncate keeps runes whole, so a prompt whose every character is wider than ctx yields no ids at
// all. With ctx below 4 that is reachable: measured, an emoji prompt at ctx=1, 2 or 3 produces zero
// tokens while the input box plainly holds text. The app called that state "还没有提示词" in both
// the status line and the empty-state card, and the card told the reader to type something — advice
// pointing at a box that already had something in it.
//
// The two zero-token states are now separate: an empty prompt says so, and a prompt that cannot fit
// says why and names the only real fix.
func TestPromptTooWideForTheWindowIsNotReportedAsEmpty(t *testing.T) {
	base, err := model.Bundled()
	if err != nil {
		t.Skip("no bundled weights")
	}
	withCtx := func(ctx int) *App {
		w := *base
		w.Config = base.Config
		w.Config.Ctx = ctx
		w.PosEmbed = base.PosEmbed[:ctx*base.Config.DModel]
		return New(&w)
	}

	tooWide := withCtx(1)
	tooWide.Prompt = "😀" // 4 bytes, cannot fit in a 1-token window whole
	tooWide.recompute()
	if n := len(tooWide.Trace.Tokens); n != 0 {
		t.Fatalf("a 4-byte character fitted into a %d-token window (%d tokens)", 1, n)
	}
	if strings.Contains(tooWide.Status, "还没有提示词") {
		t.Fatalf("the status line calls a non-empty prompt empty: %q", tooWide.Status)
	}
	if !strings.Contains(tooWide.Status, "装不进") {
		t.Fatalf("the status line does not say the prompt cannot fit: %q", tooWide.Status)
	}
	tr := ui.NewTester(tooWide.View, 1500, 940)
	tr.Frame()
	var cardText string
	for _, s := range tr.Texts() {
		if strings.Contains(s, "装不进这个窗口") {
			cardText = s
		}
	}
	if cardText == "" {
		t.Fatal("the stage card still presents the too-wide prompt as the empty state")
	}

	// A genuinely empty prompt keeps the original wording and the presets.
	empty := withCtx(64)
	empty.Prompt = ""
	empty.recompute()
	if !strings.Contains(empty.Status, "还没有提示词") {
		t.Fatalf("an empty prompt reports %q", empty.Status)
	}
	tr2 := ui.NewTester(empty.View, 1500, 940)
	tr2.Frame()
	sawEmpty := false
	for _, s := range tr2.Texts() {
		if strings.Contains(s, "还没有提示词") {
			sawEmpty = true
		}
		if strings.Contains(s, "装不进这个窗口") {
			t.Fatalf("an empty prompt is described as too wide: %q", s)
		}
	}
	if !sawEmpty {
		t.Fatal("the empty prompt lost its heading")
	}

	// And a window wide enough for the character must work normally.
	fits := withCtx(4)
	fits.Prompt = "😀"
	fits.recompute()
	if n := len(fits.Trace.Tokens); n != 4 {
		t.Fatalf("a 4-byte character in a 4-token window gave %d tokens", n)
	}
}
