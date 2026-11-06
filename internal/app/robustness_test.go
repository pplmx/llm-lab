package app

import (
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
	want := map[int]string{0: "为什么中文更长", 3: "这一行的统计", 6: "分布统计"}
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
