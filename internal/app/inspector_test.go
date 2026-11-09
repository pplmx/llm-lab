package app

import (
	"fmt"
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"
)

// inspectorCards maps a lesson to a phrase that only its inspector shows. The
// dispatch in inspector() is a plain switch, so nothing catches a step wired to
// the wrong card or a card added without a case: the panel would explain the
// residual stream while the stage draws attention, and every existing test
// would still pass because they each call one inspector function directly.
var inspectorCards = []struct {
	step  int
	label string
	// marker appears only in this step's card.
	marker string
}{
	// Step 1's card adapts: the default prompt is ASCII, so it explains why bytes and
	// runes coincide. TestByteCardFollowsThePrompt covers the other branch.
	{0, "分词", "ASCII 下字节就是字符"},
	{1, "嵌入", "位置向量"},
	{2, "注意力分数", "绝对值最大的 5 个分数"},
	{3, "Softmax", "这一行的统计"},
	{4, "残差 / LN", "进入 FFN 之前"},
	{5, "FFN", "隐藏层统计"},
	{6, "输出 / 采样", "分布统计"},
}

// TestInspectorShowsTheCardForItsStage checks each step renders its own card.
func TestInspectorShowsTheCardForItsStage(t *testing.T) {
	for _, c := range inspectorCards {
		a := newTestApp(t)
		a.SelToken = 10
		a.Step = c.step
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		if !tr.HasText(c.marker) {
			t.Fatalf("step %d (%s): the inspector does not show %q", c.step+1, c.label, c.marker)
		}
	}
}

// TestInspectorShowsNoOtherStagesCard is the half that catches a wrong case
// label: a step rendering its own card plus the previous one, or a fallthrough.
func TestInspectorShowsNoOtherStagesCard(t *testing.T) {
	for _, c := range inspectorCards {
		a := newTestApp(t)
		a.SelToken = 10
		a.Step = c.step
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		for _, other := range inspectorCards {
			if other.step == c.step {
				continue
			}
			if tr.HasText(other.marker) {
				t.Fatalf("step %d (%s) also shows step %d's card (%q)",
					c.step+1, c.label, other.step+1, other.marker)
			}
		}
	}
}

// TestInspectorEmptyStatesAreDistinct checks the two dead ends say different
// things. "No tokens" and "no selection" need different actions, and one message
// for both used to send readers looking for a token that did not exist.
func TestInspectorEmptyStatesAreDistinct(t *testing.T) {
	// Empty prompt: the message names typing as the next action.
	empty := newTestApp(t)
	empty.Prompt = ""
	empty.recompute()
	tr := ui.NewTester(empty.View, 1500, 940)
	tr.Frame()
	if !tr.HasText("先在上方输入一句提示词") {
		t.Fatal("the no-tokens state does not name the input field")
	}
	if tr.HasText("点击 token 或热力图的格子") {
		t.Fatal("the no-tokens state shows the no-selection message")
	}

	// Tokens but nothing selected: the message names clicking.
	unselected := newTestApp(t)
	unselected.SelToken = -1
	tr2 := ui.NewTester(unselected.View, 1500, 940)
	tr2.Frame()
	if !tr2.HasText("点击 token 或热力图的格子") {
		t.Fatal("the no-selection state does not name the click")
	}
	if tr2.HasText("先在上方输入一句提示词") {
		t.Fatal("the no-selection state shows the no-tokens message")
	}
}

// TestInspectorFollowsTheSelectedToken checks the card's numbers actually change
// with the selection, rather than describing whichever token happened to be
// current when the card was written.
func TestInspectorFollowsTheSelectedToken(t *testing.T) {
	// Step 4's card reports the selected row's attention statistics.
	a := newTestApp(t)
	a.Step = 3
	a.SelToken = 5
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	first := ""
	for _, s := range tr.Texts() {
		if strings.HasPrefix(s, "熵 ") || strings.Contains(s, "nats") {
			first = s
		}
	}
	a.SelToken = 30
	tr.Frame()
	second := ""
	for _, s := range tr.Texts() {
		if strings.HasPrefix(s, "熵 ") || strings.Contains(s, "nats") {
			second = s
		}
	}
	if first == "" || second == "" {
		t.Fatal("no entropy readout")
	}
	if first == second {
		t.Fatalf("the inspector reports the same row for tokens 5 and 30: %q", first)
	}
}

// TestByteCardFollowsThePrompt checks that step 1's explainer matches the prompt in the field.
//
// The card was headed "为什么中文更长" whatever the prompt was, which on an ASCII prompt is a
// question the figures underneath answer with "it does not" — 36 字节 / 36 字符 is evidence against
// the premise. A reader looking at that can only conclude they have misunderstood something.
//
// Two branches now: equal byte and rune counts explain that ASCII is one byte per character, and
// unequal ones explain the UTF-8 encoding that makes CJK longer.
func TestByteCardFollowsThePrompt(t *testing.T) {
	cases := []struct {
		prompt, want, notWant string
	}{
		{"The transformer reads the tokens and", "ASCII 下字节就是字符", "为什么中文更长"},
		{"注意力就是加权求和", "为什么中文更长", "ASCII 下字节就是字符"},
	}
	for _, c := range cases {
		a := newTestApp(t)
		a.Prompt = c.prompt
		a.recompute()
		a.Step = 0
		a.SelToken = 1
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		if !tr.HasText(c.want) {
			t.Fatalf("prompt %q: the byte card does not show %q", c.prompt, c.want)
		}
		if tr.HasText(c.notWant) {
			t.Fatalf("prompt %q: the byte card still shows %q, which the prompt contradicts",
				c.prompt, c.notWant)
		}
	}

	// And the figures have to agree with the branch that was chosen.
	a := newTestApp(t)
	a.Prompt = "注意力"
	a.recompute()
	a.Step = 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	bytes, runes := len([]byte("注意力")), len([]rune("注意力"))
	want := fmt.Sprintf("%d 字节 / %d 字符", bytes, runes)
	if !tr.HasText(want) {
		t.Fatalf("the byte card does not print %q", want)
	}
	if bytes == runes {
		t.Fatal("the CJK sample has equal byte and rune counts, so it does not exercise the " +
			"branch this test is about")
	}
}

// TestDistributionCardNamesItsPosition keeps the inspector's distribution distinguishable from
// the per-position card below it.
//
// Both quote a top-1 probability, and they are different rows: this one always the last position,
// the other whichever t the reader pinned. With t=8 pinned the two read 96.95% and 93.76%, side by
// side, and only one of them named its position — a reader comparing them has no way to tell the
// numbers are about different rows.
//
// The assertion is on the subtitle, because that is the only thing on the card that could carry
// the position.
func TestDistributionCardNamesItsPosition(t *testing.T) {
	for _, pinned := range []int{-1, 8} {
		a := newTestApp(t)
		a.Step = 6
		a.PredSel = pinned
		tr := ui.NewTester(a.View, 1500, 4000)
		tr.Frame()

		var subtitle string
		for _, s := range tr.Texts() {
			if strings.Contains(s, "最后一个位置 t=") {
				subtitle = s
			}
		}
		if subtitle == "" {
			t.Fatalf("pinned=%d: the distribution card does not say which position it is for, "+
				"so its top-1 is indistinguishable from the pinned prediction's top-1", pinned)
		}
		want := fmt.Sprintf("t=%d", len(a.Trace.Tokens)-1)
		if !strings.Contains(subtitle, want) {
			t.Fatalf("pinned=%d: the distribution subtitle reads %q, not the last position %s",
				pinned, subtitle, want)
		}
	}
}

// TestAttentionBarsDiscloseTheirSample checks the bar list beside the attention heatmap says what it
// is a sample of.
//
// The list shows at most ten bars, and topBars keeps the ten largest then restores key order — so
// on a long prompt it is neither the whole row nor sorted by size. Measured with 35 keys: ten bars
// against a 35-cell heatmap row, with 0.073 drawn above 0.419. Neither the truncation nor the
// ordering was mentioned, which leaves a reader comparing the two views with no account of the
// difference.
//
// The assertion is conditional on the row being long enough to truncate, and checks the short case
// is left alone: a caption that always talks about "the ten largest" would be wrong on every row
// short enough to be shown whole.
func TestAttentionBarsDiscloseTheirSample(t *testing.T) {
	long := newTestApp(t)
	long.Step = 3
	long.SelToken = len(long.Trace.Tokens) - 1
	tr := ui.NewTester(long.View, 1500, 3000)
	tr.Frame()

	var caption string
	for _, s := range tr.Texts() {
		if strings.Contains(s, "加起来等于 1") {
			caption = s
		}
	}
	if caption == "" {
		t.Fatal("the attention row caption is missing")
	}
	if n := long.SelToken + 1; n > 10 {
		for _, want := range []string{fmt.Sprintf("%d 个 key", n), "最大的 10 个", "不是按大小排"} {
			if !strings.Contains(caption, want) {
				t.Fatalf("a %d-key row draws only 10 bars but the caption does not say %q: %q",
					n, want, caption)
			}
		}
	}

	// A short row shows every key, so it must not claim to be a sample.
	short := newTestApp(t)
	short.Step = 3
	short.SelToken = 4
	tr2 := ui.NewTester(short.View, 1500, 3000)
	tr2.Frame()
	for _, s := range tr2.Texts() {
		if !strings.Contains(s, "加起来等于 1") {
			continue
		}
		if strings.Contains(s, "最大的 10 个") {
			t.Fatalf("a %d-key row is drawn whole but the caption calls it a sample: %q",
				short.SelToken+1, s)
		}
	}
}
