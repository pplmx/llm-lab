package app

import (
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
	{0, "分词", "为什么中文更长"},
	{1, "嵌入", "位置向量"},
	{2, "注意力分数", "分数 top-5"},
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
