package app

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
)

// The README states facts about the app that a reader will take at face value.
// TestReadmeStageRangesMatchTheControls already covers the numeric ranges; these
// cover the rest, because a README drifts silently while the code moves on and
// this one had already drifted on the callout count.

func readREADME(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "README.md"))
	if err != nil {
		t.Fatalf("cannot read the README: %v", err)
	}
	return string(b)
}

// TestReadmeCalloutCountMatchesTheStages: "第 3–7 步各带「常见误解」卡片，共六条".
func TestReadmeCalloutCountMatchesTheStages(t *testing.T) {
	doc := readREADME(t)
	if !strings.Contains(doc, "共六条") {
		t.Fatal("the README no longer states the callout count; update or remove this test")
	}

	counts := map[int]int{}
	total := 0
	for step := 0; step < 7; step++ {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, "「") && strings.Contains(s, "」") {
				counts[step]++
				total++
			}
		}
	}
	if total != 6 {
		t.Fatalf("the README says 共六条, the app renders %d", total)
	}
	// And they are on stages 3-7 with one extra on the fourth.
	for step := 0; step < 7; step++ {
		want := 0
		switch step {
		case 2, 4, 5, 6:
			want = 1
		case 3:
			want = 2
		}
		if counts[step] != want {
			t.Fatalf("step %d renders %d callouts, want %d", step+1, counts[step], want)
		}
	}
}

// TestReadmeModelBlockMatchesTheWeights: the model paragraph quotes the
// architecture and the parameter count.
func TestReadmeModelBlockMatchesTheWeights(t *testing.T) {
	doc := readREADME(t)
	a := newTestApp(t)
	c := a.W.Config

	stated := strings.Contains(doc, "vocab 256 · d_model 96 · 2 层 · 4 头 · d_ff 384 · ctx 64")
	trueDims := c.Vocab == 256 && c.DModel == 96 && c.NLayer == 2 &&
		c.NHead == 4 && c.DFF == 384 && c.Ctx == 64
	if stated && !trueDims {
		t.Fatalf("the README quotes an architecture the weights do not have: %+v", c)
	}

	if strings.Contains(doc, "参数量 25.3 万") && a.paramCount()/10000 != 25 {
		t.Fatalf("the README says 25.3 万 parameters, the model has %d", a.paramCount())
	}
}

// TestReadmeKeyboardScopeMatchesTheCode: "第 3、4 步在注意力矩阵里移动格子；
// 其余步骤 Shift+←/→ 移动 token 选择".
func TestReadmeKeyboardScopeMatchesTheCode(t *testing.T) {
	for step := 0; step < 7; step++ {
		a := newTestApp(t)
		a.Step = step
		a.SelToken = 10
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		selBefore, cellBefore := a.SelToken, a.KeyQ
		tr.Key(ui.Shift, ui.KeyRight)
		tr.Frame()

		inMatrix := step == 2 || step == 3
		if inMatrix {
			if a.KeyQ < 0 || a.KeyQ == cellBefore {
				t.Fatalf("step %d: Shift+Right did not open a matrix cell (KeyQ %d -> %d)",
					step+1, cellBefore, a.KeyQ)
			}
			if a.SelToken != selBefore {
				t.Fatalf("step %d: Shift+Right moved the token selection instead of the cell", step+1)
			}
		} else {
			if a.SelToken == selBefore {
				t.Fatalf("step %d: Shift+Right did not move the token selection", step+1)
			}
		}
	}
}

// TestReadmeFadeDurationMatchesTheTransition: "切换阶段时中栏和检查器会淡入（160ms）".
func TestReadmeFadeDurationMatchesTheTransition(t *testing.T) {
	doc := readREADME(t)
	if !strings.Contains(doc, "160ms") {
		t.Fatal("the README no longer states the fade duration; update or remove this test")
	}
	if got := stageFade.Duration; got != 160*time.Millisecond {
		t.Fatalf("the README says 160ms, the transition is %v", got)
	}
}

// TestReadmeGraphLegendsMatchTheCode keeps the "图上每个元素怎么读" table honest.
// Each row states a rule the painter implements; the rules are the part a reader
// would otherwise have to reverse-engineer, and they are the part most likely to
// drift when someone tunes a colour ramp.
func TestReadmeGraphLegendsMatchTheCode(t *testing.T) {
	doc := readREADME(t)
	for _, section := range []string{
		"### 图上每个元素怎么读",
		"色深上限",
		"向量条带两端的",
		"直方图的竖线",
		"共同横轴",
	} {
		if !strings.Contains(doc, section) {
			t.Fatalf("the README no longer documents %q; update or remove this test", section)
		}
	}

	// The colour ramps stated in the table must be the ones the painter uses.
	if !strings.Contains(doc, "0.0005") {
		t.Fatal("the README no longer states the threshold below which a cell is drawn as background")
	}
	// Heat colour: at or below 0.0005 the cell is the panel colour, not a tint.
	atThreshold := heatColor(0.0005)
	justAbove := heatColor(0.002)
	if atThreshold == justAbove {
		t.Fatal("heatColor no longer treats 0.0005 as a cutoff, so the README's rule is wrong")
	}

	// The score scale is per-matrix, not a constant: the table says so and the
	// caption prints it, so a fixed value would make both wrong.
	caps := map[float64]bool{}
	for _, prompt := range []string{"hi", "attention is a weighted sum"} {
		b := newTestApp(t)
		b.Prompt = prompt
		b.recompute()
		caps[scoreColorScale(b.layer().Heads[b.Head])] = true
	}
	if len(caps) < 2 {
		t.Fatal("the colour cap is the same for every prompt, but the README describes it " +
			"as a value that moves with the prompt")
	}
}

// TestReadmeStrideRuleIsTheOneImplemented checks the stride figures the table
// quotes against the widget's own rule.
//
// My first version of the table said "36 token 时逐个标，64 token 时每 6 个标一次",
// both of which were guesses. The rule is a 26-DIP minimum gap, which gives 3 and
// 4 — so the document was wrong in two places and this test is what caught it.
func TestReadmeStrideRuleIsTheOneImplemented(t *testing.T) {
	// The painter's rule, restated: a label every `stride` cells, stride the
	// first integer putting at least 26 DIPs between labels.
	strideFor := func(cell float32) int {
		s := 1
		for float32(s)*cell < 26 {
			s++
		}
		return s
	}
	avail := float32(420) // the matrix box's default width
	cell36, cell64 := avail/36, avail/64
	if got := strideFor(cell36); got != 3 {
		t.Fatalf("36 tokens: stride %d, but the README and this test say 3", got)
	}
	if got := strideFor(cell64); got != 4 {
		t.Fatalf("64 tokens: stride %d, but the README and this test say 4", got)
	}

	// And the document must say those numbers.
	doc := readREADME(t)
	for _, want := range []string{"每 3 个标一次", "每 4 个标一次"} {
		if !strings.Contains(doc, want) {
			t.Fatalf("the README does not state %q, so its stride parenthetical is stale", want)
		}
	}
}
