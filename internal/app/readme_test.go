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
