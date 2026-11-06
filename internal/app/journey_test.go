package app

import (
	"image"
	"testing"

	"github.com/egoist/mygo/ui"
)

// TestReaderJourney is the end-to-end acceptance test for the app as a reader
// meets it. The other tests each pin one behaviour; this one walks the path a
// reader actually takes — open, walk the pipeline, inspect a cell, empty the
// prompt, recover, hit the context ceiling — and checks the seams between them.
// The bugs this project has actually hit were seam bugs: a shortcut consumed
// twice in one frame, keyboard state erased by a pointer move, a fade whose
// first frame was blank. Each was locally correct and jointly wrong.
func TestReaderJourney(t *testing.T) {
	a := newTestApp(t)
	// The smallest supported window: the journey has to hold at the tightest
	// size, which is also the one where stages actually need to scroll.
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()

	// 1. Opening state: a prompt is loaded, the reader stands on step 1, and
	// the strip has scrolled the selected token into view.
	if a.Step != 0 {
		t.Fatalf("opens on step %d, want the first", a.Step)
	}
	if len(a.Trace.Tokens) == 0 {
		t.Fatal("opens with no tokens")
	}
	if a.SelToken != len(a.Trace.Tokens)-1 {
		t.Fatalf("opens with token %d selected, want the last (%d)",
			a.SelToken, len(a.Trace.Tokens)-1)
	}
	if a.RailScroll.X <= 0 {
		t.Error("the strip did not scroll the selected token into view on open")
	}
	if r, ok := tr.Find("token " + itoa(a.SelToken)); !ok {
		t.Fatal("the selected token's chip is not in the frame")
	} else if r.X < 0 {
		t.Fatalf("the selected chip is off-screen at X=%.0f", r.X)
	}

	// 2. The pipeline: every stage renders its lesson and its inspector, and
	// none of them falls back to an empty state.
	digits := []ui.Key{ui.Key1, ui.Key2, ui.Key3, ui.Key4, ui.Key5, ui.Key6, ui.Key7}
	for i, key := range digits {
		tr.Key(0, key)
		tr.Frame()
		if a.Step != i {
			t.Fatalf("key %d left the reader on step %d", i+1, a.Step+1)
		}
		if !tr.HasText("这一段在讲什么") {
			t.Fatalf("step %d has no lesson card", i+1)
		}
		if !tr.HasText(lessons[i].name) {
			t.Fatalf("step %d: the inspector does not name the stage", i+1)
		}
		if tr.HasText("还没有提示词") {
			t.Fatalf("step %d shows the empty-prompt card with a prompt loaded", i+1)
		}
	}

	// 3. Scrolling: scroll to the bottom of the longest stage, then switch to
	// the shortest. A stale offset would leave the reader part-way down the new
	// stage. Both stages must be scrollable for this to mean anything — the
	// scroll view clamps an offset that is too large for the content, so going
	// from a long stage to a stage that cannot scroll would pass even with the
	// reset removed.
	tr.Key(0, ui.Key6) // FFN: the longest stage at 1180x720
	tr.Frame()
	a.Scroll.Y = 100000
	tr.Frame()
	tr.Frame()
	deep := a.Scroll.Y
	tr.Key(0, ui.Key1)
	tr.Frame()
	tr.Frame()
	if deep <= 0 {
		t.Fatal("the long stage did not scroll; cannot test the reset")
	}
	if a.Scroll.Y != 0 {
		t.Fatalf("switching stage left the reader at Y=%.0f; the shortest stage "+
			"would have put them part-way down", a.Scroll.Y)
	}
	// And going back to a long stage starts at its top, not at the offset the
	// reader left it at.
	tr.Key(0, ui.Key6)
	tr.Frame()
	tr.Frame()
	if a.Scroll.Y != 0 {
		t.Fatalf("returning to the long stage resumed at Y=%.0f, want its top "+
			"(the reader had scrolled to %.0f)", a.Scroll.Y, deep)
	}

	// 4. Attention: the keyboard walks to a cell and the inspector quotes that
	// same cell from the trace, not a neighbouring one.
	tr.Key(0, ui.Key3)
	tr.Frame()
	tr.Key(ui.Shift, ui.KeyDown)
	tr.Frame()
	tr.Key(ui.Shift, ui.KeyLeft)
	tr.Frame()
	q, k := a.KeyQ, a.KeyK
	if q < 0 || k < 0 {
		t.Fatal("the keyboard did not enter the attention matrix")
	}
	if k > q {
		t.Fatalf("cell %d,%d is above the diagonal", q, k)
	}
	if !tr.HasText("注意力格子") {
		t.Fatal("no inspector cell card for the keyboard's cell")
	}
	// The keyboard's cell has to be visible, not just recorded in state. This
	// is the one assertion here that needs pixels: every text-based check above
	// still passes when the heatmap stops drawing the cursor. Run it on the
	// step these keys enter first, where the matrix card sits high enough to be
	// inside the crop.
	if !keyboardCellVisible(t, 3, 1500, 940) {
		t.Fatal("the keyboard cell is set but the heatmap does not draw it")
	}
	if a.SelToken != q {
		t.Fatalf("the strip follows token %d while the cell's query is %d", a.SelToken, q)
	}
	h := a.layer().Heads[a.Head]
	if h.Scores[q][k] > 1e9 {
		t.Fatalf("cell %d,%d is masked but the keyboard put the cursor there", q, k)
	}

	// 5. Empty the prompt from a later stage, then recover from that stage.
	tr.Key(0, ui.Key5)
	tr.Frame()
	a.Prompt = ""
	a.recompute()
	tr.Frame()
	if !tr.HasText("还没有提示词") {
		t.Fatal("emptying the prompt did not show the empty-state card")
	}
	if err := tr.Click("中文"); err != nil {
		t.Fatalf("the empty state offers no usable preset: %v", err)
	}
	tr.Frame()
	if len(a.Trace.Tokens) == 0 {
		t.Fatal("choosing a preset did not restore the pipeline")
	}
	if a.Step != 4 {
		t.Fatalf("recovering jumped the reader to step %d", a.Step+1)
	}
	if tr.HasText("还没有提示词") {
		t.Fatal("the empty-state card is still showing after recovery")
	}

	// 6. The context ceiling: the long preset fills ctx exactly and the last
	// stage still renders.
	for _, p := range promptPresets {
		if p.name == "长提示词" {
			a.applyPrompt(p.prompt)
		}
	}
	tr.Frame()
	if got, want := len(a.Trace.Tokens), a.Trace.Config.Ctx; got != want {
		t.Fatalf("long preset produced %d tokens, want the %d-token ceiling", got, want)
	}
	tr.Key(0, ui.Key7)
	tr.Frame()
	if !tr.HasText("Top-12 概率") {
		t.Fatal("the output stage does not render at the context ceiling")
	}
}

// keyboardCellVisible reports whether the heatmap draws the keyboard's cell.
//
// Isolating it takes care: KeyQ/KeyK also mark the token strip and add an
// inspector card, and the selected query's row is painted with a warm bar, so
// both a frame diff and a whole-frame colour count see plenty of change whether
// or not the box is drawn. This counts warm pixels in the heatmap's own area,
// which was 55 without a cursor and 143 with one.
func keyboardCellVisible(t *testing.T, step, w, h int) bool {
	t.Helper()
	matrixWarm := func(withCell bool) int {
		a := newTestApp(t)
		a.Step = step
		if withCell {
			// The diagonal of the selected query: a drawn cell, not a masked one.
			a.KeyQ, a.KeyK = a.SelToken, a.SelToken
		}
		tr := ui.NewTester(a.View, w, h)
		tr.Frame()
		// The heatmap card sits below the strip and the step bar, inside the
		// stage column: right of the sidebar, left of the inspector.
		crop := image.Rect(int(a.SidebarW)+76, 330, w-int(a.InspectorW)-10, 760)
		return countColour(tr.Image().SubImage(crop).(*image.RGBA), colWarm)
	}
	plain, marked := matrixWarm(false), matrixWarm(true)
	t.Logf("warm pixels in the heatmap: %d without a cell, %d with one", plain, marked)
	// Well above the noise and well below the 88 the box adds.
	return marked-plain > 40
}

// countColour counts pixels close to c, allowing for the rasterizer's
// antialiasing.
func countColour(img *image.RGBA, c ui.Color) int {
	n := 0
	for i := 0; i < len(img.Pix); i += 4 {
		if absDiff(img.Pix[i], c.R) <= 10 && absDiff(img.Pix[i+1], c.G) <= 10 &&
			absDiff(img.Pix[i+2], c.B) <= 10 {
			n++
		}
	}
	return n
}

func absDiff(a, b uint8) int {
	if a > b {
		return int(a) - int(b)
	}
	return int(b) - int(a)
}

// itoa is a tiny integer formatter so this file needs no new import.// itoa is a tiny integer formatter so this file needs no new import.
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
