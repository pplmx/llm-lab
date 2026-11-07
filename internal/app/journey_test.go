package app

import (
	"fmt"
	"image"
	"strings"
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

// TestTypingBuildsThePipelineOneKeyAtATime walks the most basic action there is: a reader with an empty prompt types a
// sentence, character by character, and the whole pipeline has to keep up.
//
// The per-keystroke recompute was once missing — the view only refreshed when some other key was pressed — and the test
// for it checks the token count after typing "hi". What it does not check is that the seven stages are still coherent at
// every intermediate state: after one character there is a single token, which is the degenerate case several cards
// branch on (the score matrix is 1×1, the attention row has nothing to compare against, the residual chain has one term).
// A reader types into that state on the way to every prompt they ever use.
func TestTypingBuildsThePipelineOneKeyAtATime(t *testing.T) {
	a := newTestApp(t)
	a.Prompt = ""
	a.recompute()
	a.Step = 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	if !tr.HasText("还没有提示词") {
		t.Fatal("the empty prompt does not show its empty-state card")
	}

	if err := tr.Click("输入"); err != nil {
		t.Fatalf("cannot focus the prompt field: %v", err)
	}
	tr.Frame()

	type keyPress struct {
		key  ui.Key
		text string
	}
	sentence := "Hi there"
	var presses []keyPress
	for _, r := range sentence {
		if r == ' ' {
			presses = append(presses, keyPress{ui.KeySpace, " "})
			continue
		}
		presses = append(presses, keyPress{ui.KeyH, string(r)})
	}

	for i, p := range presses {
		tr.TypeKey(0, p.key, p.text)
		tr.Frame()

		want := i + 1
		if got := len(a.Trace.Tokens); got != want {
			t.Fatalf("after typing %q (%s) the pipeline has %d tokens, want %d",
				string(rune(sentence[i])), a.Prompt, got, want)
		}
		if tr.HasText("还没有提示词") {
			t.Fatalf("the empty-state card is still up after typing %q", a.Prompt)
		}
		// The stage must still describe what is on screen rather than the state before the key.
		if !tr.HasText("这一段在讲什么") {
			t.Fatalf("stage 1 lost its lesson card after typing %q", a.Prompt)
		}
		if !tr.HasText(fmt.Sprintf("T = %d", want)) {
			t.Fatalf("after typing %q the tokenize card does not say T = %d", a.Prompt, want)
		}
	}

	// And the single-token state every prompt passes through has to render on every stage.
	single := newTestApp(t)
	single.Prompt = "x"
	single.recompute()
	if got := len(single.Trace.Tokens); got != 1 {
		t.Fatalf("the one-character prompt made %d tokens", got)
	}
	for step := 0; step < len(lessons); step++ {
		single.Step = step
		single.SelToken = 0
		tr := ui.NewTester(single.View, 1500, 940)
		tr.Frame()
		if tr.HasText("还没有提示词") {
			t.Fatalf("stage %d shows the empty-state card with a one-character prompt", step+1)
		}
		if !tr.HasText("这一段在讲什么") {
			t.Fatalf("stage %d has no lesson card with a one-character prompt", step+1)
		}
		if !tr.HasText(lessons[step].name) {
			t.Fatalf("stage %d's inspector does not name it with a one-character prompt", step+1)
		}
	}

	// Finally the full sentence has to render on every stage too.
	for step := 0; step < len(lessons); step++ {
		a.Step = step
		a.SelToken = len(a.Trace.Tokens) - 1
		tr.Frame()
		if !tr.HasText("这一段在讲什么") {
			t.Fatalf("stage %d has no lesson card for %q", step+1, a.Prompt)
		}
	}
}

// TestClearingThePromptResetsEverythingThatPointedIntoIt is the lifecycle rule from CONTRIBUTING, checked in one place.
//
// The app keeps several pieces of state that are indices into the current prompt: which token is selected, which matrix
// cell the keyboard is on, which cell the pointer is over, and the last sample's description. None of them mean anything
// once the prompt they index is gone, and a stale one is not harmless — SelToken would point past the end, and LastSample
// would describe a draw from text the reader no longer sees.
//
// Each has a test of its own for the action that should clear it. This covers the one action that invalidates all of them
// at once, and states the one thing that must NOT reset: the stage. Clearing the prompt is not "go back to the start".
func TestClearingThePromptResetsEverythingThatPointedIntoIt(t *testing.T) {
	a := newTestApp(t)
	a.Step = 2
	a.SelToken = 10
	a.KeyQ, a.KeyK = 5, 3
	a.HoverQ, a.HoverK = 4, 2
	a.LastSample = `"x" (id 1, p=0.500)`

	a.applyPrompt("")
	if len(a.Trace.Tokens) != 0 {
		t.Fatalf("the cleared prompt still has %d tokens", len(a.Trace.Tokens))
	}

	if a.SelToken >= 0 {
		t.Errorf("SelToken is %d with no tokens to select", a.SelToken)
	}
	if a.KeyQ >= 0 || a.KeyK >= 0 {
		t.Errorf("the keyboard cell is still %d,%d with no matrix", a.KeyQ, a.KeyK)
	}
	if a.HoverQ >= 0 || a.HoverK >= 0 {
		t.Errorf("the hovered cell is still %d,%d with no matrix", a.HoverQ, a.HoverK)
	}
	if a.LastSample != "" {
		t.Errorf("the last sample still reads %q, describing a draw from text that is gone", a.LastSample)
	}

	// The stage is a position in the lesson, not an index into the prompt.
	if a.Step != 2 {
		t.Errorf("clearing the prompt moved the reader from stage 3 to stage %d", a.Step+1)
	}

	// And the frame has to agree: the empty state, on the stage the reader was already on.
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	if !tr.HasText("还没有提示词") {
		t.Error("the frame does not show the empty state after clearing")
	}

	// Typing one character brings every one of them back to a valid state.
	a.applyPrompt("x")
	if a.SelToken < 0 || a.SelToken >= len(a.Trace.Tokens) {
		t.Fatalf("after typing, SelToken is %d and there are %d tokens", a.SelToken, len(a.Trace.Tokens))
	}
}

// TestStageSwitchKeepsTheTokenAndDropsTheCell draws the line between the three kinds of positional state.
//
// Seven stages describe the same prompt, so the selected token is meaningful in all of them: a reader who looks at token
// 10's embedding and then moves to the score stage wants token 10's scores, not token 35's. The keyboard's matrix cell and
// the hovered cell are the opposite — they name a position inside one stage's drawing, and number 10 in the softmax matrix
// is not the same place as number 10 in the residual strips.
//
// So one survives a switch and two are cleared. That split is a decision, and nothing stated it in one place; each half was
// covered incidentally by tests for other things.
func TestStageSwitchKeepsTheTokenAndDropsTheCell(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	a.SelToken = 10
	a.KeyQ, a.KeyK = 5, 3
	a.HoverQ, a.HoverK = 4, 2

	// Leaving an attention stage for one that draws no matrix.
	a.selectStep(0)
	if a.SelToken != 10 {
		t.Fatalf("switching to stage 1 moved the selection to token %d; the token means the same thing on every stage",
			a.SelToken)
	}
	if a.KeyQ >= 0 || a.KeyK >= 0 {
		t.Errorf("the matrix cell survived a switch to a stage with no matrix: %d,%d", a.KeyQ, a.KeyK)
	}
	if a.HoverQ >= 0 || a.HoverK >= 0 {
		t.Errorf("the hovered cell survived a switch to a stage with no matrix: %d,%d", a.HoverQ, a.HoverK)
	}

	// Coming back does not resurrect the old cell: it belonged to a drawing the reader had left.
	a.selectStep(3)
	if a.SelToken != 10 {
		t.Fatalf("returning to the softmax stage moved the selection to token %d", a.SelToken)
	}
	if a.KeyQ >= 0 || a.KeyK >= 0 {
		t.Errorf("returning to the matrix stage restored a stale cell %d,%d", a.KeyQ, a.KeyK)
	}

	// And with a cell set, switching within the two matrix stages clears it as well — the rows mean different things.
	a.KeyQ, a.KeyK = 6, 2
	a.selectStep(2)
	if a.KeyQ >= 0 || a.KeyK >= 0 {
		t.Errorf("the cell survived moving between two matrix stages: %d,%d", a.KeyQ, a.KeyK)
	}
	if a.SelToken != 10 {
		t.Fatalf("moving between matrix stages moved the selection to token %d", a.SelToken)
	}
}

// TestSamplingSelectsWhatItProduced covers the one action where new content appears.
//
// recompute keeps the reader's selected token when the prompt is extended, which is right for typing: a keystroke
// should not move the selection out from under someone reading a row. Sampling is the exception — the reader pressed a
// button asking for a new byte, and before this the only sign it had happened on six of the seven stages was the token
// count ticking up in the status line. The new chip was inert, the inspector still described the old token, and the
// record of the draw is only printed on the output stage.
//
// So sampling selects the token it just made. This checks that on both a short prompt and one at the context ceiling,
// where truncation shifts every index.
func TestSamplingSelectsWhatItProduced(t *testing.T) {
	check := func(name, prompt string, step int) {
		t.Helper()
		a := newTestApp(t)
		a.Prompt = prompt
		a.recompute()
		a.Step = step
		// Deliberately not the last token: if sampling left it alone, this is what would survive.
		a.SelToken = 0
		before := len(a.Trace.Tokens)

		a.sampleNext()

		after := len(a.Trace.Tokens)
		// Below the ceiling the prompt grows by one byte and so does the token list; at the ceiling the
		// window stays full and the oldest byte falls off instead.
		if want := before + 1; before < a.W.Config.Ctx && after != want {
			t.Fatalf("%s: sampling took the token count from %d to %d, want %d", name, before, after, want)
		}
		if before >= a.W.Config.Ctx && after != a.W.Config.Ctx {
			t.Fatalf("%s: sampling at the ceiling left %d tokens, want the window's %d",
				name, after, a.W.Config.Ctx)
		}
		if a.SelToken != after-1 {
			t.Fatalf("%s: after sampling there are %d tokens and the selection is %d; it should be the one just "+
				"produced", name, after, a.SelToken)
		}
		if a.LastSample == "" {
			t.Fatalf("%s: sampling recorded nothing", name)
		}

		// The chip has to be in the frame and marked as selected, so the reader sees where the new byte is.
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		if r, ok := tr.Find(fmt.Sprintf("token %d", a.SelToken)); !ok {
			t.Fatalf("%s: the selected chip is not in the frame", name)
		} else if r.X < 0 || r.X > 1400 {
			t.Fatalf("%s: the selected chip sits at x=%.0f, outside the strip", name, r.X)
		}
	}

	check("short prompt", "The", 0)
	check("mid stage", "The transformer", 2)
	check("at the ceiling", strings.Repeat("x", 64), 6)
}

// TestAutoplayWalksThePipelineOnTheClock checks that autoplay actually advances.
//
// Every existing assertion about it checks the Playing flag: space sets it, escape clears it, selecting a stage clears it.
// None checked that time passing moves the reader, so the a.Step++ inside play() could have been dropped and the suite would
// still be green — the flag would flip, the button would say "playing", and nothing would move.
//
// The tester's clock is driven explicitly, so the interval is exact rather than slept through.
func TestAutoplayWalksThePipelineOnTheClock(t *testing.T) {
	a := newTestApp(t)
	a.Step = 0
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()

	tr.Key(0, ui.KeySpace)
	tr.Frame()
	if !a.Playing {
		t.Fatal("space did not start autoplay")
	}

	// One interval, one stage.
	for want := 1; want <= 3; want++ {
		tr.Advance(playStepInterval)
		tr.Frame()
		if a.Step != want {
			t.Fatalf("after %d intervals the reader is on stage %d, want %d", want, a.Step+1, want+1)
		}
		if !a.Playing {
			t.Fatalf("autoplay stopped on its own after %d intervals, at stage %d", want, a.Step+1)
		}
	}

	// Every stage a switch lands on starts at its top, exactly as a manual switch does.
	if a.Scroll.Y != 0 {
		t.Fatalf("autoplay left the stage scrolled to Y=%.0f", a.Scroll.Y)
	}

	// Run to the end: it must stop, not wrap, and not overshoot.
	for i := 0; i < len(lessons)+2 && a.Playing; i++ {
		tr.Advance(playStepInterval)
		tr.Frame()
	}
	if a.Playing {
		t.Fatal("autoplay never stopped at the last stage")
	}
	if a.Step != len(lessons)-1 {
		t.Fatalf("autoplay stopped on stage %d, want the last (%d)", a.Step+1, len(lessons))
	}

	// Escape interrupts mid-play, and the reader keeps the stage they had reached.
	b := newTestApp(t)
	b.Step = 0
	tr2 := ui.NewTester(b.View, 1180, 720)
	tr2.Frame()
	tr2.Key(0, ui.KeySpace)
	tr2.Frame()
	tr2.Advance(playStepInterval)
	tr2.Frame()
	reached := b.Step
	if reached == 0 {
		t.Fatal("autoplay advanced nothing before the interrupt")
	}
	tr2.Key(0, ui.KeyEscape)
	tr2.Frame()
	if b.Playing {
		t.Fatal("escape did not stop autoplay")
	}
	tr2.Advance(2 * playStepInterval)
	tr2.Frame()
	if b.Step != reached {
		t.Fatalf("the stage moved from %d to %d after stopping", reached+1, b.Step+1)
	}
}

// TestPauseResumeAndRestartAtTheEnd covers what space does when playback is already under way or already over.
//
// The autoplay test checks that it starts and stops. The three cases here are the ones a reader will actually hit while
// using it, and each has a defensible-but-different answer:
//
//   - pausing keeps the stage, so resuming continues from there rather than starting the tour over;
//   - resuming continues on the clock, without replaying the interval that was already spent;
//   - pressing space on the last stage restarts from the first, because otherwise the play indicator lights up, nothing
//     moves, and it goes out again — measured: with the restart removed, Playing is true at stage 7, an interval passes,
//     and Playing turns false with the stage unchanged.
func TestPauseResumeAndRestartAtTheEnd(t *testing.T) {
	a := newTestApp(t)
	a.Step = 0
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()
	tr.Key(0, ui.KeySpace)
	tr.Frame()
	tr.Advance(playStepInterval)
	tr.Frame()
	tr.Advance(playStepInterval)
	tr.Frame()
	if a.Step != 2 {
		t.Fatalf("two intervals put the reader on stage %d, want 3", a.Step+1)
	}

	// Pause: the stage holds.
	tr.Key(0, ui.KeySpace)
	tr.Frame()
	if a.Playing {
		t.Fatal("space did not pause playback")
	}
	if a.Step != 2 {
		t.Fatalf("pausing moved the reader to stage %d", a.Step+1)
	}
	// And time no longer moves it.
	tr.Advance(3 * playStepInterval)
	tr.Frame()
	if a.Step != 2 {
		t.Fatalf("the stage advanced while paused: %d", a.Step+1)
	}

	// Resume: from here, not from the first stage.
	tr.Key(0, ui.KeySpace)
	tr.Frame()
	if !a.Playing {
		t.Fatal("space did not resume playback")
	}
	if a.Step != 2 {
		t.Fatalf("resuming restarted at stage %d instead of continuing from 3", a.Step+1)
	}
	tr.Advance(playStepInterval)
	tr.Frame()
	if a.Step != 3 {
		t.Fatalf("after resuming, one interval moved the reader to stage %d, want 4", a.Step+1)
	}

	// At the last stage, space starts the tour again rather than lighting a dead indicator.
	b := newTestApp(t)
	b.Step = len(lessons) - 1
	tr2 := ui.NewTester(b.View, 1180, 720)
	tr2.Frame()
	tr2.Key(0, ui.KeySpace)
	tr2.Frame()
	if !b.Playing {
		t.Fatal("space on the last stage did not start playback")
	}
	if b.Step != 0 {
		t.Fatalf("space on the last stage left the reader on stage %d; it should start the tour over", b.Step+1)
	}
	// And it is genuinely playing from there.
	tr2.Advance(playStepInterval)
	tr2.Frame()
	if b.Step != 1 {
		t.Fatalf("after restarting from the end, an interval moved the reader to stage %d, want 2", b.Step+1)
	}
}

// TestEveryShortcutHasAVisibleAlternative is the answer to "the help is below the fold".
//
// The shortcut line sits at the bottom of the lesson card, and at 1180×720 it is off-screen on all seven stages — measured,
// it lands between y=852 and y=1538 in a 720-tall window. The matrix stages already work around this by repeating their own
// two keys in the step bar, which is always visible. That works because every other shortcut has something on screen the
// reader can click instead; this checks that, so the day a button is removed the fact that its key was the only path is a
// test failure rather than something a reader discovers.
//
// The mapping is: arrows → the previous/next buttons, space → the play button, enter → the sample button, 1–7 → the sidebar
// stage list. Shift+arrows are the exception and are handled by the step bar directly, which the last case checks.
func TestEveryShortcutHasAVisibleAlternative(t *testing.T) {
	a := newTestApp(t)
	a.Step = 3
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()

	visible := func(label string) bool {
		r, ok := tr.Find(label)
		return ok && r.Y+r.H <= 720 && r.X+r.W <= 1180
	}
	for _, label := range []string{"▶ 自动播放", "← 上一步", "下一步 →"} {
		if !visible(label) {
			t.Errorf("%q is not on screen at 1180×720, so its keyboard shortcut has no alternative there", label)
		}
	}
	// The sample button is in the top bar, which never scrolls.
	if !visible("采样下一个 token") {
		t.Error("the sample button is not on screen at 1180×720, so Enter has no alternative")
	}
	// Every stage is reachable by clicking the sidebar, which is also always on screen.
	for i := range lessons {
		if !visible(fmt.Sprintf("步骤 %d %s", i+1, lessons[i].name)) {
			t.Errorf("stage %d is not clickable at 1180×720, so its digit key has no alternative", i+1)
		}
	}

	// The matrix keys are the ones with no button, so the step bar states them where they apply.
	for _, step := range []int{2, 3} {
		b := newTestApp(t)
		b.Step = step
		bt := ui.NewTester(b.View, 1180, 720)
		bt.Frame()
		r, ok := bt.Find("⇧ + 方向键 移动格子 · 点击固定")
		if !ok {
			t.Fatalf("stage %d draws a matrix but the step bar does not state the key for it", step+1)
		}
		if r.Y+r.H > 720 {
			t.Fatalf("stage %d states the matrix key below the fold at y=%.0f", step+1, r.Y)
		}
	}
}
