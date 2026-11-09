package app

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"
)

// TestWindowMinimumMatchesTheSmallestSnapshot ties the window's minimum size to the
// narrowest layout the snapshot suite covers.
//
// main.go pins the window to at least 1180×720, and layouts_test.go renders 1180×720
// as its smallest case. Nothing connects the two: removing the minimum lets a reader
// shrink the window to where the stage bar's buttons sit under the inspector panel
// (measured at 900×600 — "下一步" covers "← 上一步"), and widening the snapshot's
// smallest size would leave the window free to go below what is tested. Either change
// alone is silent.
//
// The pairing is what this asserts: the smallest rendered size and the window floor
// are the same numbers.
func TestWindowMinimumMatchesTheSmallestSnapshot(t *testing.T) {
	// The floor lives in app now, so it can be checked as a value rather than as a string
	// in a source file. The old version matched "MinWidth:  1180" against main.go's text,
	// which broke on a whitespace change and could not see the flag clamp at all — the
	// two places that enforce the floor are main.go's window options and its -w/-h
	// handling, and the string match only knew about the first.
	if MinWindowWidth != 1180 || MinWindowHeight != 720 {
		t.Fatalf("the window floor is %d×%d; the layouts are only verified down to 1180×720",
			MinWindowWidth, MinWindowHeight)
	}
	for _, c := range []struct{ w, h, wantW, wantH int }{
		{900, 600, 1180, 720},    // below both
		{1180, 720, 1180, 720},   // exactly at the floor: unchanged
		{1920, 1080, 1920, 1080}, // above: unchanged
		{1180, 400, 1180, 720},   // one axis below
	} {
		if gotW, gotH := ClampWindowSize(c.w, c.h); gotW != c.wantW || gotH != c.wantH {
			t.Fatalf("ClampWindowSize(%d, %d) = (%d, %d), want (%d, %d)",
				c.w, c.h, gotW, gotH, c.wantW, c.wantH)
		}
	}

	// The value test above proves ClampWindowSize works; it does not prove main.go calls
	// it. That was the one thing the old string match did cover, so it stays — main.go is
	// package main and cannot be linked into this test, leaving the source as the only
	// handle on it.
	main, err := os.ReadFile(filepath.Join("..", "..", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"app.ClampWindowSize(*width, *height)",
		"MinWidth:  app.MinWindowWidth",
		"MinHeight: app.MinWindowHeight",
	} {
		if !strings.Contains(string(main), want) {
			t.Fatalf("main.go no longer contains %q, so the window can be opened below the "+
				"size the layouts are verified at", want)
		}
	}

	// The clamp has to report itself. ClampWindowSize's own doc comment says the flags exist
	// "so a screenshot can be taken at a specific size" — raising -w 800 to 1180 in silence
	// therefore produces a screenshot at a size nobody asked for, with nothing to explain it.
	// This is the same defect -step had: the reader asked for one thing and got another.
	if !strings.Contains(string(main), "raised to %dx%d") {
		t.Fatalf("main.go clamps the window size without saying so; `-w 800` opens at 1180 " +
			"with no indication that the requested size was replaced")
	}
	// It must warn rather than refuse: a size below the verified floor still satisfies what
	// the flag means, unlike an out-of-range -step. A log.Fatalf here would break the
	// documented "小屏" recipe in the README.
	if strings.Contains(string(main), `log.Fatalf("-w`) || strings.Contains(string(main), `log.Fatalf("-h`) {
		t.Fatalf("main.go rejects an undersized window instead of raising it; a request for a " +
			"small window is satisfiable at the floor, and the README documents that path")
	}

	// And the snapshot suite must still render at exactly that size, so the
	// guarantee is demonstrated rather than assumed.
	render, err := os.ReadFile("render_sizes_test.go")
	if err != nil {
		t.Fatal(err)
	}
	want := "{1180, 720}"
	if !strings.Contains(string(render), want) {
		t.Fatalf("render_sizes_test.go no longer renders %s, so nothing shows the "+
			"layout works at the window's minimum size", want)
	}
}

// TestEveryStageIsRenderedAtEverySize keeps the snapshot matrix complete.
//
// It used to cover stages 0, 3, 5 and 6 only — four of seven. The three skipped ones
// were not accidental omissions of equivalent content: stage 2 is the embedding card
// with three strips and a histogram, stage 3 is the score heatmap, and stage 5 is the
// residual stage, which stacks six strips and is by far the tallest card. That last
// one is the most likely to overflow and had no frame at the smallest window.
//
// The claim this suite backs is "no overflow from 1180×720 to 1920×1080", and that is
// a claim about all seven stages.
func TestEveryStageIsRenderedAtEverySize(t *testing.T) {
	src, err := os.ReadFile("render_sizes_test.go")
	if err != nil {
		t.Fatal(err)
	}
	text := string(src)
	if !strings.Contains(text, "[]int{0, 1, 2, 3, 4, 5, 6}") {
		t.Fatalf("the size matrix no longer renders every stage; it should iterate " +
			"all seven so the overflow claim covers them")
	}
	// And the three sizes must all still be there.
	for _, size := range []string{"{1180, 720}", "{1500, 940}", "{1920, 1080}"} {
		if !strings.Contains(text, size) {
			t.Fatalf("the size matrix no longer includes %s", size)
		}
	}
}

// TestEveryStageRendersEmptyAtEverySize covers the state the snapshot set reaches only
// once.
//
// The empty prompt is the first thing a reader sees, and it is the state where the app has
// the least to show: no tokens, no matrix, no inspector cards. Round 142's defect lived
// here — the two matrix stages advertised cell keys over a picture that was not drawn. The
// snapshots caught it only because an empty state for step 2 was added; the guard for the
// rest of the surface is a render of every stage at every tested size, which is cheap and
// fails loudly if any stage stops coping with an empty trace.
//
// Sizes match the ones the snapshot suite uses, so the guarantee is the same one a reader
// gets at any of the three window sizes the app supports.
func TestEveryStageRendersEmptyAtEverySize(t *testing.T) {
	for _, size := range [][2]int{{1180, 720}, {1500, 940}, {1920, 1080}} {
		for step := 0; step < len(lessons); step++ {
			a := newTestApp(t)
			a.Prompt = ""
			a.recompute()
			a.Step = step
			img := ui.Render(a.View, size[0], size[1], 1)
			if img.Pix == nil || len(img.Pix) == 0 {
				t.Fatalf("stage %d renders nothing at %dx%d with an empty prompt",
					step, size[0], size[1])
			}
			// The step must survive the render: a stage that fell back to 0 would
			// still produce an image.
			if a.Step != step {
				t.Fatalf("rendering stage %d with an empty prompt moved it to %d", step, a.Step)
			}
		}
	}
}

// TestShortPromptsRenderWithoutNaNAcrossEveryStage covers the smallest non-empty inputs.
//
// A one-byte prompt makes 1×1 matrices, one-element softmax rows and single-position
// residual streams — every loop that divides by a count or a spread has its worst case
// there. A NaN or an Inf reaching the screen would print as "NaN" beside a number the
// reader is meant to trust, and no snapshot covers a one-token prompt.
//
// The attention paragraph legitimately prints "-Inf" when it explains the causal mask, so
// the check looks for the bare numeric forms and allows that sentence.
func TestShortPromptsRenderWithoutNaNAcrossEveryStage(t *testing.T) {
	const maskSentence = "真正让 key 彻底出局的只有右上角的因果掩码"
	for _, prompt := range []string{"a", "中", "a中", "\xff"} {
		for step := 0; step < len(lessons); step++ {
			a := newTestApp(t)
			a.Prompt = prompt
			a.recompute()
			a.Step = step
			tr := ui.NewTester(a.View, 1500, 940)
			tr.Frame()
			for _, s := range tr.Texts() {
				if strings.Contains(s, maskSentence) {
					continue
				}
				for _, bad := range []string{"NaN", "+Inf", "-Inf", "Infinity"} {
					if strings.Contains(s, bad) {
						t.Fatalf("prompt %q, stage %d renders %q in %q",
							prompt, step, bad, s)
					}
				}
			}
		}
	}
}

// TestFixedChromeStaysInsideTheWindow checks the parts that do not scroll.
//
// The snapshot suite is looked at by eye, and the goal asks for "不溢出、不裁切" at every
// supported size. Nothing asserted it: a stage whose header ran off the bottom would appear
// in a PNG and nobody would notice until they opened the file.
//
// The test uses Tester.Find, which returns an element's layout rect, and checks the anchors
// that must be visible on every stage at every size — the app title, the sidebar's model
// block, the inspector's header and the play button. Scrollable stage content is deliberately
// excluded: it is allowed to sit below the fold, which is what the scroll view is for.
//
// Five sizes spanning the supported range, including two that are not in the snapshot set.
func TestFixedChromeStaysInsideTheWindow(t *testing.T) {
	anchors := []string{
		"LLM Lab", // the app title in the sidebar
		"vocab",   // the sidebar's model block
		"参数",      // the model block's parameter row
		"检查器",     // the inspector's header
		"▶ 自动播放",  // the stage bar's play button
	}
	for _, size := range [][2]int{
		{1180, 720}, {1280, 800}, {1500, 940}, {1700, 1000}, {1920, 1080},
	} {
		for step := 0; step < len(lessons); step++ {
			a := newTestApp(t)
			a.Step = step
			tr := ui.NewTester(a.View, size[0], size[1])
			tr.Frame()
			for _, s := range anchors {
				r, ok := tr.Find(s)
				if !ok {
					t.Fatalf("%dx%d stage %d: %q is not rendered", size[0], size[1], step, s)
				}
				if r.X < 0 || r.Y < 0 ||
					r.X+r.W > float32(size[0]) || r.Y+r.H > float32(size[1]) {
					t.Fatalf("%dx%d stage %d: %q sits at (%.0f,%.0f) %.0fx%.0f, outside the "+
						"window", size[0], size[1], step, s, r.X, r.Y, r.W, r.H)
				}
			}
		}
	}
}

// TestExtremeAspectRatiosKeepTheChromeInside covers window shapes the size floor does not
// constrain.
//
// MinWidth and MinHeight bound each axis independently, so 1180×1440 and 2560×720 are both
// legal — a reader can drag the window into a tall column or an ultrawide strip. The snapshot
// set only covers the three diagonal sizes, so neither shape was ever rendered.
//
// Measured: all seven stages keep the title, the inspector header and the play button inside
// the window at both extremes.
//
// What this cannot see: Tester.Find returns a text element's rect, not its container's, so an
// over-subscribed row that squeezes a column narrow still passes — the label stays on screen,
// just cramped. An attempt to catch that with column ordering did not work either: the columns
// keep their order while being compressed. The floor in ClampWindowSize is what actually keeps
// the layout out of that regime, and TestWindowMinimumMatchesTheSmallestSnapshot covers it.
func TestExtremeAspectRatiosKeepTheChromeInside(t *testing.T) {
	for _, size := range [][2]int{
		{1180, 1440}, // tall: the width floor, unbounded height
		{2560, 720},  // ultrawide: the height floor, unbounded width
	} {
		for step := 0; step < len(lessons); step++ {
			a := newTestApp(t)
			a.Step = step
			tr := ui.NewTester(a.View, size[0], size[1])
			tr.Frame()
			for _, s := range []string{"LLM Lab", "检查器", "▶ 自动播放"} {
				r, ok := tr.Find(s)
				if !ok {
					t.Fatalf("%dx%d stage %d: %q is not rendered", size[0], size[1], step, s)
				}
				if r.X < 0 || r.Y < 0 ||
					r.X+r.W > float32(size[0]) || r.Y+r.H > float32(size[1]) {
					t.Fatalf("%dx%d stage %d: %q sits at (%.0f,%.0f) %.0fx%.0f, outside the "+
						"window", size[0], size[1], step, s, r.X, r.Y, r.W, r.H)
				}
			}
		}
	}
}

// TestBothUntrainedPathsAreSnapshot keeps a picture of each way into an untrained model.
//
// untrainedWhy gives different advice depending on how the app got there: the bundled file failing
// to load, or -weights pointing at a file with no training metadata. Claiming the bundled file was
// missing when the reader supplied their own would send them looking for the wrong problem.
//
// The prose branch had a unit test but no snapshot, so a layout problem specific to the longer
// second message — it says "（steps = 0）" and names no file — would not have shown up in the
// comparison set.
func TestBothUntrainedPathsAreSnapshot(t *testing.T) {
	src, err := os.ReadFile("render_sizes_test.go")
	if err != nil {
		t.Fatal(err)
	}
	text := string(src)
	for _, name := range []string{
		"size-untrained-1180.png",        // bundled file missing
		"size-untrained-custom-1180.png", // -weights with steps = 0
	} {
		// Quoted, so renaming the file to "anything-size-untrained-custom-1180.png" still fails.
		if !strings.Contains(text, `"`+name+`"`) {
			t.Fatalf("the snapshot set no longer renders %q", name)
		}
	}
	// And the custom case has to actually rename the source, or it silently renders the
	// bundled-missing banner a second time.
	if !strings.Contains(text, `custom.Source = `) {
		t.Fatal("the custom-weights snapshot does not change Source, so it shows the other " +
			"untrained message")
	}
}

// TestAllNegativeMatrixIsSnapshot keeps a picture of the branch where a head has no positive scores.
//
// Every bundled prompt has both signs in every head, so the legend's new data-driven list renders
// the same two swatches in all 40 other snapshots — the branch that hides one of them would never be
// compared. A two-token prompt reaches it.
func TestAllNegativeMatrixIsSnapshot(t *testing.T) {
	src, err := os.ReadFile("render_sizes_test.go")
	if err != nil {
		t.Fatal(err)
	}
	text := string(src)
	if !strings.Contains(text, `"size-allnegative-1180.png"`) {
		t.Fatal("no snapshot renders a matrix whose scores are all negative")
	}
	if !strings.Contains(text, `neg.Prompt = "hi"`) {
		t.Fatal("the all-negative snapshot does not use the short prompt that produces the case")
	}
}

// TestEdgeCaseBranchesAreSnapshot keeps a picture of the states the default snapshot prompts never
// reach.
//
// Three cards change shape at an input extreme: step 1's byte card switches its question (ASCII
// versus CJK), step 2's position table switches from "the first four positions" to "this prompt only
// has N", and step 4's entropy chart gives way to a sentence on row 0, where the causal mask leaves
// one key and every head's entropy is exactly 0. A fourth snapshot covers a head whose scores are all
// negative, where the legend must drop its warm swatch.
//
// The bundled prompts are all at least two tokens and never select row 0, so without these four
// files none of those branches would be compared.
func TestEdgeCaseBranchesAreSnapshot(t *testing.T) {
	src, err := os.ReadFile("render_sizes_test.go")
	if err != nil {
		t.Fatal(err)
	}
	text := string(src)
	// Match the quoted filename, not the bare one: ""size-rowzero-1180.png"" fails to match
	// "DELETED-size-rowzero-1180.png" while a substring test for the bare name would pass, which
	// is how all four of these fences read as toothless when mutated.
	for _, c := range []struct{ name, setup string }{
		{"size-onetoken-1180.png", `one.Prompt = "a"`},
		{"size-allnegative-1180.png", `neg.Prompt = "hi"`},
		{"size-rowzero-1180.png", `row0.SelToken = 0`},
	} {
		if !strings.Contains(text, `"`+c.name+`"`) {
			t.Fatalf("the snapshot set no longer renders %q", c.name)
		}
		if !strings.Contains(text, c.setup) {
			t.Fatalf("%q does not set up the state it is named for (%s)", c.name, c.setup)
		}
	}
}

// TestMainValidatesStep keeps `-step` a validated flag rather than a silently clamped one.
//
// It used to be applied only when in range: `if *step >= 0 && *step < len(LessonNames())`.
// A reader asking for `-step 99` therefore got lesson 0 with no indication that the lesson
// they named does not exist, which is indistinguishable from the flag being ignored.
//
// main.go is package main and cannot be linked into this test, so — like the CLAMP check
// above — the source is the only handle. What is asserted is the shape of the handling, not
// its wording: a bounds test on the flag with a non-silent response, and that the in-range
// path is still there.
func TestMainValidatesStep(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "..", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	main := string(src)

	// The out-of-range case must be fatal, and it must name the range. `log.Fatalf` is what
	// stops the program; a plain Printf would print and continue into the old silent clamp.
	if !strings.Contains(main, "log.Fatalf(\"-step %d is out of range") {
		t.Fatalf("main.go no longer rejects an out-of-range -step; the flag is silently " +
			"clamped again, so -step 99 opens lesson 1 with no warning")
	}
	// And the message has to say what the valid values are, not merely that the value was
	// wrong: the reader needs to know 0-6 is the range.
	for _, want := range []string{"len(names)-1", "strings.Join(names"} {
		if !strings.Contains(main, want) {
			t.Fatalf("main.go's -step error no longer names the valid range (%q): a message "+
				"that only says the value is wrong leaves the reader guessing", want)
		}
	}
	// The lesson list must come from the app, so the range cannot drift from the lessons.
	if !strings.Contains(main, "app.LessonNames()") {
		t.Fatal("main.go no longer asks the app for the lesson count, so the -step range " +
			"can disagree with the lessons that exist")
	}
	// The valid path still sets the stage unconditionally, which is only safe because of the
	// check above it — a future edit that re-adds a condition here would silently reintroduce
	// the clamp.
	if !strings.Contains(main, "\ta.Step = *step\n") {
		t.Fatal("main.go no longer applies a validated -step to the app's stage")
	}
}

// TestMainDoesNotPrefixLoadErrors keeps the weights error a caller prints verbatim.
//
// The two failure modes it guards are the ones that existed together before round 12:
// main.go wrapped the error with "weights <path>: ", and model.Load passed os.ReadFile's
// error through untouched, so a missing file printed "weights /x.json: open /x.json: no such
// file..." while a schema error printed "weights /x.json: weights: the file parsed but has no
// configuration" — the same word twice in one sentence, because Parse had already prefixed its
// own. Removing either half alone leaves one of the two wrong.
//
// model.Load now names its own operation, so main.go prints the error as-is. main.go is
// package main and cannot be linked here, so — like TestMainValidatesStep beside it — the
// source is the handle: the format string must not re-add a path or a "weights" label.
func TestMainDoesNotPrefixLoadErrors(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "..", "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	main := string(src)

	// The load failure must be printed without main.go adding its own context.
	if !strings.Contains(main, "log.Fatalf(\"%v\", err)") {
		t.Fatalf("main.go no longer prints model.Load's error verbatim, so it is adding a " +
			"prefix the model layer already provides (this is what produced " +
			"\"weights /x.json: weights: the file parsed but has no configuration\")")
	}
	// And the old form must not come back by another route.
	if strings.Contains(main, "weights %s: %v") {
		t.Fatalf("main.go still prefixes the weights path onto an error that already names " +
			"the file and the operation")
	}
	// The context has to come from somewhere: model.Load builds it, so its prefix is the
	// thing this depends on. If that is ever dropped, the missing-file case loses the word
	// "weights" entirely and the reader cannot tell which flag failed.
	load, err := os.ReadFile(filepath.Join("..", "model", "weights.go"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(load), `"read weights: %w"`) {
		t.Fatal("model.Load no longer says what it was reading, and main.go no longer adds " +
			"it, so a missing -weights file reports a bare os error with no indication of " +
			"which argument was wrong")
	}
}

// TestSidebarSurvivesALongWeightsPath keeps the file a reader loaded from pushing the rest of
// the sidebar out of the window.
//
// The sidebar is 224 DIP wide and its content column has no height bound. `-weights
// /a/very/long/path/...` used to be drawn at full length, wrapping to three lines and, past
// about 800 characters, pushing the status row to y=802 in a 720-tall window — the last rows
// became unreachable with no scrolling available for them. Measured before the fix at
// 1180x720: the weights text grew from 13 DIP tall to 219, and the status line left the window.
//
// Two properties are checked, because either alone is satisfiable by the wrong fix. The
// weights line must stay one line tall (so it cannot grow without bound), and the status row
// must stay inside the window (so the reader can still see the token count and forward-pass
// state, which is what tells them whether the app is working at all).
func TestSidebarSurvivesALongWeightsPath(t *testing.T) {
	// A path longer than any real one needs to be: the point is that the layout does not
	// depend on the length at all.
	long := "/" + strings.Repeat("directory/", 200) + "weights.json"
	for _, size := range []struct{ w, h int }{{1180, 720}, {1500, 940}} {
		a := newTestApp(t)
		a.W.Source = long
		tr := ui.NewTester(a.View, size.w, size.h)
		tr.Frame()

		var weightsRect ui.Rect
		var statusRect ui.Rect
		var sawWeights, sawStatus bool
		for _, s := range tr.Texts() {
			switch {
			case strings.HasPrefix(s, "权重："):
				weightsRect, sawWeights = tr.Find(s)
			case isStatusLine(s):
				statusRect, sawStatus = tr.Find(s)
			}
		}
		if !sawWeights || !sawStatus {
			t.Fatalf("%dx%d: the sidebar lost its weights or status line", size.w, size.h)
		}
		// One line of fontMicro is 13 DIP; two would be 26. Anything taller means the path
		// wrapped, which is the mechanism that used to push everything below it down.
		if weightsRect.H > 20 {
			t.Fatalf("%dx%d: the weights line is %.0f DIP tall, so a long path still wraps",
				size.w, size.h, weightsRect.H)
		}
		if statusRect.Y+statusRect.H > float32(size.h) {
			t.Fatalf("%dx%d: the status line ends at y=%.0f, past the %d-DIP window — a long "+
				"weights path pushed it out and the sidebar cannot scroll",
				size.w, size.h, statusRect.Y+statusRect.H, size.h)
		}
	}
}

// isStatusLine recognises the sidebar's status line in its three shapes: a truncated prompt, a
// normal forward pass, and the empty prompt. Matching on "token ·" alone also caught the shortcut
// hint's "Shift+← → 切换 token · 空格 …", which pushed a different line into the assertion —
// measured: the test reported the status line at y=907 with zero height.
func isStatusLine(s string) bool {
	if strings.HasPrefix(s, "还没有提示词 ·") {
		return true
	}
	if !strings.Contains(s, " 个 token ·") && !strings.Contains(s, "个 token ·") {
		return false
	}
	return strings.Contains(s, "层 ×") && strings.Contains(s, "头")
}
