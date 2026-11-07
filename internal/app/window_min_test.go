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
	root := filepath.Join("..", "..")

	main, err := os.ReadFile(filepath.Join(root, "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	const minW, minH = 1180, 720
	if !strings.Contains(string(main), "MinWidth:") {
		t.Fatal("main.go no longer sets a minimum window width; the layout is only " +
			"verified down to 1180×720, so the window must not go below it")
	}
	if !strings.Contains(string(main), "MinWidth:  1180") &&
		!strings.Contains(string(main), "MinWidth: 1180") {
		t.Fatalf("the window's minimum width is no longer %d", minW)
	}
	if !strings.Contains(string(main), "MinHeight: 720") {
		t.Fatalf("the window's minimum height is no longer %d", minH)
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
