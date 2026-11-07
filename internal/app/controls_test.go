package app

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"
)

// The layer and head controls are greyed out outside their stages. Greying is a
// visual promise; these check the stronger property behind it — that a control
// changes nothing on the stages where it is disabled, and changes both the stage
// and the inspector on the ones where it is enabled.
//
// Read as a table, the expected pattern is:
//
//	stage   1    2    3    4    5    6    7
//	layer   -    -    ✓    ✓    ✓    ✓    -
//	head    -    -    ✓    ✓    -    -    -
//
// which is layerFrom..layerTo = 2..5 and headFrom..headTo = 2..3, one-based.

// frameFor renders a whole frame with one control set, and returns its text.
func frameFor(t *testing.T, step, layer, head int) string {
	t.Helper()
	a := newTestApp(t)
	a.Step = step
	a.Layer = layer
	a.Head = head
	a.SelToken = 10 // a row with keys before it, so attention has something to show
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	return strings.Join(tr.Texts(), "\n")
}

// TestLayerControlScope walks the layer control across every stage.
func TestLayerControlScope(t *testing.T) {
	for step := 0; step < len(lessons); step++ {
		enabled := step >= layerFrom && step <= layerTo
		a := frameFor(t, step, 0, 0)
		b := frameFor(t, step, 1, 0)
		switch {
		case enabled && a == b:
			t.Fatalf("stage %d: the layer control is enabled here but changing it "+
				"leaves the frame identical", step+1)
		case !enabled && a != b:
			t.Fatalf("stage %d: the layer control is disabled here but changing it "+
				"still alters the frame", step+1)
		}
	}
}

// TestHeadControlScope walks the head control across every stage.
func TestHeadControlScope(t *testing.T) {
	for step := 0; step < len(lessons); step++ {
		enabled := step >= headFrom && step <= headTo
		a := frameFor(t, step, 0, 0)
		b := frameFor(t, step, 0, 1)
		switch {
		case enabled && a == b:
			t.Fatalf("stage %d: the head control is enabled here but changing it "+
				"leaves the frame identical", step+1)
		case !enabled && a != b:
			t.Fatalf("stage %d: the head control is disabled here but changing it "+
				"still alters the frame", step+1)
		}
	}
}

// TestLayerControlReachesTheInspector checks the right panel follows the layer
// too, not just the middle one. A reader switching layers reads both.
func TestLayerControlReachesTheInspector(t *testing.T) {
	inspectorText := func(step, layer int) string {
		a := newTestApp(t)
		a.Step = step
		a.Layer = layer
		a.SelToken = 10
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		var parts []string
		for _, s := range tr.Texts() {
			if r, ok := tr.Find(s); ok && r.X > 1170 {
				parts = append(parts, s)
			}
		}
		return strings.Join(parts, "|")
	}
	for step := layerFrom; step <= layerTo; step++ {
		if inspectorText(step, 0) == inspectorText(step, 1) {
			t.Fatalf("stage %d: the inspector does not follow the layer control", step+1)
		}
	}
}

// TestStageReadsTheSelectedLayersTensor is the content-level check: on the FFN
// stage the plotted vector must be the selected layer's, not layer 0's by
// accident. Comparing frames only proves something changed.
func TestStageReadsTheSelectedLayersTensor(t *testing.T) {
	for _, layer := range []int{0, 1} {
		a := newTestApp(t)
		a.Step = 5
		a.Layer = layer
		l := a.layer()
		lo, hi := minMaxFloat(l.Hidden[a.SelToken])
		want := fmtRange(lo, hi)
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		found := false
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, want) {
				found = true
			}
		}
		if !found {
			t.Fatalf("layer %d: the FFN strip does not show that layer's range %q",
				layer, want)
		}
	}
}

// TestResidualStageReadsTheSelectedLayersInput is the same check for stage 5.
func TestResidualStageReadsTheSelectedLayersInput(t *testing.T) {
	for _, layer := range []int{0, 1} {
		a := newTestApp(t)
		a.Step = 4
		a.Layer = layer
		l := a.layer()
		lo, hi := minMaxFloat(l.In[a.SelToken])
		want := fmtRange(lo, hi)
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		found := false
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, want) {
				found = true
			}
		}
		if !found {
			t.Fatalf("layer %d: the residual stage does not show that layer's input "+
				"range %q", layer, want)
		}
	}
}

// fmtRange is the strip caption's min/max line, which every vector strip prints.
func fmtRange(lo, hi float32) string {
	return fmt.Sprintf("min %+.2f · max %+.2f", lo, hi)
}

// TestTemperatureNoteFollowsItsRange keeps the slider's tooltip derived from the
// range it describes.
//
// The layer and head controls build their tooltips from layerFrom/layerTo and
// headFrom/headTo. The temperature one was a literal — "只有第 7 步的 softmax 除以温度" —
// which happened to match tempFrom = 6 but would not have followed a change to it,
// and nothing would have caught that: the tooltip is not in Texts(), so no assertion
// could see it drift.
func TestTemperatureNoteFollowsItsRange(t *testing.T) {
	// The source must not carry the literal any more.
	src, err := os.ReadFile("app.go")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(src), `"只有第 7 步的 softmax 除以温度"`) {
		t.Fatal("the temperature tooltip is still a hard-coded string; it must be " +
			"built from tempFrom/tempTo like the other two controls")
	}
	if !strings.Contains(string(src), "tempFrom+1") {
		t.Fatal("the temperature tooltip does not derive its stage number from tempFrom")
	}

	// And the range it describes must still be the single sampling stage.
	if tempFrom != tempTo {
		t.Fatalf("the temperature range became %d–%d, so the note's single-stage "+
			"wording is stale", tempFrom, tempTo)
	}
	if tempFrom != 6 {
		t.Fatalf("the temperature stage moved to %d; the sampling stage is 7", tempFrom+1)
	}
}
