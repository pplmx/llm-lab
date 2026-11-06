package app

import (
	"fmt"
	"image/png"
	"os"
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"
)

// TestRenderSizes draws the busiest stages at the window's minimum and at a
// large size, so overflow can be spotted without launching the app.
func TestRenderSizes(t *testing.T) {
	if os.Getenv("RENDER") == "" {
		t.Skip("set RENDER=1 to write snapshots")
	}
	dir := os.Getenv("RENDER_DIR")
	if dir == "" {
		dir = "/tmp"
	}
	sizes := []struct{ w, h int }{{1180, 720}, {1500, 940}, {1920, 1080}}
	a := newTestApp(t)
	for _, s := range sizes {
		for _, step := range []int{0, 3, 5, 6} {
			a.Step = step
			img := ui.Render(a.View, s.w, s.h, 1)
			path := fmt.Sprintf("%s/size-%dx%d-step%d.png", dir, int(s.w), int(s.h), step)
			f, err := os.Create(path)
			if err != nil {
				t.Fatal(err)
			}
			if err := png.Encode(f, img); err != nil {
				t.Fatal(err)
			}
			if err := f.Close(); err != nil {
				t.Fatal(err)
			}
			t.Logf("wrote %s", path)
		}
	}

	// Long prompt: chips and the heatmap at the 64-token ceiling.
	a.Prompt = strings.Repeat("attention is a weighted sum ", 3)
	a.recompute()
	a.Step = 3
	a.SelToken = len(a.Trace.Tokens) - 1
	for _, w := range []int{1180, 1500} {
		img := ui.Render(a.View, w, 940, 1)
		f, err := os.Create(fmt.Sprintf("%s/size-longprompt-%d.png", dir, w))
		if err != nil {
			t.Fatal(err)
		}
		if err := png.Encode(f, img); err != nil {
			t.Fatal(err)
		}
		if err := f.Close(); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("tokens=%d", len(a.Trace.Tokens))
}
