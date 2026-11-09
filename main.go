// LLM Lab is a Transformer anatomy viewer built with MyGo's native UI: no
// webview, no HTML, every pixel drawn by Go on the GPU.
package main

import (
	"flag"
	"log"
	"strings"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"

	"llmlab/internal/app"
	"llmlab/internal/model"
)

func main() {
	weightsPath := flag.String("weights", "", "path to a weights.json file (default: the bundled toy model)")
	width := flag.Int("w", 1500, "window width")
	height := flag.Int("h", 940, "window height")
	step := flag.Int("step", 0, "lesson to open on (0-6)")
	flag.Parse()

	*width, *height = app.ClampWindowSize(*width, *height)

	var (
		w   *model.Weights
		err error
	)
	if *weightsPath != "" {
		w, err = model.Load(*weightsPath)
		if err != nil {
			log.Fatalf("weights %s: %v", *weightsPath, err)
		}
	} else {
		w, err = model.Bundled()
		if err != nil {
			log.Printf("bundled weights unavailable (%v); falling back to random weights", err)
			w = model.Random(model.DefaultConfig, 42)
		}
	}

	// -step is validated rather than clamped. It used to be applied only when in range, so
	// `-step 99` opened the first lesson with no indication that the requested one did not
	// exist — the reader asks for the output stage and gets the tokenising stage, which reads
	// as the flag being ignored rather than as a mistake in its value. A bad range is a
	// command-line error, like a weights file that will not parse, so it stops the program
	// and says what the valid values are.
	names := app.LessonNames()
	if *step < 0 || *step >= len(names) {
		log.Fatalf("-step %d is out of range: lessons are numbered 0-%d (%s)",
			*step, len(names)-1, strings.Join(names, ", "))
	}

	a := app.New(w)
	a.Step = *step

	mygo.App.WhenReady(func() {
		mygo.NewWindow(mygo.WindowOptions{
			Title:     "LLM Lab — Transformer 解剖台",
			Width:     *width,
			Height:    *height,
			MinWidth:  app.MinWindowWidth,
			MinHeight: app.MinWindowHeight,
			Content:   ui.View(a.View),
		})
	})

	if err := mygo.App.Run(); err != nil {
		log.Fatal(err)
	}
}
