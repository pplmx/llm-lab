// LLM Lab is a Transformer anatomy viewer built with MyGo's native UI: no
// webview, no HTML, every pixel drawn by Go on the GPU.
package main

import (
	"flag"
	"log"

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

	if *width < 1180 {
		*width = 1180
	}
	if *height < 720 {
		*height = 720
	}

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

	a := app.New(w)
	if *step >= 0 && *step < len(app.LessonNames()) {
		a.Step = *step
	}

	mygo.App.WhenReady(func() {
		mygo.NewWindow(mygo.WindowOptions{
			Title:     "LLM Lab — Transformer 解剖台",
			Width:     *width,
			Height:    *height,
			MinWidth:  1180,
			MinHeight: 720,
			Content:   ui.View(a.View),
		})
	})

	if err := mygo.App.Run(); err != nil {
		log.Fatal(err)
	}
}
