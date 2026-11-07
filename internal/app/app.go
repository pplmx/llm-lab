package app

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/egoist/mygo/ui"

	"llmlab/internal/model"
)

type modelToken = model.Token

const defaultPrompt = "The transformer reads the tokens and"

// promptPreset is one example prompt offered as a chip.
type promptPreset struct {
	name   string
	prompt string
}

// promptPresets are short examples that exercise the tokenizer's edges:
// ASCII, CJK (three bytes per rune), a prompt past the context window, and
// the empty state.
var promptPresets = []promptPreset{
	{"英文", defaultPrompt},
	{"中文", "注意力就是加权求和"},
	{"长提示词", strings.Repeat("attention ", 12)},
	{"空", ""},
}

// App holds everything the view reads: the weights, the last trace, and the
// user's place in the lesson.
type App struct {
	W      *model.Weights
	Trace  *model.Trace
	Prompt string

	Step  int
	Layer int
	Head  int
	Temp  float64

	SelToken int

	// HoverQ/HoverK are the cell under the pointer. The heatmap rewrites them
	// every frame it is drawn — to -1 when the pointer is elsewhere — so they
	// cannot also carry a keyboard selection, which has to survive the pointer
	// moving at all.
	HoverQ, HoverK int
	// KeyQ/KeyK are the cell chosen with the keyboard, -1 when none. The
	// heatmap draws them like the hovered cell but in the selection colour, so
	// a reader who never touches the mouse can still walk the matrix; the
	// inspector reads whichever pair is live.
	KeyQ, KeyK int

	// HoverDim* carry the dimension under the pointer for each strip, -1 when
	// none. They are separate because the strips are not all the same width:
	// the FFN stage draws the hidden row (d_ff wide) and the output row
	// (d_model wide) in one card, so one index cannot name a dimension in both.
	HoverDimIn           int
	HoverDimQ, HoverDimK int
	HoverDimV, HoverDimH int
	// HoverDimOut is the FFN stage's output row, kept apart from HoverDimH so
	// pointing at d200 of the hidden row does not silently retarget the output
	// formula to out[0].
	HoverDimOut int
	// HoverDimHeadOut and HoverDimProj are the attention stage's head output
	// (d_head wide) and the two projections (d_model wide). They share a card
	// the way the FFN card does, so they get the same treatment: one index
	// cannot be a dimension in a 24-wide row and a 96-wide row at once.
	HoverDimHeadOut, HoverDimProj int

	SidebarW   float32
	InspectorW float32

	Scroll ui.ScrollState
	// RailScroll keeps the prompt strip's horizontal offset across stages.
	RailScroll ui.ScrollState
	// RevealRail asks the next frame to scroll the selected chip into view,
	// once. It is set when the selection moves without the pointer, so that
	// the strip never steals the scroll back from a reader browsing it.
	RevealRail bool

	// heatmapAxesForTest lets a test render the matrices with the edge labels
	// forced on or off, so the difference between the two can be measured. Nil
	// means "use the call site's choice".
	heatmapAxesForTest *bool

	LastSample string
	Status     string

	// cancelStats is the per-trace cancellation summary the attention stage
	// quotes. It depends only on the trace, so it is computed when the trace is
	// rebuilt rather than on every frame: the walk is 29us, which is small
	// against a 12.6ms frame but was pure repetition at 60fps.
	cancelStats cancellationStats
	// cancelStatsFor records which trace cancelStats describes, so a stale
	// summary can never be quoted after recompute forgets to refresh it.
	cancelStatsFor *model.Trace

	Playing bool
	NextAt  time.Time

	theme *ui.Theme

	lastPrompt string
}

// New builds the app with the bundled weights.
func New(w *model.Weights) *App {
	a := &App{
		W:               w,
		Prompt:          defaultPrompt,
		Step:            0,
		Temp:            0.8,
		SelToken:        -1,
		HoverQ:          -1,
		HoverK:          -1,
		KeyQ:            -1,
		KeyK:            -1,
		HoverDimIn:      -1,
		HoverDimQ:       -1,
		HoverDimK:       -1,
		HoverDimV:       -1,
		HoverDimH:       -1,
		HoverDimOut:     -1,
		HoverDimHeadOut: -1,
		HoverDimProj:    -1,
		SidebarW:        224,
		InspectorW:      330,
		theme:           labTheme(),
	}
	a.recompute()
	// The inspector explains the selected token, so the strip has to show it
	// from the first frame. ScrollIntoView moves as little as it can, so a
	// prompt that fits is not scrolled at all.
	a.RevealRail = true
	return a
}

// applyPrompt replaces the prompt and re-runs the forward pass, dropping the
// sample and autoplay state that belonged to the previous prompt.
func (a *App) applyPrompt(p string) {
	a.Prompt = p
	a.promptChanged()
}

// promptChanged drops the state that belonged to the previous prompt and re-runs the
// forward pass. Both ways of changing the prompt go through it — the preset chips and
// the text field — because they have to leave the app in the same state: a preset click
// and an equivalent edit must not differ in whether playback stops or a stale sample
// survives.
//
// The field's own handler calls this after ui.TextInput has already written a.Prompt,
// which is why the assignment stays in applyPrompt rather than moving in here.
func (a *App) promptChanged() {
	a.LastSample = ""
	a.Playing = false
	a.NextAt = time.Time{}
	a.recompute()
}

func (a *App) recompute() {
	prevPrompt := a.lastPrompt
	prevSel := a.SelToken
	prevTraceLen := 0
	if a.Trace != nil {
		prevTraceLen = len(a.Trace.Tokens)
	}
	ids := model.Truncate(promptIDs(a.Prompt), a.W.Config.Ctx)

	a.Trace = model.Forward(a.W, ids)
	if n := len(a.Trace.Tokens); n > 0 {
		switch {
		case prevSel < 0:
			a.SelToken = n - 1
		case strings.HasPrefix(a.Prompt, prevPrompt) && prevTraceLen > 0:
			// The prompt was extended, so the old token's absolute position
			// is still meaningful. Shift it by the bytes truncation dropped
			// on both sides of the edit.
			droppedOld := len(prevPrompt) - prevTraceLen
			droppedNew := len(a.Prompt) - len(ids)
			shifted := droppedOld + prevSel - droppedNew
			if shifted >= 0 && shifted < n {
				a.SelToken = shifted
			} else {
				a.SelToken = n - 1
			}
		case prevSel >= n:
			a.SelToken = n - 1
		case a.Prompt != prevPrompt:
			// A replacement (not an extension) has no meaningful old token
			// at the same index; fall back to the new last token.
			a.SelToken = n - 1
		default:
			a.SelToken = prevSel
		}
	} else {
		a.SelToken = -1
	}
	a.lastPrompt = a.Prompt
	// The pointer's hover belongs to the old layout; the pinned cell does not.
	a.clearPointer()
	a.clampPinnedCell()
	// Refresh the per-trace summary alongside the trace it describes.
	a.cancelStats = measureCancellation(a.Trace)
	a.cancelStatsFor = a.Trace
	a.Status = fmt.Sprintf("%d tokens · %d 层 × %d 头 · 前向完成", len(ids), a.W.Config.NLayer, a.W.Config.NHead)
	if len(a.Prompt) > a.W.Config.Ctx {
		a.Status = fmt.Sprintf("已截断到最近 %d 个 token · %d 层 × %d 头", len(ids), a.W.Config.NLayer, a.W.Config.NHead)
	}
}

func promptIDs(s string) []int {
	ids := make([]int, 0, len(s))
	for i := 0; i < len(s); i++ {
		ids = append(ids, int(s[i]))
	}
	return ids
}

// cellCursor is the attention cell the reader is looking at, keyboard first:
// a cell walked to with the arrow keys stays put while the pointer moves, and
// while it is set it is the one the inspector and the strip describe.
func (a *App) cellCursor() (q, k int) {
	if a.KeyQ >= 0 && a.KeyK >= 0 {
		return a.KeyQ, a.KeyK
	}
	return a.HoverQ, a.HoverK
}

// hasMatrix reports whether the current stage draws a heatmap, and so has cells
// for the keyboard to walk: step 3 (注意力分数) draws the score matrix and step
// 4 (Softmax) draws the attention matrix. The other five stages have no matrix,
// so the arrow keys there go on moving the token selection.
func (a *App) hasMatrix() bool { return a.Step == 2 || a.Step == 3 }

// stepCell moves the keyboard's attention cell by (dq, dk). Only the stages
// that draw a heatmap have cells; elsewhere the keys do nothing, which is why
// the caller gates on the same range the strip's q/k marks use.
//
// The cell never leaves the causal triangle: the first press enters from
// wherever the pointer last hovered, or from the selected query's own row, and
// the key stays at or before the query because everything past it is masked.
func (a *App) stepCell(dq, dk int) {
	if a.Step < 2 || a.Step > 3 {
		return
	}
	tr := a.Trace
	n := len(tr.Tokens)
	if n == 0 {
		return
	}
	q, k := a.KeyQ, a.KeyK
	if q < 0 || k < 0 || q >= n || k >= n {
		// Entering the matrix: start from the pointer if it is on a cell,
		// else from the selected query at its first key.
		if a.HoverQ >= 0 && a.HoverK >= 0 && a.HoverQ < n && a.HoverK < n {
			q, k = a.HoverQ, a.HoverK
		} else {
			q = clampIndex(a.SelToken, n)
			k = 0
		}
	}
	q += dq
	if dq != 0 {
		// Moving to another query starts at that row's own diagonal: the row's
		// most recent key, which is where the interesting weight usually is.
		if q < 0 {
			q = 0
		}
		if q >= n {
			q = n - 1
		}
		k = q
	}
	k += dk
	if k < 0 {
		k = 0
	}
	if k > q {
		k = q
	}
	a.KeyQ, a.KeyK = q, k
	// Selecting a cell is also selecting its query: the inspector and the
	// strip should describe the row the reader is walking.
	a.selectToken(q, true)
}

// selectToken moves the selection. reveal asks the next frame to scroll the
// token strip to it, for selections made from outside the strip (the keyboard,
// a heatmap cell, a new prompt): the reader's eye is elsewhere, and the strip
// has to follow.
func (a *App) selectToken(i int, reveal bool) {
	a.SelToken = i
	if reveal {
		a.RevealRail = true
	}
	// A pinned cell belongs to the token the reader was looking at when they
	// clicked it. Selecting a different token moves them somewhere else, and the
	// inspector would then head "query = token 20" while the cell underneath it
	// still described token 10's row. The stage switch already drops the pin for
	// the same reason; this is the other way the context changes.
	if a.KeyQ >= 0 && a.KeyK >= 0 && a.KeyQ != i {
		a.KeyQ, a.KeyK = -1, -1
	}
}

// stepToken moves the selection by delta, clamped to the sequence, for the
// Shift+Left/Right shortcuts.
func (a *App) stepToken(delta int) {
	n := len(a.Trace.Tokens)
	if n == 0 {
		return
	}
	i := a.SelToken
	if i < 0 {
		i = 0
	}
	a.selectToken(clampIndex(i+delta, n), true)
}

func (a *App) clearHover() {
	a.HoverQ, a.HoverK = -1, -1
	a.KeyQ, a.KeyK = -1, -1
	a.HoverDimIn = -1
	a.HoverDimQ, a.HoverDimK = -1, -1
	a.HoverDimV, a.HoverDimH = -1, -1
	a.HoverDimOut = -1
	a.HoverDimHeadOut, a.HoverDimProj = -1, -1
}

// clampPinnedCell keeps a pinned attention cell inside the triangle after the
// token count changes. Extending the prompt keeps every existing cell valid, so
// the pin normally survives untouched; truncation can shrink the matrix under
// it, and a pin outside the new bounds would draw on a cell that no longer
// exists.
func (a *App) clampPinnedCell() {
	n := 0
	if a.Trace != nil {
		n = len(a.Trace.Tokens)
	}
	if a.KeyQ < 0 && a.KeyK < 0 {
		return
	}
	if n == 0 {
		a.KeyQ, a.KeyK = -1, -1
		return
	}
	if a.KeyQ >= n {
		a.KeyQ = n - 1
	}
	if a.KeyK > a.KeyQ {
		a.KeyK = a.KeyQ
	}
	if a.KeyK < 0 {
		a.KeyK = 0
	}
}

// clearPointer is clearHover without the keyboard cursor. Editing the prompt
// rebuilds the trace but does not move anything the reader chose: the pointer
// hover is stale by definition (the page reflowed under it), while the pinned
// cell is a deliberate selection that should survive. Sampling used to drop the
// pin silently — the reader set it, pressed 采样并续写, and the card under it
// changed from "已固定" back to nothing.
// clearPointer drops every hover state but leaves a pinned cell alone: the pointer
// leaving the matrix should not unpin what a click fixed.
func (a *App) clearPointer() {
	a.HoverQ, a.HoverK = -1, -1
	a.HoverDimIn = -1
	a.HoverDimQ, a.HoverDimK = -1, -1
	a.HoverDimV, a.HoverDimH = -1, -1
	a.HoverDimOut = -1
	a.HoverDimHeadOut, a.HoverDimProj = -1, -1
}

// cancellation returns the cached summary for the current trace, computing it
// if the cache is missing or describes a different trace. The fallback keeps a
// caller that renders before recompute from quoting zeroes.
func (a *App) cancellation() cancellationStats {
	if a.Trace == nil {
		return cancellationStats{}
	}
	if a.cancelStatsFor != a.Trace {
		a.cancelStats = measureCancellation(a.Trace)
		a.cancelStatsFor = a.Trace
	}
	return a.cancelStats
}

// stageFade is how a stage switch animates: both panels that describe the stage
// fade in over 160 ms.
//
// The motion is opacity only and never position or size. The lab exists to
// compare one stage's numbers against the next, so content that slid into place
// would make that comparison harder, not easier.
//
// Enter starts at 0.35 rather than 0 on purpose. At 0 mygo draws one frame of
// the fade with the content fully transparent, which measured over the stage
// column is plain background — a visible flash of emptiness on every switch.
// Starting part-opaque keeps every frame of the transition readable.
var stageFade = ui.ElementTransition{
	Duration: 160 * time.Millisecond,
	Ease:     ui.EaseOut,
	Enter:    &ui.Motion{Opacity: 0.35},
}

// stageTransition is the fade to use for this frame, or the zero value when the
// reader has asked for less motion.
//
// mygo honours ReduceMotion for moves and size changes but still plays a
// fade-only Enter: transition.go gates it on `!still || m.fades()`, so a
// fade-only Enter runs under the preference. That is the right default for the
// fades mygo applies to dialogs and menus, which are short and convey arrival;
// here it would leave a reader who turned animation off watching eight frames
// of fade on every stage switch, so this checks the preference itself and asks
// for no transition at all.
func (a *App) stageTransition(c *ui.Context) ui.ElementTransition {
	if c.Preferences().ReduceMotion {
		return ui.ElementTransition{}
	}
	return stageFade
}

// selectStep changes the lesson and returns the main scroll area to the top.
// It is the manual counterpart to play's automatic advance.
func (a *App) selectStep(step int) {
	if step < 0 || step >= len(lessons) {
		return
	}
	a.Step = step
	a.Scroll.X, a.Scroll.Y = 0, 0
	a.Playing = false
	a.NextAt = time.Time{}
	a.clearHover()
}

// playStepInterval is how long each stage stays up while the lesson plays itself.
// One constant for both places it is used: the first wait after starting and every
// wait after that. Two literals would let the first step take a different time from
// the rest, which reads as a stutter rather than a deliberate pace.
const playStepInterval = 1300 * time.Millisecond

// play advances the lesson every playStepInterval while the user watches it.
func (a *App) play(c *ui.Context) {
	if !a.Playing {
		return
	}
	now := c.Now()
	if a.NextAt.IsZero() {
		a.NextAt = now.Add(playStepInterval)
	}
	if !now.Before(a.NextAt) {
		if a.Step >= len(lessons)-1 {
			a.Playing = false
			return
		}
		a.Step++
		a.Scroll.X, a.Scroll.Y = 0, 0
		a.clearHover()
		a.NextAt = now.Add(playStepInterval)
	}
	c.After(a.NextAt.Sub(now))
}

func (a *App) layer() *model.LayerTrace {
	if len(a.Trace.Layers) == 0 {
		return nil
	}
	l := a.Layer
	if l < 0 {
		l = 0
	}
	if l >= len(a.Trace.Layers) {
		l = len(a.Trace.Layers) - 1
	}
	return &a.Trace.Layers[l]
}

func (a *App) View(c *ui.Context) {
	a.handleKeys(c)
	a.play(c)
	ui.Themed(c, a.theme).Fill().Children(func() {
		ui.Row(c).Fill().Background(colBg).AlignItems(ui.Stretch).Children(func() {
			a.sidebar(c)
			ui.Box(c).Width(1).FillHeight().Background(colBorder)
			// FillHeight bounds the column to the window. With only Grow(1) the
			// column sized itself to its content and the overflow was centred,
			// which cut the top bar off the top of a 720px window.
			ui.Column(c).Grow(1).FillHeight().Children(func() {
				a.topbar(c)
				a.tokenRail(c)
				a.stepbar(c)
				// MinHeight(0) lets the scroll view give up height; without it
				// the fixed rows above are squeezed to nothing at 1180×720.
				ui.Scroll(c).Grow(1).MinHeight(0).Padding(16, 16, 24).Gap(12).TrackScroll(&a.Scroll).Children(func() {
					// One key per stage, so switching rebuilds this subtree as
					// new content and it fades in rather than swapping
					// instantly. The motion is opacity only, never position:
					// the whole point of the lab is comparing two stages' data,
					// and content that slides would make that harder to follow.
					ui.Column(c).Key(a.Step).Gap(12).FillWidth().
						Transition(a.stageTransition(c)).Children(func() {
						a.stageHeader(c)
						a.stage(c)
						a.lesson(c)
					})
				})
			})
			ui.Box(c).Width(1).FillHeight().Background(colBorder)
			a.inspector(c)
		})
	})
}

// handleKeys wires the shortcuts that work anywhere in the window. A focused
// text field or button consumes its own keys first, so typing and Space still
// reach them instead of the lesson.
func (a *App) handleKeys(c *ui.Context) {
	switch {
	case c.Shortcut(0, ui.KeyRight):
		a.selectStep(a.Step + 1)
	case c.Shortcut(0, ui.KeyLeft):
		a.selectStep(a.Step - 1)
	case c.Shortcut(0, ui.KeySpace):
		a.togglePlay()
	case c.Shortcut(0, ui.KeyEscape):
		// Esc is the one "let go of everything" key: it stops playback and
		// releases a pinned cell, so a reader who clicked the matrix can get
		// back to plain hovering without hunting for the cell again.
		a.Playing = false
		a.NextAt = time.Time{}
		a.KeyQ, a.KeyK = -1, -1
	// Shift+arrows. On the two attention stages they walk the matrix, since a
	// matrix cell is what those stages teach and was the last interaction only
	// the mouse could reach; everywhere else they move the token selection, as
	// they always have.
	//
	// The two behaviours share one Shortcut call each rather than appearing as
	// `Shortcut(...) && hasMatrix()` followed by a bare Shortcut. A second call
	// for the same key in one frame reports false — the first ask used the
	// press up — so the guarded form silently ate every Shift+arrow outside
	// the attention stages.
	case c.Shortcut(ui.Shift, ui.KeyLeft):
		if a.hasMatrix() {
			a.stepCell(0, -1)
		} else {
			a.stepToken(-1)
		}
	case c.Shortcut(ui.Shift, ui.KeyRight):
		if a.hasMatrix() {
			a.stepCell(0, 1)
		} else {
			a.stepToken(1)
		}
	case c.Shortcut(ui.Shift, ui.KeyUp):
		if a.hasMatrix() {
			a.stepCell(-1, 0)
		}
	case c.Shortcut(ui.Shift, ui.KeyDown):
		if a.hasMatrix() {
			a.stepCell(1, 0)
		}
	default:
		digits := [...]ui.Key{ui.Key1, ui.Key2, ui.Key3, ui.Key4, ui.Key5, ui.Key6, ui.Key7}
		for i, key := range digits {
			if i < len(lessons) && c.Shortcut(0, key) {
				a.selectStep(i)
				break
			}
		}
	}
}

func (a *App) togglePlay() {
	a.Playing = !a.Playing
	a.NextAt = time.Time{}
	if a.Playing && a.Step == len(lessons)-1 {
		a.Step = 0
		a.Scroll.X, a.Scroll.Y = 0, 0
		a.clearHover()
	}
}

// stageHeader states where the reader is before the figures: the step, what
// it teaches, and the formula it derives.
func (a *App) stageHeader(c *ui.Context) {
	l := lessons[a.Step]
	ui.Row(c).FillWidth().AlignItems(ui.Center).Gap(10).Children(func() {
		ui.Text(c, fmt.Sprintf("第 %d / %d 步", a.Step+1, len(lessons))).
			FontSize(fontMicro).Bold().TextColor(colBg).
			Padding(4, 8).Radius(5).Background(colAccent).Shrink(0)
		ui.Text(c, l.title).FontSize(fontTitle).Bold().TextColor(colText)
		ui.Spacer(c).Grow(1)
		ui.Text(c, l.formula).FontSize(fontSmall).TextColor(colAccent2).
			Padding(4, 8).Radius(5).Background(colPanel2).Shrink(0)
	})
}

// ---------------------------------------------------------------- sidebar

func (a *App) sidebar(c *ui.Context) {
	ui.Column(c).Width(a.SidebarW).FillHeight().Padding(16).Gap(14).Children(func() {
		ui.Row(c).AlignItems(ui.Center).Gap(8).Children(func() {
			ui.Box(c).Size(26, 26).Radius(7).Background(colAccent).Children(func() {
				ui.Text(c, "λ").FontSize(fontHero).Bold().TextColor(colBg)
			})
			ui.Column(c).Gap(0).Children(func() {
				ui.Text(c, "LLM Lab").FontSize(fontTitle).Bold().TextColor(colText)
				ui.Text(c, "Transformer 解剖台").FontSize(fontMicro).TextColor(colFaint)
			})
		})

		ui.Box(c).FillWidth().Height(1).Background(colBorder)

		a.stepper(c)

		ui.Spacer(c).Grow(1)

		ui.Column(c).FillWidth().Padding(10).Gap(6).Background(colPanel).Radius(8).Children(func() {
			ui.Text(c, "模型").FontSize(fontMicro).TextColor(colFaint)
			kv(c, "vocab", fmt.Sprint(a.W.Config.Vocab))
			kv(c, "d_model", fmt.Sprint(a.W.Config.DModel))
			kv(c, "layers", fmt.Sprint(a.W.Config.NLayer))
			kv(c, "heads", fmt.Sprint(a.W.Config.NHead))
			kv(c, "ctx", fmt.Sprint(a.W.Config.Ctx))
			kv(c, "参数", humanCount(a.paramCount()))
		})
		ui.Text(c, "权重："+a.W.Source).FontSize(fontMicro).TextColor(colFaint)
		if a.W.Steps > 0 {
			ui.Text(c, fmt.Sprintf("训练 %d 步 · loss %.3f", a.W.Steps, a.W.Loss)).FontSize(fontMicro).TextColor(colFaint)
		} else {
			// The one line that stops a reader from reading meaning into the
			// logits: these weights were never trained.
			ui.Text(c, "未训练 · 数值真实但输出无意义").FontSize(fontMicro).TextColor(colWarm).LineHeight(1.35)
		}
		// The status line used to sit in the top bar, where it pushed the
		// temperature slider out of the window at 1180 wide.
		ui.Text(c, a.Status).FontSize(fontMicro).TextColor(colFaint).LineHeight(1.35)
	})
}

// untrained reports whether the app is running on weights that never saw a
// training step. main.go falls back to model.Random when the bundled file
// cannot be read, and random weights still produce perfectly valid numbers —
// so every visualisation in the lesson stays plausible while the outputs are
// gibberish. Nothing else in the UI would say so.
func (a *App) untrained() bool {
	return a.W != nil && a.W.Steps == 0
}

// untrainedWhy distinguishes the two ways to reach an untrained model: the
// bundled file failed to load and main fell back to random weights, or the
// reader pointed -weights at a file that carries no training metadata. They
// need different advice, so the banner does not claim the bundled file was
// missing when a custom one was given.
func (a *App) untrainedWhy() string {
	if strings.HasPrefix(a.W.Source, "random(") {
		return "读不到内置权重，正在跑一个随机初始化的模型。"
	}
	return "这份权重没有训练信息（steps = 0），数值来自未训练的模型。"
}

func (a *App) paramCount() int {
	c := a.W.Config
	d := c.DModel
	n := c.Vocab*d + c.Ctx*d + 2*d
	per := 4*d*d + 2*d*c.DFF + 4*d
	return n + c.NLayer*per
}

func humanCount(n int) string {
	if n >= 1000000 {
		return fmt.Sprintf("%.2fM", float64(n)/1e6)
	}
	if n >= 1000 {
		return fmt.Sprintf("%.1fK", float64(n)/1e3)
	}
	return fmt.Sprint(n)
}

// ----------------------------------------------------------------- top bar

func (a *App) topbar(c *ui.Context) {
	ui.Column(c).FillWidth().Padding(16, 14, 10).Gap(10).Background(colPanel).
		BorderColor(colBorder).Shrink(0).Children(func() {
		if a.untrained() {
			a.untrainedBanner(c)
		}
		ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
			field := ui.TextInput(c, &a.Prompt).Grow(1).Label("输入").SingleLine()
			if field.Changed() {
				a.promptChanged()
			}
			if ui.Button(c, "重置").Clicked() {
				a.applyPrompt(defaultPrompt)
			}
			if ui.PrimaryButton(c, "采样下一个 token").Clicked() {
				a.sampleNext()
			}
		})

		// Each group is dimmed and disabled outside the stages that read it:
		// the layer and head select nothing while tokenising, and the
		// temperature only enters at the output's softmax. A live control that
		// changes no pixel tells the reader the pipeline depends on it
		// everywhere, which is the opposite of what the stages show.
		ui.Row(c).Gap(12).Wrap().AlignItems(ui.Center).Children(func() {
			a.layerControl(c)
			a.headControl(c)
			a.tempControl(c)
		})
	})
}

// untrainedBanner warns that the numbers on screen come from random weights.
// It sits in the top bar, above every stage, because the failure is invisible
// in the lesson: a random transformer still produces a causal triangle, rows
// that sum to 1, and a top-12 distribution. Only the sampled text gives it
// away, and by then the reader has already trusted the rest.
func (a *App) untrainedBanner(c *ui.Context) {
	ui.Row(c).FillWidth().Padding(8, 10).Radius(7).Gap(8).AlignItems(ui.Center).
		Background(colWarm.Alpha(0.12)).Border(1, colWarm.Alpha(0.45)).
		Children(func() {
			ui.Text(c, "⚠").FontSize(fontData).TextColor(colWarm).Shrink(0)
			ui.Text(c, "权重未训练").
				FontSize(fontSmall).Bold().TextColor(colWarm).Shrink(0)
			ui.Text(c, a.untrainedWhy()+
				"形状、掩码、每行和为 1 都是真算出来的，但权重是随机数——别从输出里读出意义。").
				FontSize(fontMicro).TextColor(colText).LineHeight(1.35)
		})
}

// ------------------------------------------------------- stage-scoped controls

// Which stages read which control. The layer and the head select nothing before
// the attention stages and nothing after them; the temperature enters only at
// the output, where softmax divides the logits by it. Dimming the controls that
// do nothing is part of the lesson: it shows the reader where in the pipeline a
// knob starts to matter.
const (
	layerFrom, layerTo = 2, 5 // 注意力分数 … FFN reads a layer
	headFrom, headTo   = 2, 3 // 注意力分数 / Softmax are the stages with heads
	tempFrom, tempTo   = 6, 6 // 输出 / 采样
)

func inStages(step, from, to int) bool { return step >= from && step <= to }

// dimmed is the opacity of a control outside the current stage. MyGo's
// Disabled removes the interaction but paints the widget as usual, so the
// dimming is the only visible cue; 0.55 keeps the widget's shape readable while
// taking the labels to about 2.5:1, which reads as "off" without hiding that
// the control is there.
const dimmed = 0.55

// controlGroup lays out one knob of the top bar: a label, its widget and the
// rule that scopes them to a range of stages. A control outside its stages is
// dimmed and disabled; MyGo's Disabled takes the pointer and the keyboard away
// but paints the widget as usual, so Opacity is what actually says "off". The
// tip goes on the wrapper because a disabled element hides its own tip, and the
// wrapper is what stays under the pointer.
func (a *App) controlGroup(c *ui.Context, label string, from, to int, tip string, build func(disabled bool)) {
	enabled := inStages(a.Step, from, to)
	group := ui.Row(c).Gap(6).AlignItems(ui.Center).Shrink(0).Tooltip(tip)
	if !enabled {
		group = group.Opacity(dimmed)
	}
	group.Children(func() {
		ui.Text(c, label).FontSize(fontSmall).TextColor(colFaint).Shrink(0)
		build(!enabled)
	})
}

func (a *App) layerControl(c *ui.Context) {
	labels := make([]string, a.W.Config.NLayer)
	for i := range labels {
		labels[i] = fmt.Sprintf("%d", i)
	}
	build := func(disabled bool) {
		cur := a.Layer
		if ui.Segmented(c, &cur, labels...).Disabled(disabled).Changed() {
			a.Layer = cur
			a.clearHover()
		}
	}
	a.controlGroup(c, "层", layerFrom, layerTo,
		fmt.Sprintf("第 %d–%d 步读这一层的权重", layerFrom+1, layerTo+1), build)
}

func (a *App) headControl(c *ui.Context) {
	heads := make([]string, a.W.Config.NHead)
	for i := range heads {
		heads[i] = fmt.Sprintf("H%d", i)
	}
	build := func(disabled bool) {
		h := a.Head
		if ui.Segmented(c, &h, heads...).Disabled(disabled).Changed() {
			a.Head = h
			a.clearHover()
		}
	}
	a.controlGroup(c, "头", headFrom, headTo,
		fmt.Sprintf("第 %d–%d 步按头拆开注意力", headFrom+1, headTo+1), build)
}

func (a *App) tempControl(c *ui.Context) {
	build := func(disabled bool) {
		if ui.Slider(c, &a.Temp, 0.1, 2.0).Width(130).Shrink(0).Disabled(disabled).Changed() {
			a.LastSample = ""
		}
	}
	// The other two controls build this sentence from their own range constants;
	// this one was hard-coded, so changing tempFrom would have left the tooltip
	// pointing at a stage the slider no longer applies to.
	note := fmt.Sprintf("只有第 %d 步的 softmax 除以温度", tempFrom+1)
	if tempFrom != tempTo {
		note = fmt.Sprintf("只有第 %d–%d 步的 softmax 除以温度", tempFrom+1, tempTo+1)
	}
	a.controlGroup(c, fmt.Sprintf("温度 %.2f", a.Temp), tempFrom, tempTo, note, build)
}

// ---------------------------------------------------------------- step bar

func (a *App) stepbar(c *ui.Context) {
	ui.Row(c).FillWidth().Padding(16, 8, 12).Gap(10).AlignItems(ui.Center).Shrink(0).Children(func() {
		// A progress trail instead of a second list of lesson names: the
		// sidebar already names every stage, and at 1180×720 a wrapped chip
		// row pushed the lesson below the fold.
		ui.Row(c).Gap(4).AlignItems(ui.Center).Children(func() {
			for i := range lessons {
				i := i
				var dot ui.Element
				switch {
				case i == a.Step:
					dot = ui.Box(c).Size(18, 6).Radius(3).Background(colAccent)
				case i < a.Step:
					dot = ui.Box(c).Size(6, 6).Radius(3).Background(colMuted)
				default:
					dot = ui.Box(c).Size(6, 6).Radius(3).Background(colBorder)
				}
				// These are 6-DIP dots that jump to a stage, and they gave no sign
				// of being reachable: no hover change, no tooltip, nothing to
				// separate them from the label beside them.
				//
				// Only the colour changes, never the width. Growing the dot to the
				// current dot's 18 DIPs did move the row, so it was reverted.
				// Hovered() has to be asked on the element that is already laid
				// out, before painting — asking it after building a replacement
				// shifts every later dot, because the replacement is a new element
				// in the flex row.
				hovered := i != a.Step && dot.Hovered()
				if hovered {
					dot = dot.Background(colText.Alpha(0.8))
				}
				if dot.Clicked() {
					a.selectStep(i)
				}
				// Same text as the sidebar's row tooltip. The two controls
				// reach the same place, so they should not describe it differently:
				// the stage's name is already visible in the bar beside the dots,
				// and the title says what the stage teaches.
				dot.Tooltip(fmt.Sprintf("第 %d / %d 步 · %s", i+1, len(lessons), lessons[i].title))
			}
		})
		ui.Text(c, fmt.Sprintf("%d / %d · %s", a.Step+1, len(lessons), lessons[a.Step].name)).
			FontSize(fontSmall).TextColor(colMuted).NoWrap()
		// The full shortcut list sits at the foot of the lesson card, which on the
		// two matrix stages is 300-450 DIPs below the fold at every window size.
		// Those stages are exactly where Shift+arrows change meaning, so the one
		// place the reader cannot see the help is the place it matters. Say it in
		// the step bar instead, where it is always on screen.
		if a.hasMatrix() {
			ui.Text(c, "⇧ + 方向键 移动格子 · 点击固定").
				FontSize(fontMicro).TextColor(colFaint).NoWrap()
		}
		ui.Spacer(c).Grow(1)
		ui.Row(c).Gap(6).AlignItems(ui.Center).Shrink(0).Children(func() {
			label := "▶ 自动播放"
			if a.Playing {
				label = "⏸ 暂停"
			}
			if ui.Button(c, label).Shrink(0).Clicked() {
				a.togglePlay()
			}
			// The step buttons dim at the ends of the lesson, like the stage
			// controls: a button that cannot move the reader should not look
			// like it can, and the dimming also says "this is the first step".
			atStart, atEnd := a.Step == 0, a.Step == len(lessons)-1
			prev := ui.Button(c, "← 上一步").Shrink(0).Disabled(atStart)
			if atStart {
				prev = prev.Opacity(dimmed)
			}
			if prev.Clicked() {
				a.selectStep(a.Step - 1)
			}
			next := ui.Button(c, "下一步 →").Shrink(0).Disabled(atEnd)
			if atEnd {
				next = next.Opacity(dimmed)
			}
			if next.Clicked() {
				a.selectStep(a.Step + 1)
			}
		})
	})
}

// ---------------------------------------------------------------- inspector

// inspector follows the lesson: each step shows the tensors that step is
// about, so the right panel never explains attention while the left panel is
// still tokenizing.
func (a *App) inspector(c *ui.Context) {
	ui.Scroll(c).Width(a.InspectorW).FillHeight().Padding(14).Gap(12).Children(func() {
		ui.Row(c).AlignItems(ui.Center).Gap(8).Children(func() {
			ui.Text(c, "检查器").FontSize(fontBody).Bold().TextColor(colMuted)
			ui.Text(c, lessons[a.Step].name).FontSize(fontMicro).TextColor(colFaint)
			// The stage area now names the layer in its card headings; the
			// inspector reads the same tensors and has the same problem, more so
			// because the top bar's controls are what select them and the panel
			// is on the far side of the window. State the scope once here rather
			// than repeating it in every card below.
			//
			// Only when there is something to scope: with no prompt every tensor
			// is empty, and "层 0 · 头 0" beside an empty-state paragraph reads as
			// though those two controls still select something.
			if len(a.Trace.Tokens) > 0 {
				if scope := a.inspectorScope(); scope != "" {
					ui.Box(c).Grow(1)
					ui.Text(c, scope).FontSize(fontMicro).TextColor(colAccent2).Shrink(0)
				}
			}
		})
		ui.Box(c).FillWidth().Height(1).Background(colBorder)

		tr := a.Trace
		if a.SelToken < 0 || a.SelToken >= len(tr.Tokens) {
			// Both empty states are dead ends unless they name the next
			// action: with no tokens there is nothing to click, and with no
			// selection the reader needs to know what to click.
			if len(tr.Tokens) == 0 {
				mutedText(c, "检查器跟着选中的 token 走。先在上方输入一句提示词，"+
					"中栏会列出从这里开始的几条路。")
				return
			}
			mutedText(c, "点击 token 或热力图的格子，这里会显示它的数值。")
			return
		}
		// The same faint-fade as the stage area, under the same key, so the
		// two panels that describe one stage change together instead of one
		// snapping while the other eases.
		ui.Column(c).Key(a.Step).Gap(12).FillWidth().
			Transition(a.stageTransition(c)).Children(func() {
			a.inspectorToken(c, tr)
			switch a.Step {
			case 0:
				a.inspectorBytes(c, tr)
			case 1:
				a.inspectorEmbed(c, tr)
			case 2:
				a.inspectorScores(c, tr)
			case 3:
				a.inspectorAttention(c, tr)
			case 4:
				a.inspectorResidual(c)
			case 5:
				a.inspectorFFN(c)
			case 6:
				a.inspectorLogits(c, tr)
			}
			a.inspectorHoverCell(c)
		})
	})
}

func (a *App) inspectorToken(c *ui.Context, tr *model.Trace) {
	tk := tr.Tokens[a.SelToken]
	// A byte of a multi-byte character has no glyph of its own; showing the
	// model's "\xb3" escape as the headline glyph is accurate about the byte and
	// unreadable as an answer to "which token is this". The strip already shows
	// 注₂ for this token, so the header says the same thing and names the
	// character's bytes.
	glyph, detail := tk.Text, fmt.Sprintf("byte 0x%02X · rune %q", tk.ID, tk.Rune)
	if tk.Rune != 0 && tk.ID >= 0x80 {
		if rb := runeAt(tr, tk.Start); rb.OK && len(rb.Bytes) > 1 {
			idx := 0
			for k := a.SelToken; k > 0; k-- {
				if tr.Tokens[k-1].Start != tk.Start || tr.Tokens[k-1].Rune != tk.Rune {
					break
				}
				idx++
			}
			sub := "₁₂₃₄"
			if idx < len(sub)/3 {
				glyph = fmt.Sprintf("%c%s", tk.Rune, string([]rune(sub)[idx]))
			}
			parts := make([]string, 0, len(rb.Bytes))
			for _, b := range rb.Bytes {
				parts = append(parts, fmt.Sprintf("%02X", b))
			}
			detail = fmt.Sprintf("byte 0x%02X · 是 %q 的第 %d/%d 个字节（%s）",
				tk.ID, tk.Rune, idx+1, len(rb.Bytes), strings.Join(parts, " "))
		}
	}
	card(c, func() {
		ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
			ui.Box(c).Size(34, 34).Radius(6).Background(colAccent.Alpha(0.16)).Children(func() {
				ui.Text(c, glyph).FontSize(fontHero).Bold().TextColor(colAccentOnTint).NoWrap()
			})
			ui.Column(c).Gap(1).Children(func() {
				ui.Text(c, fmt.Sprintf("token #%d / %d", a.SelToken, len(tr.Tokens)-1)).FontSize(fontBody).Bold()
				ui.Text(c, detail).FontSize(fontMicro).TextColor(colFaint).NoWrap()
			})
		})
	})
}

// weightedWorkings prints how one dimension of a head's output is assembled
// from the attention weights and the value vectors. On this model a single key
// usually dominates, which is the lesson: alpha is a distribution, but the
// output is close to "the few values it points at".
func weightedWorkings(c *ui.Context, q, dim int, p weightedParts) {
	// Four decimals, matching the remainder and the total on the next line. At
	// three, each term was rounded by up to 5e-4 and three of them plus the
	// remainder put the reader's sum in the wrong fourth decimal on every
	// dimension measured (worst gap 9.8e-4) — the line is here to be re-added, so
	// the pieces and the total have to be at the same precision.
	var terms []string
	for _, tm := range p.Top {
		terms = append(terms, fmt.Sprintf("k%d %+.4f", tm.Key, tm.Val))
	}
	lines := []string{
		fmt.Sprintf("Out[%d][d%d] = Σ α·V 的三项最大贡献：%s", q, dim, strings.Join(terms, "  ")),
	}
	if p.Rest != 0 {
		lines = append(lines, fmt.Sprintf("+ 其余 %d 项 %+.4f ≈ %+.4f", q+1-len(p.Top), p.Rest, p.Out))
	} else {
		lines = append(lines, fmt.Sprintf("≈ %+.4f", p.Out))
	}
	workingsLines(c, lines)
	mutedText(c, "各项按模型的 float32 精度算，这里四舍五入到四位小数，所以自己相加时末位可能差一两个单位。")
	share := 0.0
	if p.AbsTotal > 0 {
		var top float64
		for _, tm := range p.Top {
			top += math.Abs(tm.Val)
		}
		share = 100 * top / p.AbsTotal
	}
	ui.Text(c, fmt.Sprintf("三项占绝对值总量的 %.0f%%——α 是分布，输出却往往只由少数几个 key 决定。", share)).
		FontSize(fontMicro).TextColor(colFaint)
}

// softmaxWorkings shows how the quoted weight was computed, with this cell's own
// numbers, so a reader can check the arithmetic by hand instead of taking
// "0.0096" on trust. The subtractions and the sum are the real ones: rewriting
// the printed line gives the printed weight back.
func (a *App) softmaxWorkings(c *ui.Context, h model.HeadTrace, q, k int) {
	p := computeSoftmaxParts(h, q, k)
	if !p.ok {
		return
	}
	rowMax, denom, num := p.rowMax, p.denom, p.num
	// Two lines: the row maximum is subtracted first (that is why the exponent
	// is negative even when the score is not), then the numerator over the
	// denominator. The result printed here is the weight line above.
	workingsLines(c, []string{
		fmt.Sprintf("分母 = Σ exp(score − %+.4f) = %.4f", rowMax, denom),
		fmt.Sprintf("α = exp(%+.4f − %+.4f) ÷ %.4f = %.4f",
			h.Scores[q][k], rowMax, denom, num/denom),
	})
}

func (a *App) inspectorHoverCell(c *ui.Context) {
	tr := a.Trace
	l := a.layer()
	q, k := a.cellCursor()
	if l == nil || q < 0 || k < 0 || q >= len(tr.Tokens) || k >= len(tr.Tokens) {
		return
	}
	card(c, func() {
		// Say which cursor this describes and whether it stays put: a pinned
		// cell survives the pointer leaving, which is the difference between
		// reading a number and trying to remember it.
		source := "指针"
		if a.KeyQ >= 0 && a.KeyK >= 0 {
			source = "已固定 · Esc 或再点一次解除"
		}
		sectionTitle(c, "注意力格子", source)
		h := l.Heads[a.Head]
		score := h.Scores[q][k]
		kv(c, "query", fmt.Sprintf("%d %q", q, tr.Tokens[q].Text))
		kv(c, "key", fmt.Sprintf("%d %q", k, tr.Tokens[k].Text))
		if math.IsInf(float64(score), -1) {
			kv(c, "分数", "−∞（被因果掩码遮住）")
			kv(c, "权重", "0")
			mutedText(c, "掩码位置不进 softmax：分母只对第 0 到第 "+fmt.Sprint(q)+" 列求和。")
		} else {
			kv(c, "分数", fmt.Sprintf("%+.4f", score))
			kv(c, "权重", fmt.Sprintf("%.4f", h.Attn[q][k]))
			a.softmaxWorkings(c, h, q, k)
		}
	})
}

// draw describes one deterministic sampling step: the seed the prompt hashed
// to, the position in the cumulative distribution it selected, and the interval
// that position fell in.
type draw struct {
	seed   uint64
	x      float64
	lo, hi float64
	pick   int
	prob   float64
}

// sampleDraw computes the draw sampleNext would make, without changing
// anything. The sampling card shows these numbers so "采样" stops looking like
// an unreproducible random choice: the position comes from the prompt itself,
// so the same prompt always lands in the same interval, and that interval's
// width is exactly the drawn token's probability.
func (a *App) sampleDraw(probs []float32) (draw, bool) {
	if len(probs) == 0 {
		return draw{}, false
	}
	// FNV-1a over the prompt bytes: a tiny, dependency-free hash, chosen so the
	// same prompt always draws the same token.
	var seed uint64 = 1469598103934665603
	for _, b := range []byte(a.Prompt) {
		seed ^= uint64(b)
		seed *= 1099511628211
	}
	x := float64(seed%1000003) / 1000003.0
	var acc float64
	d := draw{seed: seed, x: x, pick: -1}
	for i, p := range probs {
		lo := acc
		acc += float64(p)
		if d.pick < 0 && x <= acc {
			d.pick, d.lo, d.hi, d.prob = i, lo, acc, float64(p)
		}
	}
	if d.pick < 0 {
		// Floating point can leave acc a hair under x; fall back to the
		// smallest id that actually has mass.
		for i := len(probs) - 1; i >= 0; i-- {
			if probs[i] > 0 {
				d.pick = i
				d.hi = acc
				d.lo = acc - float64(probs[i])
				d.prob = float64(probs[i])
				break
			}
		}
	}
	if d.pick < 0 {
		return draw{}, false
	}
	return d, true
}

func (a *App) sampleNext() {
	if a.Trace == nil || len(a.Trace.Tokens) == 0 {
		a.Status = "输入一些文字后再采样"
		return
	}
	probs := a.Trace.Probabilities(a.Temp)
	if len(probs) == 0 {
		return
	}
	// Deterministic draw seeded by the prompt, so the page is reproducible.
	d, ok := a.sampleDraw(probs)
	if !ok {
		return
	}
	pick := d.pick
	a.LastSample = fmt.Sprintf("%q (id %d, p=%.3f)", byteText(pick), pick, probs[pick])
	// A byte-level model predicts bytes, not runes: append exactly one byte.
	a.Prompt += string([]byte{byte(pick)})
	a.recompute()
}

// topBars keeps the n largest rows, then restores their original order so the
// reader can still follow the sequence. Ordering by Index, not by Label: the
// labels are text like "10 x" and "2 y", which do not sort numerically.
func topBars(rows []barRow, n int) []barRow {
	sorted := append([]barRow(nil), rows...)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Value > sorted[j].Value })
	if n > len(sorted) {
		n = len(sorted)
	}
	out := sorted[:n]
	sort.SliceStable(out, func(i, j int) bool { return out[i].Index < out[j].Index })
	return out
}

// inspectorScope names the layer and head the inspector's numbers belong to, for
// the stages whose figures depend on those controls. It returns "" on stages that
// read neither, where a scope line would be noise.
func (a *App) inspectorScope() string {
	switch {
	case inStages(a.Step, layerFrom, layerTo) && inStages(a.Step, headFrom, headTo):
		return fmt.Sprintf("层 %d · 头 %d", a.Layer, a.Head)
	case inStages(a.Step, layerFrom, layerTo):
		return fmt.Sprintf("层 %d", a.Layer)
	}
	return ""
}
