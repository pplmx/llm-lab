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

// promptPresets are short examples that exercise the tokenizer's edges:
// ASCII, CJK (three bytes per rune), a prompt past the context window, and
// the empty state.
var promptPresets = []struct {
	name   string
	prompt string
}{
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

	HoverQ, HoverK       int
	HoverDimIn           int
	HoverDimQ, HoverDimK int
	HoverDimV, HoverDimH int

	SidebarW   float32
	InspectorW float32

	Scroll ui.ScrollState
	// RailScroll keeps the prompt strip's horizontal offset across stages.
	RailScroll ui.ScrollState
	// RevealRail asks the next frame to scroll the selected chip into view,
	// once. It is set when the selection moves without the pointer, so that
	// the strip never steals the scroll back from a reader browsing it.
	RevealRail bool

	LastSample string
	Status     string

	Playing bool
	NextAt  time.Time

	theme *ui.Theme

	lastPrompt string
}

// New builds the app with the bundled weights.
func New(w *model.Weights) *App {
	a := &App{
		W:          w,
		Prompt:     defaultPrompt,
		Step:       0,
		Temp:       0.8,
		SelToken:   -1,
		HoverQ:     -1,
		HoverK:     -1,
		HoverDimIn: -1,
		HoverDimQ:  -1,
		HoverDimK:  -1,
		HoverDimV:  -1,
		HoverDimH:  -1,
		SidebarW:   224,
		InspectorW: 330,
		theme:      labTheme(),
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
	a.clearHover()
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

// selectToken moves the selection. reveal asks the next frame to scroll the
// token strip to it, for selections made from outside the strip (the keyboard,
// a heatmap cell, a new prompt): the reader's eye is elsewhere, and the strip
// has to follow.
func (a *App) selectToken(i int, reveal bool) {
	a.SelToken = i
	if reveal {
		a.RevealRail = true
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
	a.HoverDimIn = -1
	a.HoverDimQ, a.HoverDimK = -1, -1
	a.HoverDimV, a.HoverDimH = -1, -1
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

// play advances the lesson every 1.3 seconds while the user watches it.
func (a *App) play(c *ui.Context) {
	if !a.Playing {
		return
	}
	now := c.Now()
	if a.NextAt.IsZero() {
		a.NextAt = now.Add(1300 * time.Millisecond)
	}
	if !now.Before(a.NextAt) {
		if a.Step >= len(lessons)-1 {
			a.Playing = false
			return
		}
		a.Step++
		a.Scroll.X, a.Scroll.Y = 0, 0
		a.clearHover()
		a.NextAt = now.Add(1300 * time.Millisecond)
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
					a.stageHeader(c)
					a.stage(c)
					a.lesson(c)
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
		a.Playing = false
		a.NextAt = time.Time{}
	case c.Shortcut(ui.Shift, ui.KeyLeft):
		a.stepToken(-1)
	case c.Shortcut(ui.Shift, ui.KeyRight):
		a.stepToken(1)
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
			FontSize(10).Bold().TextColor(colBg).
			Padding(4, 8).Radius(5).Background(colAccent).Shrink(0)
		ui.Text(c, l.title).FontSize(15).Bold().TextColor(colText)
		ui.Spacer(c).Grow(1)
		ui.Text(c, l.formula).FontSize(11).TextColor(colAccent2).
			Padding(4, 8).Radius(5).Background(colPanel2).Shrink(0)
	})
}

// ---------------------------------------------------------------- sidebar

func (a *App) sidebar(c *ui.Context) {
	ui.Column(c).Width(a.SidebarW).FillHeight().Padding(16).Gap(14).Children(func() {
		ui.Row(c).AlignItems(ui.Center).Gap(8).Children(func() {
			ui.Box(c).Size(26, 26).Radius(7).Background(colAccent).Children(func() {
				ui.Text(c, "λ").FontSize(16).Bold().TextColor(colBg)
			})
			ui.Column(c).Gap(0).Children(func() {
				ui.Text(c, "LLM Lab").FontSize(15).Bold().TextColor(colText)
				ui.Text(c, "Transformer 解剖台").FontSize(10).TextColor(colFaint)
			})
		})

		ui.Box(c).FillWidth().Height(1).Background(colBorder)

		a.stepper(c)

		ui.Spacer(c).Grow(1)

		ui.Column(c).FillWidth().Padding(10).Gap(6).Background(colPanel).Radius(8).Children(func() {
			ui.Text(c, "模型").FontSize(10).TextColor(colFaint)
			kv(c, "vocab", fmt.Sprint(a.W.Config.Vocab))
			kv(c, "d_model", fmt.Sprint(a.W.Config.DModel))
			kv(c, "layers", fmt.Sprint(a.W.Config.NLayer))
			kv(c, "heads", fmt.Sprint(a.W.Config.NHead))
			kv(c, "ctx", fmt.Sprint(a.W.Config.Ctx))
			kv(c, "参数", humanCount(a.paramCount()))
		})
		ui.Text(c, "权重："+a.W.Source).FontSize(9).TextColor(colFaint)
		if a.W.Steps > 0 {
			ui.Text(c, fmt.Sprintf("训练 %d 步 · loss %.3f", a.W.Steps, a.W.Loss)).FontSize(9).TextColor(colFaint)
		} else {
			// The one line that stops a reader from reading meaning into the
			// logits: these weights were never trained.
			ui.Text(c, "未训练 · 数值真实但输出无意义").FontSize(9).TextColor(colWarm).LineHeight(1.35)
		}
		// The status line used to sit in the top bar, where it pushed the
		// temperature slider out of the window at 1180 wide.
		ui.Text(c, a.Status).FontSize(9).TextColor(colFaint).LineHeight(1.35)
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
				a.LastSample = ""
				a.Playing = false
				a.NextAt = time.Time{}
				a.recompute()
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
			ui.Text(c, "⚠").FontSize(13).TextColor(colWarm).Shrink(0)
			ui.Text(c, "权重未训练").
				FontSize(11).Bold().TextColor(colWarm).Shrink(0)
			ui.Text(c, a.untrainedWhy()+
				"形状、掩码、每行和为 1 都是真算出来的，但权重是随机数——别从输出里读出意义。").
				FontSize(10.5).TextColor(colText).LineHeight(1.35)
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
		ui.Text(c, label).FontSize(11).TextColor(colFaint).Shrink(0)
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
	a.controlGroup(c, fmt.Sprintf("温度 %.2f", a.Temp), tempFrom, tempTo,
		"只有第 7 步的 softmax 除以温度", build)
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
				if dot.Clicked() {
					a.selectStep(i)
				}
			}
		})
		ui.Text(c, fmt.Sprintf("%d / %d · %s", a.Step+1, len(lessons), lessons[a.Step].name)).
			FontSize(11).TextColor(colMuted).NoWrap()
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
			ui.Text(c, "检查器").FontSize(12).Bold().TextColor(colMuted)
			ui.Text(c, lessons[a.Step].name).FontSize(10).TextColor(colFaint)
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
}

func (a *App) inspectorToken(c *ui.Context, tr *model.Trace) {
	tk := tr.Tokens[a.SelToken]
	card(c, func() {
		ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
			ui.Box(c).Size(34, 34).Radius(6).Background(colAccent.Alpha(0.16)).Children(func() {
				ui.Text(c, tk.Text).FontSize(16).Bold().TextColor(colAccent).NoWrap()
			})
			ui.Column(c).Gap(1).Children(func() {
				ui.Text(c, fmt.Sprintf("token #%d / %d", a.SelToken, len(tr.Tokens)-1)).FontSize(12).Bold()
				ui.Text(c, fmt.Sprintf("byte 0x%02X · rune %q", tk.ID, tk.Rune)).FontSize(10).TextColor(colFaint)
			})
		})
	})
}

func (a *App) inspectorHoverCell(c *ui.Context) {
	tr := a.Trace
	l := a.layer()
	if l == nil || a.HoverQ < 0 || a.HoverK < 0 || a.HoverQ >= len(tr.Tokens) || a.HoverK >= len(tr.Tokens) {
		return
	}
	card(c, func() {
		sectionTitle(c, "注意力格子", "")
		h := l.Heads[a.Head]
		score := h.Scores[a.HoverQ][a.HoverK]
		kv(c, "query", fmt.Sprintf("%d %q", a.HoverQ, tr.Tokens[a.HoverQ].Text))
		kv(c, "key", fmt.Sprintf("%d %q", a.HoverK, tr.Tokens[a.HoverK].Text))
		if math.IsInf(float64(score), -1) {
			kv(c, "分数", "−∞（被因果掩码遮住）")
			kv(c, "权重", "0")
		} else {
			kv(c, "分数", fmt.Sprintf("%+.4f", score))
			kv(c, "权重", fmt.Sprintf("%.4f", h.Attn[a.HoverQ][a.HoverK]))
		}
	})
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
	var seed uint64 = 1469598103934665603
	for _, b := range []byte(a.Prompt) {
		seed ^= uint64(b)
		seed *= 1099511628211
	}
	x := float64(seed%1000003) / 1000003.0
	var acc float64
	pick, lastNonZero := -1, -1
	for i, p := range probs {
		if p > 0 {
			lastNonZero = i
		}
		if pick >= 0 {
			continue
		}
		acc += float64(p)
		if x <= acc {
			pick = i
		}
	}
	// Floating point can leave acc a hair under x; fall back to the smallest
	// id that actually has mass instead of silently picking id 0.
	if pick < 0 {
		pick = lastNonZero
	}
	if pick < 0 {
		return
	}
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
