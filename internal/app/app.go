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
		Prompt:     "The transformer reads the tokens and",
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
	return a
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
				a.stepbar(c)
				// MinHeight(0) lets the scroll view give up height; without it
				// the fixed rows above are squeezed to nothing at 1180×720.
				ui.Scroll(c).Grow(1).MinHeight(0).Padding(16, 16, 24).Gap(12).TrackScroll(&a.Scroll).Children(func() {
					a.stage(c)
					a.lesson(c)
				})
			})
			ui.Box(c).Width(1).FillHeight().Background(colBorder)
			a.inspector(c)
		})
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

		ui.Column(c).Gap(2).Children(func() {
			for i, s := range lessons {
				i, s := i, s
				row := ui.Row(c).FillWidth().Padding(9, 10).Radius(7).Gap(8).AlignItems(ui.Center).
					Children(func() {
						ui.Text(c, fmt.Sprintf("%d", i+1)).FontSize(10).TextColor(colFaint).Width(12)
						ui.Text(c, s.name).FontSize(12.5).TextColor(colMuted)
					})
				if i == a.Step {
					row.Background(colAccent.Alpha(0.14)).Border(1, colAccent.Alpha(0.5))
					row.Children(func() {}) // keep element stable
				}
				if row.Clicked() {
					a.selectStep(i)
				}
			}
		})

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
		}
		// The status line used to sit in the top bar, where it pushed the
		// temperature slider out of the window at 1180 wide.
		ui.Text(c, a.Status).FontSize(9).TextColor(colFaint).LineHeight(1.35)
	})
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
		ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
			field := ui.TextInput(c, &a.Prompt).Grow(1).Label("输入").SingleLine()
			if field.Changed() {
				a.LastSample = ""
				a.Playing = false
				a.NextAt = time.Time{}
				a.recompute()
			}
			if ui.Button(c, "重置").Clicked() {
				a.Prompt = "The transformer reads the tokens and"
				a.LastSample = ""
				a.Playing = false
				a.NextAt = time.Time{}
				a.recompute()
			}
			if ui.PrimaryButton(c, "采样下一个 token").Clicked() {
				a.sampleNext()
			}
		})

		ui.Row(c).Gap(12).Wrap().AlignItems(ui.Center).Children(func() {
			ui.Text(c, "层").FontSize(11).TextColor(colFaint).Shrink(0)
			labels := make([]string, a.W.Config.NLayer)
			for i := range labels {
				labels[i] = fmt.Sprintf("%d", i)
			}
			cur := a.Layer
			if ui.Segmented(c, &cur, labels...).Changed() {
				a.Layer = cur
				a.clearHover()
			}
			ui.Text(c, "头").FontSize(11).TextColor(colFaint).Shrink(0)
			heads := make([]string, a.W.Config.NHead)
			for i := range heads {
				heads[i] = fmt.Sprintf("H%d", i)
			}
			h := a.Head
			if ui.Segmented(c, &h, heads...).Changed() {
				a.Head = h
				a.clearHover()
			}
			ui.Text(c, fmt.Sprintf("温度 %.2f", a.Temp)).FontSize(11).TextColor(colFaint).Width(64).Shrink(0)
			if ui.Slider(c, &a.Temp, 0.1, 2.0).Width(130).Shrink(0).Changed() {
				a.LastSample = ""
			}
		})
	})
}

// ---------------------------------------------------------------- step bar

func (a *App) stepbar(c *ui.Context) {
	ui.Row(c).FillWidth().Padding(16, 8, 12).Gap(8).AlignItems(ui.Center).Shrink(0).Children(func() {
		// The chips wrap instead of shrinking. Without Wrap and NoWrap a narrow
		// window squeezes each chip until the label stacks one character per
		// line and the whole row grows three times taller than the top bar.
		ui.Row(c).Grow(1).Wrap().Gap(6).AlignItems(ui.Center).Children(func() {
			for i, s := range lessons {
				i, s := i, s
				chip := ui.Row(c).Padding(7, 9).Radius(6).Gap(6).AlignItems(ui.Center).
					Shrink(0).Background(colPanel2).Border(1, colBorder).
					Children(func() {
						ui.Text(c, fmt.Sprintf("%d", i+1)).FontSize(9).TextColor(colFaint).NoWrap()
						ui.Text(c, s.name).FontSize(11.5).TextColor(colMuted).NoWrap()
					})
				if i == a.Step {
					chip = chip.Background(colAccent.Alpha(0.18)).Border(1, colAccent)
				}
				if chip.Clicked() {
					a.selectStep(i)
				}
			}
		})
		ui.Row(c).Gap(6).AlignItems(ui.Center).Shrink(0).Children(func() {
			label := "▶ 自动播放"
			if a.Playing {
				label = "⏸ 暂停"
			}
			if ui.Button(c, label).Shrink(0).Clicked() {
				a.Playing = !a.Playing
				a.NextAt = time.Time{}
				if a.Playing && a.Step == len(lessons)-1 {
					a.Step = 0
					a.Scroll.X, a.Scroll.Y = 0, 0
					a.clearHover()
				}
			}
			if ui.Button(c, "← 上一步").Shrink(0).Clicked() && a.Step > 0 {
				a.selectStep(a.Step - 1)
			}
			if ui.Button(c, "下一步 →").Shrink(0).Clicked() && a.Step < len(lessons)-1 {
				a.selectStep(a.Step + 1)
			}
		})
	})
}

// ---------------------------------------------------------------- inspector

func (a *App) inspector(c *ui.Context) {
	ui.Scroll(c).Width(a.InspectorW).FillHeight().Padding(14).Gap(12).Children(func() {
		ui.Text(c, "检查器").FontSize(12).Bold().TextColor(colMuted)
		ui.Box(c).FillWidth().Height(1).Background(colBorder)

		tr := a.Trace
		if a.SelToken < 0 || a.SelToken >= len(tr.Tokens) {
			mutedText(c, "点击上方任意 token 或注意力热力图的格子，这里会显示它的数值。")
			return
		}
		tk := tr.Tokens[a.SelToken]

		card(c, func() {
			ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
				ui.Box(c).Size(34, 34).Radius(6).Background(colAccent.Alpha(0.16)).Children(func() {
					ui.Text(c, tk.Text).FontSize(16).Bold().TextColor(colAccent).NoWrap()
				})
				ui.Column(c).Gap(1).Children(func() {
					ui.Text(c, fmt.Sprintf("token #%d", a.SelToken)).FontSize(12).Bold()
					ui.Text(c, fmt.Sprintf("byte 0x%02X · rune %q", tk.ID, tk.Rune)).FontSize(10).TextColor(colFaint)
				})
			})
		})

		l := a.layer()
		if l == nil {
			return
		}

		// The attention the selected token pays to the past.
		card(c, func() {
			sectionTitle(c, "注意力分配", fmt.Sprintf("层 %d · 头 %d", a.Layer, a.Head))
			row := l.Heads[a.Head].Attn[a.SelToken]
			var max float32
			rows := make([]barRow, 0, len(row))
			for _, w := range row {
				if w > max {
					max = w
				}
			}
			for j := 0; j <= a.SelToken && j < len(row); j++ {
				rows = append(rows, barRow{
					Label:      fmt.Sprintf("%d %s", j, tr.Tokens[j].Text),
					Value:      row[j],
					ValueText:  fmt.Sprintf("%.1f%%", row[j]*100),
					Color:      colAccent,
					LabelWidth: 66,
					ValueWidth: 46,
					Tip:        fmt.Sprintf("q=%q, k=%q, %.4f", tk.Text, tr.Tokens[j].Text, row[j]),
					Index:      j,
				})
			}
			// Show the strongest few rather than all 64.
			if len(rows) > 8 {
				rows = topBars(rows, 8)
			}
			bars(c, rows, max, colAccent)
			ent := model.Entropy(row[:min(len(row), a.SelToken+1)])
			mutedText(c, fmt.Sprintf("熵 %.2f nats · 越接近 0 越专注，越接近 log(n) 越分散", ent))
		})

		if a.HoverQ >= 0 && a.HoverK >= 0 && a.HoverQ < len(tr.Tokens) && a.HoverK < len(tr.Tokens) {
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

		// Vector statistics of the selected token.
		card(c, func() {
			sectionTitle(c, "向量", "残差流")
			vec := l.AfterFFN[a.SelToken]
			mean, std := meanStd(vec)
			kv(c, "维度", fmt.Sprint(len(vec)))
			kv(c, "均值", fmt.Sprintf("%+.4f", mean))
			kv(c, "标准差", fmt.Sprintf("%.4f", std))
			kv(c, "L2", fmt.Sprintf("%.3f", norm(vec)))
			lo, hi := minMaxFloat(vec)
			kv(c, "范围", fmt.Sprintf("%+.2f … %+.2f", lo, hi))
		})

		mutedText(c, "提示：热力图中行是 query，列是 key；右上三角被因果掩码遮住。")
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
