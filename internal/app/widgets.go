package app

import (
	"fmt"
	"math"

	"github.com/egoist/mygo/ui"
)

// tokenChips draws the prompt as one chip per byte token.
func (a *App) tokenChips(c *ui.Context, tokens []modelToken) {
	ui.Row(c).Wrap().Gap(4).Children(func() {
		for i, tk := range tokens {
			i := i
			chip := ui.Box(c).Height(44).MinWidth(26).Padding(6, 7, 5, 8).Radius(7).
				Background(colPanel2).
				Border(1, colBorder)
			if i == a.SelToken {
				chip = chip.Border(1.5, colAccent).Background(colAccent.Alpha(0.12))
			}
			chip.Children(func() {
				ui.Column(c).Gap(1).Center().Children(func() {
					ui.Text(c, tk.Text).FontSize(15).TextColor(colText)
					ui.Text(c, fmt.Sprint(tk.ID)).FontSize(9).TextColor(colFaint)
				})
			})
			if chip.Clicked() {
				if a.SelToken == i {
					a.SelToken = -1
				} else {
					a.SelToken = i
				}
			}
			chip.Tooltip(fmt.Sprintf("token %d · byte 0x%02X · %q", i, tk.ID, tk.Rune))
		}
	})
}

// vectorStrip paints a row of cells, one per dimension, colored by sign.
func vectorStrip(c *ui.Context, v []float32, height float32, hover *int) {
	if len(v) == 0 {
		return
	}
	lo, hi := minMaxFloat(v)
	maxAbs := maxAbsFloat(v)
	label := fmt.Sprintf("min %+.2f · max %+.2f · ‖v‖ %.2f", lo, hi, norm(v))
	el := ui.Box(c).FillWidth().Height(height).Radius(4).
		Draw(func(p *ui.Painter, r ui.Rect) {
			cell := r.W / float32(len(v))
			for i, x := range v {
				p.Fill(ui.Rect{X: r.X + float32(i)*cell, Y: r.Y, W: cell - 0.6, H: r.H},
					valueColor(x, maxAbs), 0)
			}
			if *hover >= 0 && *hover < len(v) {
				hx := r.X + float32(*hover)*cell
				p.Stroke(ui.Rect{X: hx, Y: r.Y, W: cell, H: r.H}, colText, 0, 1.4)
			}
		})
	if x, _, ok := el.PointerPosition(); ok {
		*hover = int(x / (el.Bounds().W) * float32(len(v)))
		if *hover >= len(v) {
			*hover = len(v) - 1
		}
	}
	ui.Row(c).Gap(8).Children(func() {
		ui.Text(c, label).FontSize(10).TextColor(colFaint)
		if *hover >= 0 && *hover < len(v) {
			ui.Text(c, fmt.Sprintf("d%d = %+.4f", *hover, v[*hover])).FontSize(10).TextColor(colAccent2)
		}
	})
}

// heatmap draws a T×T attention matrix, with the causal mask left empty.
// kind "attn" colors probabilities, "score" colors raw logits after scaling.
func (a *App) heatmap(c *ui.Context, attn [][]float32, kind string) {
	T := len(attn)
	if T == 0 {
		return
	}
	var scale float32
	if kind == "score" {
		for i := 0; i < T; i++ {
			for j := 0; j <= i && j < len(attn[i]); j++ {
				if v := float32(math.Abs(float64(attn[i][j]))); v > scale {
					scale = v
				}
			}
		}
	}
	// Size the matrix from the space the window actually leaves for it, so the
	// cells stay square and every column fits instead of being squeezed by the
	// flexbox. The right-hand column of the card needs roughly 240px.
	avail := float32(420)
	if w, _ := c.Size(); w > 0 {
		if room := w - a.SidebarW - a.InspectorW - 2 - 32 - 28 - 240; room < avail {
			avail = room
		}
	}
	if avail < 160 {
		avail = 160
	}
	cell := avail / float32(T)
	if cell > 20 {
		cell = 20
	}
	if cell < 2.5 {
		cell = 2.5
	}
	size := cell * float32(T)
	q, k := -1, -1
	el := ui.Box(c).Size(size, size).Radius(4).Shrink(0).
		Draw(func(p *ui.Painter, r ui.Rect) {
			// Derive the cell from the rect we were actually given, not from the
			// requested size: a long prompt makes the box wider than the card,
			// and the layout shrinks it. Drawing with the requested cell then
			// silently cut the right-hand columns off the matrix.
			cw, ch := r.W/float32(T), r.H/float32(T)
			// The masked cells are colPanel at 45% over the card's colPanel,
			// which is colPanel again, so they paint nothing the card does not
			// already show, and neither do the 0.4px gaps between the cells:
			// draw only the cells the mask leaves visible, halving the fills
			// of a 64×64 matrix.
			for i := 0; i < T; i++ {
				row := attn[i]
				for j := 0; j <= i && j < len(row); j++ {
					c := heatColor(row[j])
					if kind == "score" {
						c = scoreColor(row[j], scale)
					}
					p.Fill(ui.Rect{
						X: r.X + float32(j)*cw,
						Y: r.Y + float32(i)*ch,
						W: cw - 0.4, H: ch - 0.4,
					}, c, 0)
				}
			}
			// Crosshair for the cell under the pointer, from the last frame.
			if a.HoverQ >= 0 && a.HoverQ < T && a.HoverK >= 0 && a.HoverK < T {
				p.Stroke(ui.Rect{
					X: r.X, Y: r.Y + float32(a.HoverQ)*ch, W: r.W, H: ch,
				}, colAccent2, 0, 1)
				p.Stroke(ui.Rect{
					X: r.X + float32(a.HoverK)*cw, Y: r.Y, W: cw, H: r.H,
				}, colAccent2, 0, 1)
			}
			// The selected query row. A stroked rectangle on the last row reads
			// as a stray underline, so tint the row and mark its left edge.
			if a.SelToken >= 0 && a.SelToken < T {
				y := r.Y + float32(a.SelToken)*ch
				p.Fill(ui.Rect{X: r.X, Y: y, W: r.W, H: ch - 0.4}, colWarm.Alpha(0.13), 0)
				p.Fill(ui.Rect{X: r.X - 3, Y: y, W: 2, H: ch - 0.4}, colWarm, 0)
			}
		})
	if b := el.Bounds(); b.W > 0 && b.H > 0 {
		cw, ch := b.W/float32(T), b.H/float32(T)
		if x, y, ok := el.PointerPosition(); ok {
			q = int(y / ch)
			k = int(x / cw)
			if q >= T {
				q = T - 1
			}
			if k >= T {
				k = T - 1
			}
			if q >= 0 && k >= 0 {
				a.HoverQ, a.HoverK = q, k
			}
		} else {
			a.HoverQ, a.HoverK = -1, -1
		}
	}
	if el.Clicked() && q >= 0 && k >= 0 {
		a.SelToken = q
	}
}

// bars draws a horizontal bar list, used for attention rows and probabilities.
func bars(c *ui.Context, rows []barRow, max float32, color ui.Color) {
	for _, r := range rows {
		row := r
		ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
			ui.Text(c, row.Label).FontSize(11).TextColor(colMuted).Width(row.LabelWidth).
				TextAlign(ui.End).NoWrap()
			bar := ui.Box(c).Grow(1).Height(14).Radius(3).Background(colPanel2).
				Draw(func(p *ui.Painter, rr ui.Rect) {
					w := float32(0)
					if max > 0 {
						w = rr.W * float32(math.Min(1, float64(row.Value/max)))
					}
					p.Fill(ui.Rect{X: rr.X, Y: rr.Y, W: w, H: rr.H}, row.Color.Alpha(0.9), 3)
				})
			if row.Tip != "" {
				bar.Tooltip(row.Tip)
			}
			ui.Text(c, row.ValueText).FontSize(11).TextColor(colFaint).Width(row.ValueWidth).NoWrap()
		})
	}
}

type barRow struct {
	Label      string
	Value      float32
	ValueText  string
	Color      ui.Color
	LabelWidth float32
	ValueWidth float32
	Tip        string
	Index      int // position in the sequence, for chronological re-ordering
}

// histogram paints the distribution of a vector's values, before and after a
// normalization.
func histogram(c *ui.Context, values []float32, bins int, color ui.Color) {
	if len(values) == 0 || bins <= 0 {
		return
	}
	lo, hi := minMaxFloat(values)
	if hi-lo < 1e-6 {
		hi = lo + 1
	}
	counts := make([]float32, bins)
	for _, v := range values {
		b := int((v - lo) / (hi - lo) * float32(bins))
		if b >= bins {
			b = bins - 1
		}
		if b < 0 {
			b = 0
		}
		counts[b]++
	}
	var peak float32
	for _, n := range counts {
		if n > peak {
			peak = n
		}
	}
	ui.Box(c).FillWidth().Height(64).Radius(4).Background(colPanel2).
		Draw(func(p *ui.Painter, r ui.Rect) {
			w := r.W / float32(bins)
			for i, n := range counts {
				h := float32(0)
				if peak > 0 {
					h = (r.H - 6) * n / peak
				}
				p.Fill(ui.Rect{
					X: r.X + float32(i)*w + 0.5, Y: r.Y + r.H - h,
					W: w - 1, H: h,
				}, color.Alpha(0.85), 1)
			}
		})
	ui.Row(c).FillWidth().Gap(0).Children(func() {
		ui.Text(c, fmt.Sprintf("%+.2f", lo)).FontSize(9).TextColor(colFaint).Width(40)
		ui.Spacer(c).Grow(1)
		ui.Text(c, fmt.Sprintf("%+.2f", hi)).FontSize(9).TextColor(colFaint).Width(40).TextAlign(ui.End)
	})
}

func minMaxFloat(v []float32) (float32, float32) {
	if len(v) == 0 {
		return 0, 0
	}
	lo, hi := v[0], v[0]
	for _, x := range v[1:] {
		if x < lo {
			lo = x
		}
		if x > hi {
			hi = x
		}
	}
	return lo, hi
}

func maxAbsFloat(v []float32) float32 {
	var m float32
	for _, x := range v {
		if x < 0 {
			x = -x
		}
		if x > m {
			m = x
		}
	}
	return m
}

func norm(v []float32) float32 {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	return float32(math.Sqrt(s))
}

func meanStd(v []float32) (float32, float32) {
	if len(v) == 0 {
		return 0, 0
	}
	var m float64
	for _, x := range v {
		m += float64(x)
	}
	m /= float64(len(v))
	var s float64
	for _, x := range v {
		d := float64(x) - m
		s += d * d
	}
	return float32(m), float32(math.Sqrt(s / float64(len(v))))
}
