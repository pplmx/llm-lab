package app

import (
	"fmt"
	"math"
	"unicode/utf8"

	"github.com/egoist/mygo/ui"

	"llmlab/internal/model"
)

// vectorStrip paints a row of cells, one per dimension, colored by sign.
func vectorStrip(c *ui.Context, v []float32, height float32, hover *int) {
	if len(v) == 0 {
		return
	}
	lo, hi := minMaxFloat(v)
	maxAbs := maxAbsFloat(v)
	label := fmt.Sprintf("min %+.2f · max %+.2f · ‖v‖ %.2f", lo, hi, norm(v))
	// pointerOn is filled in after layout and read by Draw, which runs later in the
	// same frame. It exists because the caret below must not outlive the pointer
	// while the *value* of *hover must: the formulas under the strip follow the last
	// hovered dimension on purpose, so clearing it on leave would snap them to d0
	// the moment the reader moved towards the working they were reading.
	pointerOn := false
	el := ui.Box(c).FillWidth().Height(height).Radius(4).
		Draw(func(p *ui.Painter, r ui.Rect) {
			cell := r.W / float32(len(v))
			for i, x := range v {
				p.Fill(ui.Rect{X: r.X + float32(i)*cell, Y: r.Y, W: cell - 0.6, H: r.H},
					valueColor(x, maxAbs), 0)
			}
			if pointerOn && *hover >= 0 && *hover < len(v) {
				hx := r.X + float32(*hover)*cell
				p.Stroke(ui.Rect{X: hx, Y: r.Y, W: cell, H: r.H}, colText, 0, 1.4)
			}
		})
	// onStrip says whether the pointer is over this row right now. The field
	// keeps its value when the pointer leaves, because the formulas below the
	// strip follow it: clearing on leave would snap them back to d0 the moment
	// the reader moved towards the working they were reading. What must not
	// outlive the pointer is the readout, which claims "this is the cell under
	// your cursor" and would be a lie.
	x, _, onStrip := el.PointerPosition()
	pointerOn = onStrip
	if onStrip {
		*hover = int(x / (el.Bounds().W) * float32(len(v)))
		if *hover >= len(v) {
			*hover = len(v) - 1
		}
	}
	ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
		// The two ends of the row. Without them the strip is an unlabelled
		// smear: a reader cannot tell which dimension is leftmost, how many
		// there are, or that hovering halfway gives roughly d48 — the readout
		// only appears while the pointer is on the strip, which is exactly when
		// it is not needed to explain the strip's extent.
		ui.Text(c, "d0").FontSize(fontMicro).TextColor(colFaint).Shrink(0)
		ui.Text(c, label).FontSize(fontMicro).TextColor(colFaint).Shrink(0)
		if onStrip && *hover >= 0 && *hover < len(v) {
			ui.Text(c, fmt.Sprintf("d%d = %+.4f", *hover, v[*hover])).FontSize(fontMicro).TextColor(colAccent2).Shrink(0)
		}
		// The right-hand end marker is pushed to the far edge so it sits under
		// the strip's right edge; without the spacer it followed the readout and
		// landed a fifth of the way across a 96-dimension row.
		ui.Box(c).Grow(1)
		ui.Text(c, fmt.Sprintf("d%d", len(v)-1)).FontSize(fontMicro).TextColor(colFaint).Shrink(0)
	})
}

// ------------------------------------------------------------ token rail

// tokenRail is the prompt strip that stays above the lesson on every stage.
// The token sequence is the input of all seven stages, yet it used to live
// inside the tokenising stage only: from stage 2 on, the reader saw a
// highlighted heatmap row without knowing which token it was.
func (a *App) tokenRail(c *ui.Context) {
	tr := a.Trace
	ui.Column(c).FillWidth().Padding(16, 8, 6).Shrink(0).Children(func() {
		ui.Row(c).FillWidth().Gap(10).AlignItems(ui.Center).Children(func() {
			ui.Text(c, "token").FontSize(fontMicro).TextColor(colFaint).Shrink(0).Width(34)
			if len(tr.Tokens) == 0 {
				ui.Text(c, "空提示词 · 在上面输入文字，整条流水线会从这里开始").
					FontSize(fontSmall).TextColor(colFaint)
				return
			}
			// Grow(1) plus MinWidth(0) lets the strip give up width to the
			// readout instead of pushing it out of the window at 1180px.
			ui.ScrollHorizontal(c).Grow(1).MinWidth(0).Scrollbars(false).
				TrackScroll(&a.RailScroll).Children(func() {
				ui.Row(c).Gap(3).AlignItems(ui.Center).Children(func() {
					a.tokenRailChips(c, tr.Tokens)
				})
			})
			a.tokenRailReadout(c)
		})
	})
	if a.RevealRail {
		a.RevealRail = false
	}
}

func (a *App) tokenRailChips(c *ui.Context, tokens []modelToken) {
	for i, tk := range tokens {
		i, tk := i, tk
		chip := ui.Box(c).Size(0, 32).MinWidth(24).Padding(2, 5, 12, 5).Radius(5).
			Background(colPanel).Border(1, colBorder).
			Justify(ui.Center).AlignItems(ui.Center).Label(fmt.Sprintf("token %d", i))
		marks := a.railMarks(i)
		if i == a.SelToken {
			chip = chip.Background(colAccent.Alpha(0.16)).Border(1, colAccent)
		} else if chip.Hovered() {
			chip = chip.Background(colPanel2)
		}
		if marks != "" {
			// The heatmap under the pointer names its row and column here,
			// so a crosshair in the matrix reads as two tokens.
			chip = chip.BorderWidth(1, 1, 2, 1).BorderColor(railMarkColor(marks))
		}
		// Only the glyph is a layout element: the byte id below it is painted,
		// because a second Text per chip costs about a third of a frame at 64
		// tokens and the id is a caption, not something the reader interacts
		// with.
		chip.Children(func() {
			ui.Text(c, chipGlyph(a.Trace, i)).FontSize(fontBody).TextColor(colText).NoWrap()
		})
		chip.Draw(func(p *ui.Painter, r ui.Rect) {
			// colMuted, not colFaint: the chip sits on colPanel normally,
			// colPanel2 under the pointer and an accent tint when selected.
			// colFaint measures 4.46:1 on colPanel2 and 3.58–3.88:1 on the tints
			// (3.58 at 16% over colPanel2, 3.88 at 16% over colPanel) — all under
			// the 4.5 a 10px digit needs. colMuted clears every state, worst case
			// 5.22:1.
			id := ui.Span{Text: fmt.Sprint(tk.ID), Size: fontMicro, Color: colMuted}
			w, _ := p.MeasureText(0, id)
			p.RichText(r.X+(r.W-w)/2, r.Y+r.H-13, 0, id)
		})
		if chip.Clicked() {
			// A chip the pointer reached is already on screen: do not queue a
			// reveal that would scroll the strip under the reader's hand.
			// Through selectToken rather than assigning SelToken directly, so
			// the side effects that go with a selection change — dropping a pin
			// that belonged to the previous token — happen on this path too.
			a.selectToken(i, false)
		}
		chip.Tooltip(fmt.Sprintf("token %d · byte 0x%02X · %q", i, tk.ID, tk.Rune))
		if i == a.SelToken && a.RevealRail {
			chip.ScrollIntoView()
		}
	}
}

// chipGlyph is what one token shows in the strip.
//
// The model renders a non-ASCII byte as the four-character escape "\xe6",
// which is honest about the byte but unreadable in a strip of them: a Chinese
// prompt became a row of "\xb3 \xa8 \xe6 \x84 ...". The strip is a picture of
// the text, so a byte belonging to a multi-byte character shows the character
// itself plus a subscript naming which byte of it this token is. Single-byte
// tokens keep the model's own rendering, which already covers space, tab, and
// newline with their own glyphs.
func chipGlyph(tr *model.Trace, i int) string {
	if tr == nil || i < 0 || i >= len(tr.Tokens) {
		return "?"
	}
	tk := tr.Tokens[i]
	if tk.ID < 0x80 {
		return tk.Text
	}
	if tk.Rune == 0 {
		// An invalid byte decodes to no character; there is nothing to show
		// but the byte itself.
		return tk.Text
	}
	// Which byte of its character is this token?
	n := utf8.RuneLen(tk.Rune)
	if n <= 1 {
		return tk.Text
	}
	// Walk back to the first token of this character: same Start, same rune.
	idx := 0
	for j := i; j > 0; j-- {
		if tr.Tokens[j-1].Start != tk.Start || tr.Tokens[j-1].Rune != tk.Rune {
			break
		}
		idx++
	}
	sub := "₁₂₃₄"
	if idx < len(sub)/3 {
		return fmt.Sprintf("%c%s", tk.Rune, string([]rune(sub)[idx]))
	}
	return string(tk.Rune)
}

// railMarks says how the strip marks token i for the attention cell under the
// pointer: "q" for the query row, "k" for the key column, "qk" when both.
func (a *App) railMarks(i int) string {
	if !a.hasMatrix() {
		return ""
	}
	q, k := a.cellCursor()
	var m string
	if i == q {
		m += "q"
	}
	if i == k {
		m += "k"
	}
	return m
}

func railMarkColor(marks string) ui.Color {
	switch marks {
	case "q":
		return colWarm
	case "k":
		return colAccent2
	case "qk":
		// A diagonal cell: the query and the key are the same token, which is
		// exactly when a token attends to itself. Green used to be the fallback,
		// so this common case looked like "neither" — the one reading that is
		// certainly wrong here. The mark reads qk, so the colour is the query's,
		// which is the row the reader is looking along.
		return colWarm
	}
	return colGreen
}

// tokenRailReadout answers "which token is this, and why" for whatever the
// reader is pointing at: the hovered attention cell first, the selection
// otherwise.
func (a *App) tokenRailReadout(c *ui.Context) {
	tr := a.Trace
	ui.Row(c).Gap(8).AlignItems(ui.Center).Shrink(0).Children(func() {
		cq, ck := a.cellCursor()
		switch {
		case cq >= 0 && ck >= 0 && cq < len(tr.Tokens) && ck < len(tr.Tokens):
			ui.Row(c).Gap(4).AlignItems(ui.Center).Children(func() {
				markChip(c, "Q", colWarm)
				ui.Text(c, railTokenText(tr.Tokens, cq)).FontSize(fontSmall).TextColor(colText).NoWrap()
				ui.Text(c, "×").FontSize(fontSmall).TextColor(colFaint)
				markChip(c, "K", colAccent2)
				ui.Text(c, railTokenText(tr.Tokens, ck)).FontSize(fontSmall).TextColor(colText).NoWrap()
			})
		case a.SelToken >= 0 && a.SelToken < len(tr.Tokens):
			tk := tr.Tokens[a.SelToken]
			ui.Text(c, fmt.Sprintf("#%d %q · byte %d", a.SelToken, tk.Rune, tk.ID)).
				FontSize(fontSmall).TextColor(colMuted).NoWrap()
			ui.Text(c, fmt.Sprintf("位置 %d / %d", a.SelToken, len(tr.Tokens)-1)).
				FontSize(fontMicro).TextColor(colFaint).NoWrap()
		default:
			ui.Text(c, "点一个 token 看数值").FontSize(fontMicro).TextColor(colFaint).NoWrap()
		}
	})
}

func markChip(c *ui.Context, label string, color ui.Color) {
	ui.Text(c, label).FontSize(fontMicro).Bold().TextColor(colBg).
		Padding(2, 4).Radius(3).Background(color).Shrink(0)
}

func railTokenText(tokens []modelToken, i int) string {
	if i < 0 || i >= len(tokens) {
		return "?"
	}
	return fmt.Sprintf("#%d %q", i, tokens[i].Rune)
}

// ------------------------------------------------------------ stepper

// stepper is the lesson as one vertical pipeline: a connecting rail, a dot
// per stage, and three states — done, current, still to come. The old list
// gave every row the same weight, so the seven stages read as a menu rather
// than as the sequence of a forward pass.
func (a *App) stepper(c *ui.Context) {
	const railW, rowH = 22, 32
	ui.Column(c).Gap(0).Children(func() {
		for i, s := range lessons {
			i, s := i, s
			state := 0 // 0 = done, 1 = current, 2 = ahead
			switch {
			case i == a.Step:
				state = 1
			case i > a.Step:
				state = 2
			}
			row := ui.Row(c).FillWidth().Height(rowH).Radius(6).Gap(8).
				AlignItems(ui.Center).Padding(0, 6).Label(fmt.Sprintf("步骤 %d %s", i+1, s.name))
			switch state {
			case 1:
				row = row.Background(colAccent.Alpha(0.14))
			default:
				if row.Hovered() {
					row = row.Background(colPanel2)
				}
			}
			row.Children(func() {
				ui.Box(c).Width(railW).Height(rowH).Shrink(0).
					Draw(func(p *ui.Painter, r ui.Rect) {
						drawStepDot(p, r, i, len(lessons), state, a.Step)
					})
				name := ui.Text(c, s.name).FontSize(fontBody).TextColor(stepTextColor(state))
				if state == 1 {
					name = name.Bold()
				}
			})
			if row.Clicked() {
				a.selectStep(i)
			}
			row.Tooltip(fmt.Sprintf("第 %d / %d 步 · %s", i+1, len(lessons), s.title))
		}
	})
}

func stepTextColor(state int) ui.Color {
	switch state {
	case 1:
		return colText
	case 0:
		return colMuted
	}
	return colFaint
}

// drawStepDot paints the dot of stage i and the rail segments around it. The
// segment above a dot carries the accent once the reader has passed it, so
// the rail shows how far the pass has travelled.
func drawStepDot(p *ui.Painter, r ui.Rect, i, n, state, current int) {
	cx, cy := r.X+r.W/2, r.Y+r.H/2
	const radius, lineW = 7, 2
	// Segments run to the row's edges and the rows have no gap, so the rail is
	// continuous through the dots. The segment above carries the accent once
	// the reader has passed the stage, so the rail fills up as the pass goes.
	if i > 0 {
		c := colBorder
		if i <= current {
			c = colMuted.Alpha(0.7)
		}
		top := r.Y
		bot := cy - radius - 2
		p.Fill(ui.Rect{X: cx - lineW/2, Y: top, W: lineW, H: bot - top}, c, lineW/2)
	}
	if i < n-1 {
		c := colBorder
		if i < current {
			c = colMuted.Alpha(0.7)
		}
		top := cy + radius + 2
		bot := r.Y + r.H
		p.Fill(ui.Rect{X: cx - lineW/2, Y: top, W: lineW, H: bot - top}, c, lineW/2)
	}

	num := fmt.Sprint(i + 1)
	label := ui.Span{Text: num, Size: fontMicro, Weight: 600}
	tw, th := p.MeasureText(0, label)
	switch state {
	case 1:
		p.Fill(ui.Rect{X: cx - radius, Y: cy - radius, W: 2 * radius, H: 2 * radius}, colAccent, radius)
		p.Fill(ui.Rect{X: cx - radius - 3, Y: cy - radius - 3, W: 2*radius + 6, H: 2*radius + 6},
			colAccent.Alpha(0.18), radius+3)
		label.Color = colBg
	case 0:
		// Opaque, not half-transparent: a micro-sized digit on a 50% fill of
		// colBg lands near 2.7:1, which is unreadable, and the number is the
		// stage's keyboard shortcut as much as its name.
		p.Fill(ui.Rect{X: cx - radius, Y: cy - radius, W: 2 * radius, H: 2 * radius}, colMuted, radius)
		label.Color = colBg
	default:
		p.Fill(ui.Rect{X: cx - radius, Y: cy - radius, W: 2 * radius, H: 2 * radius}, colBg, radius)
		p.Stroke(ui.Rect{X: cx - radius, Y: cy - radius, W: 2 * radius, H: 2 * radius}, colBorder, radius, 1.5)
		label.Color = colFaint
	}
	p.RichText(cx-tw/2, cy-th/2, 0, label)
}

// ------------------------------------------------------------ heatmap

// kind "attn" colors probabilities, "score" colors raw logits after scaling.
// heatmapAxed draws the attention or score matrix with position numbers along the
// top and left edges.
//
// The heading says "行 = query，列 = key", but neither edge carried a number: a
// reader had to count cells or hover to learn which key a column belonged to,
// which is the question this stage exists to answer. The labels live inside the
// box's own rect, so they cost no layout width beyond the gutter reserved here.
func (a *App) heatmapAxed(c *ui.Context, attn [][]float32, kind string, axes bool) {
	if a.heatmapAxesForTest != nil {
		axes = *a.heatmapAxesForTest
	}
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
	// The labels sit in a gutter inside the same box. Reserving it here keeps the
	// matrix inside the card at 1180px, where the card is already the narrowest
	// thing on the stage.
	var gutter ui.Rect
	if axes {
		gutter = ui.Rect{X: 20, Y: 9}
	}
	size := cell * float32(T)
	q, k := -1, -1
	el := ui.Box(c).Size(size+gutter.X, size+gutter.Y).Radius(4).Shrink(0).
		Draw(func(p *ui.Painter, r ui.Rect) {
			// Derive the cell from the rect we were actually given, not from the
			// requested size: a long prompt makes the box wider than the card,
			// and the layout shrinks it. Drawing with the requested cell then
			// silently cut the right-hand columns off the matrix.
			grid := ui.Rect{X: r.X + gutter.X, Y: r.Y + gutter.Y,
				W: r.W - gutter.X, H: r.H - gutter.Y}
			cw, ch := grid.W/float32(T), grid.H/float32(T)
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
						X: grid.X + float32(j)*cw,
						Y: grid.Y + float32(i)*ch,
						W: cw - 0.4, H: ch - 0.4,
					}, c, 0)
				}
			}
			// Crosshair for the cell under the pointer, from the last frame.
			if a.HoverQ >= 0 && a.HoverQ < T && a.HoverK >= 0 && a.HoverK < T {
				p.Stroke(ui.Rect{
					X: grid.X, Y: grid.Y + float32(a.HoverQ)*ch, W: grid.W, H: ch,
				}, colAccent2, 0, 1)
				p.Stroke(ui.Rect{
					X: grid.X + float32(a.HoverK)*cw, Y: grid.Y, W: cw, H: grid.H,
				}, colAccent2, 0, 1)
			}
			// The keyboard's cell, as a filled box rather than a crosshair: the
			// two can be on screen at once, and they have to stay tellable
			// apart. Only the pointer asks for a frame on move, so this box is
			// static until the reader presses a key.
			if q, k := a.cellCursor(); q >= 0 && q < T && k >= 0 && k < T {
				p.Stroke(ui.Rect{
					X: grid.X + float32(k)*cw - 0.5, Y: grid.Y + float32(q)*ch - 0.5,
					W: cw + 1, H: ch + 1,
				}, colWarm, 0, 2)
			}
			// Edge labels. A 64-token matrix has cells under 3 DIPs wide, so
			// labels are drawn at a stride wide enough not to overlap; the row
			// the reader selected is always labelled, since that is the one the
			// inspector and the strip agree on.
			if axes {
				put := func(text string, x, y float32, colour ui.Color) {
					sp := ui.Span{Text: text, Size: fontMicro, Color: colour}
					w, h := p.MeasureText(0, sp)
					p.RichText(x-w/2, y-h/2, 0, sp)
				}
				stride := 1
				for float32(stride)*cw < 26 {
					stride++
				}
				for j := 0; j < T; j += stride {
					put(fmt.Sprint(j), grid.X+float32(j)*cw+cw/2, grid.Y-5, colFaint)
				}
				for i := 0; i < T; i++ {
					if i%stride != 0 && i != a.SelToken {
						continue
					}
					colour := colFaint
					if i == a.SelToken {
						colour = colWarm
					}
					put(fmt.Sprint(i), grid.X-11, grid.Y+float32(i)*ch+ch/2, colour)
				}
			}
			// The selected query row. A stroked rectangle on the last row reads
			// as a stray underline, so tint the row and mark its left edge.
			if a.SelToken >= 0 && a.SelToken < T {
				y := grid.Y + float32(a.SelToken)*ch
				p.Fill(ui.Rect{X: grid.X, Y: y, W: grid.W, H: ch - 0.4}, colWarm.Alpha(0.13), 0)
				p.Fill(ui.Rect{X: grid.X - 3, Y: y, W: 2, H: ch - 0.4}, colWarm, 0)
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
			// Only the causal triangle exists. A pointer drag from a lower
			// cell to an upper one lands outside it, and the inspector would
			// otherwise print a score for a cell the model never computes.
			if q >= 0 && k >= 0 && k <= q {
				a.HoverQ, a.HoverK = q, k
			} else {
				a.HoverQ, a.HoverK = -1, -1
			}
		} else {
			a.HoverQ, a.HoverK = -1, -1
		}
	}
	// A drag can end outside the causal triangle; pinning there would put a
	// number in the inspector for a cell that does not exist.
	if el.Clicked() && q >= 0 && k >= 0 && k <= q {
		// The click landed on the matrix, not the strip: pull the strip to the
		// query row so the highlighted heatmap row and the rail agree.
		a.selectToken(q, true)
		// Pin the cell. Without this a mouse user could not keep a cell while
		// reaching for pen and paper: the hover pair is rewritten every frame
		// the pointer is elsewhere, so the inspector would forget the number
		// the reader came to copy. The keyboard already had this through
		// KeyQ/KeyK; clicking now sets the same state.
		if a.KeyQ == q && a.KeyK == k {
			a.KeyQ, a.KeyK = -1, -1 // clicking the pinned cell again unpins it
		} else {
			a.KeyQ, a.KeyK = q, k
		}
	}
}

// bars draws a horizontal bar list, used for attention rows and probabilities.
func bars(c *ui.Context, rows []barRow, max float32, color ui.Color) {
	for _, r := range rows {
		row := r
		ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
			ui.Text(c, row.Label).FontSize(fontSmall).TextColor(colMuted).Width(row.LabelWidth).
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
			ui.Text(c, row.ValueText).FontSize(fontSmall).TextColor(colFaint).Width(row.ValueWidth).NoWrap()
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
	histogramRange(c, values, bins, color, 0, 0)
}

// histogramRange draws a histogram on a caller-supplied axis. Passing lo == hi == 0
// means "use the data's own range", which is what a single chart wants.
//
// Two charts compared side by side need a shared axis: LayerNorm's before and
// after pictures each scaled to their own extremes, so a row whose values shrank
// from a span of 10.8 to one of 6.1 still filled both charts edge to edge and
// looked identical. The numbers printed underneath disagreed with the shapes.
func histogramRange(c *ui.Context, values []float32, bins int, color ui.Color, axisLo, axisHi float32) {
	if len(values) == 0 || bins <= 0 {
		return
	}
	lo, hi := axisLo, axisHi
	if hi <= lo {
		lo, hi = minMaxFloat(values)
	}
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
			// Where zero sits on this axis. The FFN histogram is the case that
			// needs it: its range runs -0.17 to +4.11, so zero lands 4% along and
			// the negative region — the thing the callout asks the reader to look
			// at — is a sliver with nothing saying where it ends. Drawn after the
			// bars so it stays visible over them.
			if lo < 0 && hi > 0 {
				x := r.X + (0-lo)/(hi-lo)*r.W
				p.Fill(ui.Rect{X: x, Y: r.Y, W: 1, H: r.H}, colText.Alpha(0.45), 0)
			}
		})
	ui.Row(c).FillWidth().Gap(0).Children(func() {
		ui.Text(c, fmt.Sprintf("%+.2f", lo)).FontSize(fontMicro).TextColor(colFaint).Width(40)
		ui.Spacer(c).Grow(1)
		ui.Text(c, fmt.Sprintf("%+.2f", hi)).FontSize(fontMicro).TextColor(colFaint).Width(40).TextAlign(ui.End)
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
