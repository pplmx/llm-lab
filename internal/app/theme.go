package app

import (
	"math"

	"github.com/egoist/mygo/ui"
)

// Palette of the lab. Kept in one place so every view shares it.
var (
	colBg     = ui.Hex("#0E1117")
	colPanel  = ui.Hex("#151A23")
	colPanel2 = ui.Hex("#1B212C")
	colBorder = ui.Hex("#252C39")
	colText   = ui.Hex("#E7EBF3")
	colMuted  = ui.Hex("#9BA5B8")
	// colFaint is used for 9–10px labels, so it has to clear 4.5:1 on the
	// panel colour; the previous #5C6577 sat at 3.0:1 and disappeared.
	colFaint  = ui.Hex("#7C879B")
	colAccent = ui.Hex("#8B7BFF")
	// colAccentOnTint is colAccent lightened just enough to read on the 16%
	// accent tint behind the inspector's token glyph. Plain colAccent lands at
	// 4.22:1 there, under the 4.5 the small bold glyph needs; this clears 5.7:1
	// while still reading as the accent rather than as plain text.
	colAccentOnTint = ui.Hex("#A79AFF")
	colAccent2      = ui.Hex("#4CC2FF")
	colPos          = ui.Hex("#F0596B")
	colNeg          = ui.Hex("#4C8DF6")
	colWarm         = ui.Hex("#FFB86B")
	colGreen        = ui.Hex("#3DD68C")
)

// Type scale. The lesson used to carry ten sizes — 9, 10, 10.5, 11, 12, 12.5,
// 13, 14, 15, 16 — which is what happens when each new piece of text picks a
// size by eye. Six steps cover every role, named by the job rather than the
// number, and each substitution below moves a glyph by at most 1px:
//
//	fontMicro 9 10 10.5  captions, axis labels, chip names, status lines
//	fontSmall 11         secondary prose, kv keys, computed readouts
//	fontBody  12 12.5    prose, kv values, stepper labels
//	fontData  13 14      the token glyph and byte numbers being taught
//	fontTitle 15         card and section titles
//	fontHero  16         the selected token, the largest thing on screen
const (
	fontMicro = 10
	fontSmall = 11
	fontBody  = 12
	fontData  = 13
	fontTitle = 15
	fontHero  = 16
)

func labTheme() *ui.Theme {
	t := ui.DarkTheme()
	t.Background = colBg
	t.Surface = colPanel
	t.SurfaceHover = colPanel2
	t.SurfacePressed = colPanel2
	t.Border = colBorder
	t.Text = colText
	t.TextMuted = colMuted
	t.Accent = colAccent
	t.AccentHover = colAccent
	t.AccentPressed = colAccent
	t.AccentText = ui.Hex("#0E1117")
	t.Selection = colAccent.Alpha(0.35)
	t.Focus = colAccent
	t.Scrollbar = colFaint
	t.Radius = 8
	t.Spacing = 4
	t.FontSize = 13
	return t
}

// valueColor maps a signed number onto the diverging red/blue scale used for
// vectors: positive is red, negative is blue, zero fades into the panel.
func valueColor(v, maxAbs float32) ui.Color {
	if maxAbs <= 0 {
		return colFaint.Alpha(0.25)
	}
	t := float32(math.Abs(float64(v / maxAbs)))
	if t > 1 {
		t = 1
	}
	base := colPos
	if v < 0 {
		base = colNeg
	}
	return base.Alpha(0.10 + 0.90*t)
}

// heatColor maps an attention weight in [0,1] onto the accent ramp. A zero
// weight is the panel background so masked cells read as empty.
func heatColor(w float32) ui.Color {
	if w <= 0.0005 {
		return colPanel2
	}
	return colAccent.Alpha(0.12 + 0.88*float32(math.Sqrt(float64(w))))
}

func probColor(p, max float32) ui.Color {
	if max <= 0 {
		return colFaint
	}
	return colAccent2.Alpha(0.35 + 0.65*float32(math.Sqrt(float64(p/max))))
}

// scoreColor maps a raw (un-softmaxed) attention score onto a diverging scale.
// Sign carries meaning here — a negative score says "this key actively
// discourages the query" — so warm means positive and blue means negative.
func scoreColor(v, scale float32) ui.Color {
	if scale <= 0 {
		return colPanel2
	}
	t := float32(math.Abs(float64(v / scale)))
	if t > 1 {
		t = 1
	}
	base := colWarm
	if v < 0 {
		base = colNeg
	}
	return base.Alpha(0.06 + 0.9*t)
}

func sectionTitle(c *ui.Context, title, note string) {
	ui.Row(c).AlignItems(ui.Center).Gap(8).Children(func() {
		ui.Text(c, title).FontSize(fontTitle).Bold().TextColor(colText)
		if note != "" {
			ui.Text(c, note).FontSize(fontSmall).TextColor(colFaint)
		}
	})
}

func card(c *ui.Context, fn func()) ui.Element {
	return ui.Column(c).FillWidth().Padding(14).Gap(10).
		Background(colPanel).Radius(10).Border(1, colBorder).Children(fn)
}

func mutedText(c *ui.Context, s string) ui.Element {
	return ui.Text(c, s).FontSize(fontBody).TextColor(colMuted).LineHeight(1.45)
}

// misconception is the callout for a reading that the picture invites but that
// is wrong. Each one quotes the belief a reader is likely to form from what is
// on screen, then corrects it with a number this model actually produces, so
// the correction can be checked against the view above it rather than taken on
// faith.
//
// The warm left bar is the only place in the lesson that uses colWarm for
// structure rather than for data: it lets a reader who is not confused skip the
// block, while a reader who is finds it in the same place on every stage.
func misconception(c *ui.Context, belief, correction string) ui.Element {
	return ui.Row(c).FillWidth().AlignItems(ui.Start).Gap(10).
		Padding(9, 11).Radius(7).
		Background(colWarm.Alpha(0.07)).Border(1, colWarm.Alpha(0.28)).
		Children(func() {
			ui.Column(c).Gap(4).Grow(1).Children(func() {
				ui.Row(c).Gap(6).AlignItems(ui.Center).Children(func() {
					ui.Text(c, "常见误解").FontSize(fontMicro).Bold().TextColor(colWarm).Shrink(0)
					ui.Text(c, belief).FontSize(fontSmall).TextColor(colMuted).LineHeight(1.4)
				})
				ui.Text(c, correction).FontSize(fontSmall).TextColor(colText).LineHeight(1.45)
			})
		})
}

func kv(c *ui.Context, k, v string) {
	kvW(c, k, v, 70)
}

func kvW(c *ui.Context, k, v string, width float32) {
	ui.Row(c).Gap(6).AlignItems(ui.Center).Children(func() {
		ui.Text(c, k).FontSize(fontSmall).TextColor(colFaint).Width(width).NoWrap()
		ui.Text(c, v).FontSize(fontBody).TextColor(colText).Grow(1)
	})
}
