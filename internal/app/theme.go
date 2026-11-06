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
	colFaint   = ui.Hex("#7C879B")
	colAccent  = ui.Hex("#8B7BFF")
	colAccent2 = ui.Hex("#4CC2FF")
	colPos     = ui.Hex("#F0596B")
	colNeg     = ui.Hex("#4C8DF6")
	colWarm    = ui.Hex("#FFB86B")
	colGreen   = ui.Hex("#3DD68C")
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
		ui.Text(c, title).FontSize(15).Bold().TextColor(colText)
		if note != "" {
			ui.Text(c, note).FontSize(11).TextColor(colFaint)
		}
	})
}

func card(c *ui.Context, fn func()) ui.Element {
	return ui.Column(c).FillWidth().Padding(14).Gap(10).
		Background(colPanel).Radius(10).Border(1, colBorder).Children(fn)
}

func mutedText(c *ui.Context, s string) ui.Element {
	return ui.Text(c, s).FontSize(12).TextColor(colMuted).LineHeight(1.45)
}

func kv(c *ui.Context, k, v string) {
	kvW(c, k, v, 70)
}

func kvW(c *ui.Context, k, v string, width float32) {
	ui.Row(c).Gap(6).AlignItems(ui.Center).Children(func() {
		ui.Text(c, k).FontSize(11).TextColor(colFaint).Width(width).NoWrap()
		ui.Text(c, v).FontSize(12).TextColor(colText).Grow(1)
	})
}
