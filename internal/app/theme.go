package app

import (
	"math"
	"strconv"
	"strings"

	"github.com/egoist/mygo/ui"
)

// Palette of the lab. Kept in one place so every view shares it.
//
// Each colour answers a question, and the two accents are easy to confuse by name:
//
//	colAccent   violet — "this is the thing being shown". The selected step, the
//	            chosen layer/head, the selected token, the progress dot, and the
//	            bars of whichever row the stage is presenting.
//	colAccent2  cyan  — "this is what came out of the step". The line under a strip
//	            saying x = embedding + position, the normalised row, the merged
//	            attention output, the marked head in a comparison, and the
//	            probability ramp.
//
// The remainder are data colours and carry the meaning the reader is meant to read
// off a chart: colPos/colNeg for a signed vector, colWarm/colNeg for a scored
// matrix (see valueColor and scoreColor — the two ramps differ on purpose),
// colAccent2 for a probability ramp, colGreen for bar fills.
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

// workingsGap is the space between the lines of one workings block, measured
// from the layout this helper replaced: the embed, score, LayerNorm and FFN
// blocks each had their lines two pixels apart, so a 10px line sits 15px below
// the previous one.
//
// The sampling block was the exception — its three lines were separate children
// of the card, so they inherited the card's 10px gap and sat 23px apart. It
// asks for workingsGapWide rather than letting the refactor quietly tighten one
// card while leaving the others alone.
const (
	workingsGap     = 2
	workingsGapWide = 10
)

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

// workingRow draws one workings block and, on the block's last line, a small
// "复制" control that puts the whole derivation on the clipboard.
//
// Why copy at all: every workings block exists to be re-added by hand — each one
// says so about itself ("照着手算并得到卡片上那个值"). A reader doing that needs the
// numbers somewhere they can use, and the app's text is painted rather than
// selectable, so before this there was no way to get "0.932751" out of it except by
// reading it off the screen and retyping it.
//
// The control shares the last line's row instead of taking a row of its own. That
// is not cosmetic: every stage's column already overflows the window (measured
// MaxY 184–870 DIP at 1180×720), so a row per workings block would push real
// teaching content further out of reach to make room for a convenience.
//
// It is a Text with a tooltip rather than a Button because Buttons are focusable:
// each block would add its own Tab stop to a cycle the guide already measures in
// the double digits (see the focus table in CONTRIBUTING). Both numbers here were
// once written down — "eleven rows" and "a cycle of six" — and both had gone stale:
// the bundled prompt renders far more than eleven blocks, and the attention stage's
// cycle is twelve stops. Counting call sites is not counting either, since several
// sit inside loops. So the comment states the reason and leaves the measurement to
// the guards that re-derive it (TestCopyControlExistsOnEveryStageWithAWorkingsBlock,
// TestContributingFocusTableMatchesTheApp).
func workingsLinesCopyable(c *ui.Context, lines []string, gap ...float32) {
	g := float32(workingsGap)
	if len(gap) > 0 {
		g = gap[0]
	}
	last := len(lines) - 1
	ui.Column(c).Gap(g).Children(func() {
		for i, line := range lines {
			if i != last {
				ui.Text(c, line).FontSize(fontMicro).TextColor(colFaint)
				continue
			}
			ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
				ui.Text(c, line).FontSize(fontMicro).TextColor(colAccent2)
				ui.Spacer(c).Grow(1)
				copy := ui.Text(c, "复制").FontSize(fontMicro).TextColor(colFaint).
					Padding(1, 6).Radius(4).Background(colPanel2).
					Label("复制算式 " + strconv.Itoa(len(lines)) + " 行")
				if copy.Hovered() {
					copy = copy.TextColor(colText)
				}
				if copy.Clicked() {
					c.WriteClipboard(strings.Join(lines, "\n"))
				}
				copy.Tooltip("把这段算式复制到剪贴板")
			})
		}
	})
}

// cardNamed builds a card that the step bar's card row can jump to.
//
// chip is the short label the row shows, and it is deliberately separate from whatever
// sectionTitle the card draws: several cards change their heading with the prompt
// ("位置向量是每个位置一行" when the prompt is short, "位置向量随位置变化" otherwise),
// and a navigation label that moved with the data would be hard to click and impossible
// to test. The chip says what the card is about; the heading says what it currently
// shows.
//
// The scroll is one-shot: it fires in the frame after the click and not again, or the
// reader could not scroll the card back out of view. ScrollStart lines the card's top up
// with the viewport's rather than scrolling as little as possible — "as little as
// possible" would do nothing whenever a sliver of the target was already visible, which
// is exactly the case when a reader is working down a long stage.
func (a *App) cardNamed(c *ui.Context, chip string, fn func()) {
	el := card(c, fn)
	if a.RevealCard == chip {
		el.ScrollIntoView(ui.ScrollIntoViewOptions{Block: ui.ScrollStart})
		a.RevealCard = ""
	}
}
