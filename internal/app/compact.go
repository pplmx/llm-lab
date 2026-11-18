package app

import (
	"fmt"

	"github.com/egoist/mygo/ui"
)

// This file is the layout the app takes on a phone, or a tablet in portrait: one column
// instead of three.
//
// The desktop layout is a sidebar of stages, a stage column and an inspector, side by side, and
// it needs MinWindowWidth to put them there without squeezing the middle one. A phone is
// 375–430 DIP wide and a tablet in portrait 744–1024, so on both the three panels do not fit at
// any font size, and drawing them anyway left the inspector's cards past the right edge
// (measured at 393 DIP: an inspector heading at x=521 in a 393-wide window).
//
// The compact layout is not a scaled-down desktop. It keeps the same view function, the same
// stages, the same keys and the same state; what changes is where the panels go:
//
//   - the stage list becomes a sheet, opened by the button in the header, so the stage column
//     gets the whole width;
//   - the inspector moves below the lesson, because "what is this token" is a question the
//     reader asks about what is above it, and a phone has no room for a second column;
//   - the step bar keeps its dots and its buttons and drops the labels that no longer fit:
//     there are no arrow keys on a phone, so the buttons are the only way to move between
//     stages, and they must stay on screen.

// compact reports whether the window is too narrow for the three-column layout.
//
// It reads the width from the frame rather than from a field set when the window opens: every
// caller already has the context, and a layout decision has to be made from the window that is
// actually being drawn — a tablet that turns from portrait to landscape has to change layouts
// without anything telling the app it was resized. A window that reports no size (the first
// frame of a tester, a zero-size box) is not compact, so nothing changes for the renderers that
// never set one.
func (a *App) compact(c *ui.Context) bool {
	w, _ := c.Size()
	return w > 0 && w < CompactWidth
}

// compactView is the one-column layout of a phone and of a tablet in portrait.
func (a *App) compactView(c *ui.Context) {
	ui.Column(c).Fill().Background(colBg).Children(func() {
		a.compactHeader(c)
		// The prompt field and the segment controls wrap rather than shrink: at 393 DIP the
		// desktop top bar's single row gave the field 48 DIP and pushed 采样 past the edge
		// (measured before this layout existed: the field at width 48, the sample button
		// ending at x=393 in a 393-wide window).
		a.topbar(c)
		a.tokenRail(c)
		// One scroll container, not two: the inspector is part of the same column, so the
		// reader reaches it by scrolling past the lesson instead of by finding a second panel
		// that does not fit.
		scroll := ui.Scroll(c).Grow(1).MinHeight(0).Padding(16, 16, 24).Gap(12).
			TrackScroll(&a.Scroll)
		if b := scroll.LayoutBounds(); b.H > 0 {
			a.StageViewportH = b.H
		}
		scroll.Children(func() {
			ui.Column(c).Key(a.Step).Gap(12).FillWidth().
				Transition(a.stageTransition(c)).Children(func() {
				a.stageHeader(c)
				a.cardRow(c)
				a.stage(c)
				a.lesson(c)
				a.inspectorHeading(c)
				a.inspectorBody(c)
			})
		})
		a.compactStepbar(c)
		// The sheet is an overlay of the whole window, not of the scroll column: it has to
		// cover the screen while it is open.
		a.stageSheet(c)
	})
}

// compactHeader is the row the desktop layout gets from the sidebar: the app's name and the
// button that opens the stage list.
func (a *App) compactHeader(c *ui.Context) {
	ui.Row(c).FillWidth().Padding(12, 10, 6).Gap(10).AlignItems(ui.Center).Shrink(0).
		Background(colPanel).Children(func() {
		// The label says what opens, not "☰": this app is a lesson, and a reader who has
		// never seen the desktop layout has no way to guess that a hamburger hides the
		// stage list.
		list := ui.Button(c, "☰ 阶段")
		if list.Clicked() {
			a.StageSheet = !a.StageSheet
		}
		// The label names the current stage, which is what the desktop sidebar shows and
		// what the step bar no longer has room for.
		list.Label(fmt.Sprintf("阶段列表，共 %d 个阶段，当前第 %d 个 %s",
			len(lessons), a.Step+1, lessons[a.Step].name))
		ui.Text(c, "LLM Lab").FontSize(fontTitle).Bold().TextColor(colText).Shrink(0)
		ui.Spacer(c).Grow(1)
		// The desktop step bar's position label, word for word: the two layouts say where the
		// reader is the same way. It carries the stage's name rather than its title for the
		// reason the desktop bar does — the title is the stage heading's job, one row below.
		ui.Text(c, fmt.Sprintf("%d / %d · %s", a.Step+1, len(lessons), lessons[a.Step].name)).
			FontSize(fontMicro).TextColor(colMuted).NoWrap().Shrink(0)
	})
}

// compactStepbar is the desktop step bar without the parts that do not fit at 393 DIP: the
// progress dots stay (they are the only sign of how far the pass has come) and the buttons that
// move between stages stay (there are no arrow keys on a phone, so they are the only way to
// move). The shortcut hint is dropped: it names keys a phone does not have.
func (a *App) compactStepbar(c *ui.Context) {
	ui.Row(c).FillWidth().Padding(12, 6, 10).Gap(8).AlignItems(ui.Center).Shrink(0).
		Background(colPanel).Children(func() {
		ui.Row(c).Gap(4).AlignItems(ui.Center).Shrink(0).Children(func() {
			a.stepDots(c)
		})
		ui.Spacer(c).Grow(1)
		ui.Row(c).Gap(6).AlignItems(ui.Center).Shrink(0).Children(func() {
			label := "▶"
			if a.Playing {
				label = "⏸"
			}
			play := ui.Button(c, label).Shrink(0)
			play.Label(playLabel(a.Playing))
			if play.Clicked() {
				a.togglePlay()
			}
			atStart, atEnd := a.Step == 0, a.Step == len(lessons)-1
			prev := ui.Button(c, "←").Shrink(0).Disabled(atStart)
			prev.Label("上一步")
			if atStart {
				prev = prev.Opacity(dimmed)
			}
			if prev.Clicked() {
				a.selectStep(a.Step - 1)
			}
			next := ui.Button(c, "→").Shrink(0).Disabled(atEnd)
			next.Label("下一步")
			if atEnd {
				next = next.Opacity(dimmed)
			}
			if next.Clicked() {
				a.selectStep(a.Step + 1)
			}
		})
	})
}

// playLabel is the accessibility label of the play/pause button, which the compact step bar
// draws as a glyph. The desktop bar spells the action out ("▶ 自动播放"), and a screen reader
// on a phone should hear the same words rather than a triangle.
func playLabel(playing bool) string {
	if playing {
		return "暂停"
	}
	return "自动播放"
}

// stageSheet shows the list of stages over the window while StageSheet is set. It is a sheet
// rather than a third panel because the stage list is a table of contents: the reader opens it,
// jumps, and is back in the lesson, so it must not cost the lesson width while it is closed.
func (a *App) stageSheet(c *ui.Context) {
	if !a.StageSheet {
		return
	}
	ui.Overlay(c, func() {
		ui.Column(c).Fill().Background(colBg.Alpha(0.92)).Padding(16, 16, 24).
			Justify(ui.Start).Children(func() {
			ui.Column(c).FillWidth().Background(colPanel).Radius(12).
				Padding(14).Gap(10).Border(1, colBorder).Children(func() {
				ui.Row(c).AlignItems(ui.Center).Gap(8).Children(func() {
					ui.Text(c, "七个阶段").FontSize(fontTitle).Bold().TextColor(colText)
					ui.Spacer(c).Grow(1)
					if ui.Button(c, "关闭").Clicked() {
						a.StageSheet = false
					}
				})
				ui.Box(c).FillWidth().Height(1).Background(colBorder)
				a.stepper(c)
			})
		})
	})
}
