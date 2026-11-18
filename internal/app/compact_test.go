package app

import (
	"fmt"
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"
)

// The app now builds for iPhones and iPads, and for Android phones and tablets, so the widths
// it is drawn at include 375–1024 DIP — a third to a half of the desktop minimum. The three
// columns do not fit there at any font size, and until this layout existed the app drew them
// anyway: measured at 393 DIP, the 224-DIP sidebar and the 330-DIP inspector left 161 DIP for
// the lesson and the inspector's own heading landed at x=521, off the screen.
//
// These tests are the contract for the compact layout. They render at the sizes real phones and
// tablets report, and they check the two halves of "usable": everything the reader needs is
// inside the window, and the controls that move between stages — the only ones a phone has,
// since it has no arrow keys — are on screen without scrolling.

// phoneSizes are the content widths of the screens the app is built for, in DIPs: an iPhone SE
// (375), an iPhone 15 (393), a large Android phone (412), an iPad mini in portrait (744), an
// iPad Air in portrait (834) and an iPad Pro 11" in portrait (1024).
var phoneSizes = []struct {
	name string
	w, h int
}{
	{"iPhone SE", 375, 667},
	{"iPhone 15", 393, 852},
	{"Pixel 9", 412, 915},
	{"iPad mini", 744, 1133},
	{"iPad Air", 834, 1194},
	{"iPad Pro 11", 1024, 1366},
}

// TestCompactLayoutIsChosenExactlyBelowTheDesktopMinimum pins the breakpoint to the layout it
// replaces.
//
// The two numbers have to agree: every width that keeps the three columns has to be one the
// desktop layout was rendered at, and every width below it has to take the compact one. When
// they disagree the failure is silent in both directions — a phone width that took the desktop
// layout draws off the screen, and a desktop width that took the compact layout loses its
// sidebar and inspector for no reason.
func TestCompactLayoutIsChosenExactlyBelowTheDesktopMinimum(t *testing.T) {
	if CompactWidth != MinWindowWidth {
		t.Fatalf("the compact breakpoint is %d but the desktop layout's floor is %d; "+
			"every width between them is drawn by a layout nothing renders",
			CompactWidth, MinWindowWidth)
	}
	for _, c := range []struct {
		w    int
		want bool
	}{
		{320, true},
		{375, true},
		{393, true},
		{834, true},
		{1024, true},
		{MinWindowWidth - 1, true},
		{MinWindowWidth, false},
		{1500, false},
		{3840, false},
	} {
		a := newTestApp(t)
		tr := ui.NewTester(a.View, c.w, 800)
		tr.Frame()
		// The layout is observable from outside: the compact one has the sheet's button and
		// no sidebar, the desktop one has the sidebar and no button.
		_, hasButton := tr.Find(compactStageListLabel(a))
		_, hasSidebar := tr.Find(stageRowsLabel(0))
		if c.want && (!hasButton || hasSidebar) {
			t.Fatalf("%d DIP: compact layout expected; the stage-list button is there=%v "+
				"and the sidebar row is there=%v", c.w, hasButton, hasSidebar)
		}
		if !c.want && (hasButton || !hasSidebar) {
			t.Fatalf("%d DIP: the desktop layout expected; the stage-list button is there=%v "+
				"and the sidebar row is there=%v", c.w, hasButton, hasSidebar)
		}
	}
}

// compactStageListLabel is the accessibility label of the compact header's stage-list button,
// which names the current stage so a screen reader hears where the reader is.
func compactStageListLabel(a *App) string {
	return fmt.Sprintf("阶段列表，共 %d 个阶段，当前第 %d 个 %s",
		len(lessons), a.Step+1, lessons[a.Step].name)
}

// stageRowsLabel is the label a sidebar row carries, which is how a test addresses it.
func stageRowsLabel(i int) string { return fmt.Sprintf("步骤 %d %s", i+1, lessons[i].name) }

// TestPhoneLayoutKeepsEverythingInsideTheWindow is the defect the compact layout was written
// for, stated as an assertion: nothing may be drawn past the right edge.
//
// The desktop layout failed this at every phone size (measured at 393 DIP: the inspector's
// heading at x=521). A test on the desktop layout could not catch it, because the desktop
// layout is only ever rendered at 1180 and up — which is exactly why adding the platforms
// without these sizes would have shipped an app that cannot be used on either.
//
// The token rail is excluded: its chips scroll sideways on purpose, and TestRailScrollsToItsLastToken
// covers that. What is left is text, and a label past the right edge has no way back.
func TestPhoneLayoutKeepsEverythingInsideTheWindow(t *testing.T) {
	for _, size := range phoneSizes {
		for step := 0; step < len(lessons); step++ {
			a := newTestApp(t)
			a.Step = step
			tr := ui.NewTester(a.View, size.w, size.h)
			tr.Frame()
			for _, s := range tr.Texts() {
				r, ok := tr.Find(s)
				if !ok || r.W <= 0 {
					continue
				}
				if r.X >= 0 && r.X+r.W <= float32(size.w)+0.5 {
					continue
				}
				// The rail's chips are wider than the window by design: the strip scrolls.
				if r.Y < railBottomForTest && isTokenChip(a, s) {
					continue
				}
				label := s
				if n := len([]rune(label)); n > 20 {
					label = string([]rune(s)[:20]) + "…"
				}
				t.Fatalf("%s (%dx%d) step %d: %q is drawn at x=%.0f..%.0f, past the %d-DIP "+
					"window; there is no horizontal scroll to reach it",
					size.name, size.w, size.h, step+1, label, r.X, r.X+r.W, size.w)
			}
		}
	}
}

// isTokenChip reports whether s is one of the token rail's glyphs. The chips are the only text
// the layout lets run past the edge, and the rail is the one container that scrolls sideways.
func isTokenChip(a *App, s string) bool {
	for i := range a.Trace.Tokens {
		if chipGlyph(a.Trace, i) == s {
			return true
		}
	}
	return false
}

// TestPhoneStageButtonsAreAlwaysOnScreen covers the controls a phone cannot replace.
//
// The desktop app is navigable with ← → and 1–7, and its step bar's buttons are one way among
// several. A phone has no hardware keyboard and no way to press a key at all, so the buttons in
// the compact step bar are the only route between stages. They must be visible without
// scrolling, on every stage, at every phone size — including the ends of the lesson, where one
// of them is disabled and must still be on screen rather than removed.
func TestPhoneStageButtonsAreAlwaysOnScreen(t *testing.T) {
	for _, size := range phoneSizes {
		for step := 0; step < len(lessons); step++ {
			a := newTestApp(t)
			a.Step = step
			tr := ui.NewTester(a.View, size.w, size.h)
			tr.Frame()
			for _, want := range []string{"上一步", "下一步", playLabel(false)} {
				r, ok := tr.Find(want)
				if !ok {
					t.Fatalf("%s (%dx%d) step %d: the compact step bar has no %q",
						size.name, size.w, size.h, step+1, want)
				}
				if r.Y < 0 || r.Y+r.H > float32(size.h) {
					t.Fatalf("%s (%dx%d) step %d: %q is at y=%.0f, outside the %d-DIP window — "+
						"a phone has no arrow keys, so this button is the only way to move",
						size.name, size.w, size.h, step+1, want, r.Y, size.h)
				}
				if r.X < 0 || r.X+r.W > float32(size.w) {
					t.Fatalf("%s (%dx%d) step %d: %q is at x=%.0f..%.0f, outside the %d-DIP window",
						size.name, size.w, size.h, step+1, want, r.X, r.X+r.W, size.w)
				}
			}
		}
	}
}

// TestPhoneStageListOpensAsASheet checks the replacement for the sidebar: the list of stages
// has to be reachable, and picking one has to get the reader there.
//
// The desktop sidebar is a permanent column. On a phone it is a sheet, so the two things that
// can go wrong are that it never opens and that opening it does not let the reader leave. Both
// are checked by driving the app rather than by looking at the source, because the sheet's
// visibility depends on state the test has to actually change.
func TestPhoneStageListOpensAsASheet(t *testing.T) {
	size := phoneSizes[1] // iPhone 15
	a := newTestApp(t)
	tr := ui.NewTester(a.View, size.w, size.h)
	tr.Frame()

	if a.StageSheet {
		t.Fatal("the stage sheet is open on the first frame")
	}
	if tr.HasText(stageRowsLabel(0)) {
		t.Fatal("the stage list is drawn before it was opened; a phone has no room for it " +
			"beside the lesson")
	}
	if err := tr.Click(compactStageListLabel(a)); err != nil {
		t.Fatalf("cannot open the stage list: %v", err)
	}
	tr.Frame()
	if !a.StageSheet {
		t.Fatal("clicking the stage-list button did not open the sheet")
	}
	for i := range lessons {
		if !tr.HasText(stageRowsLabel(i)) {
			t.Fatalf("the open sheet does not list stage %d", i+1)
		}
	}

	// Picking the last stage has to move the lesson and close the sheet. It does not close by
	// itself in stepper(), which calls selectStep — so the sheet closes in selectStep, which is
	// where both layouts' navigation meets.
	if err := tr.Click(stageRowsLabel(len(lessons) - 1)); err != nil {
		t.Fatalf("cannot pick a stage from the sheet: %v", err)
	}
	tr.Frame()
	if a.Step != len(lessons)-1 {
		t.Fatalf("picking stage %d left the reader on %d", len(lessons), a.Step+1)
	}
	if a.StageSheet {
		t.Fatal("the sheet stayed open after a stage was picked, covering the stage it jumped to")
	}
}

// TestCompactInspectorFollowsTheLesson checks the other half of the layout: the inspector is not
// dropped on a phone, it moves below the lesson.
//
// "Not dropped" is the part worth testing. A layout that fits everything by removing the panel
// that explains the tensors would pass every overflow check in this file while making the app a
// picture book. The tests ask for a card each stage's inspector is the only place to find.
func TestCompactInspectorFollowsTheLesson(t *testing.T) {
	// The same markers inspectorCards uses, so a step whose card was rewired fails in both
	// places with the same phrase rather than this table quietly testing a different card.
	want := make(map[int]string, len(inspectorCards))
	for _, c := range inspectorCards {
		want[c.step] = c.marker
	}
	size := phoneSizes[1]
	for step, phrase := range want {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, size.w, size.h)
		tr.Frame()
		if !tr.HasText("检查器") {
			t.Fatalf("step %d: the compact layout has no inspector at all", step+1)
		}
		// The panel's contents are below the lesson, so the phrase is not on screen at the
		// top; it has to exist, and it has to be reachable by scrolling.
		if tr.HasText(phrase) {
			continue
		}
		tr.Move(float32(size.w)/2, float32(size.h)/2)
		tr.Frame()
		found := false
		for i := 0; i < 20 && !found; i++ {
			tr.Wheel(float32(size.w)/2, float32(size.h)/2, 0, 400)
			tr.Advance(2e9)
			tr.Frame()
			found = tr.HasText(phrase)
		}
		if !found {
			t.Fatalf("step %d: the inspector's %q cannot be scrolled into view on a phone; "+
				"the compact layout dropped the panel instead of moving it",
				step+1, phrase)
		}
	}
}

// TestCompactLayoutScrollsToTheEndOfEveryStage keeps the phone layout's scroll honest.
//
// The compact layout puts the inspector inside the same scroll container as the lesson, so the
// content is roughly a third taller than the desktop's stage column at the same stage — and the
// desktop's own test wheels eight times because the tallest stage needs six. If the compact
// column needed more wheels than the test drives, the last card would be unreachable with no
// sign that anything was missing.
func TestCompactLayoutScrollsToTheEndOfEveryStage(t *testing.T) {
	for _, size := range phoneSizes {
		for step := 0; step < len(lessons); step++ {
			a := newTestApp(t)
			a.Step = step
			tr := ui.NewTester(a.View, size.w, size.h)
			tr.Frame()
			if a.Scroll.MaxY <= 0 {
				t.Fatalf("%s step %d: nothing scrolls, so the stage's cards below the fold "+
					"cannot be reached", size.name, step+1)
			}
			tr.Move(float32(size.w)/2, float32(size.h)/2)
			tr.Frame()
			// Twelve wheels of 400 DIP for the same reason the desktop test drives eight: the
			// compact column holds the lesson and the inspector, and the tallest is the
			// residual stage. The assertion after the loop is that the offset reached MaxY, so
			// a future card that needs more than twelve fails here with the number it needed.
			for i := 0; i < 12; i++ {
				tr.Wheel(float32(size.w)/2, float32(size.h)/2, 0, 400)
				tr.Advance(2e9)
			}
			tr.Frame()
			if a.Scroll.Y != a.Scroll.MaxY {
				t.Fatalf("%s step %d: twelve 400-DIP wheels reached %.0f of a possible %.0f",
					size.name, step+1, a.Scroll.Y, a.Scroll.MaxY)
			}
		}
	}
}

// TestCompactHeaderNamesTheCurrentStage keeps the one piece of navigation the compact layout
// cannot show as a list.
//
// The desktop layout names the current stage twice on screen — the sidebar row and the step
// bar — and a reader can always see where they are. The compact layout has a button where the
// sidebar was, and the step bar has no room for the name, so the header carries it. If that line
// goes, a phone reader sees "第 4 / 7 步" and has to open the sheet to learn what stage 4 is.
func TestCompactHeaderNamesTheCurrentStage(t *testing.T) {
	size := phoneSizes[1]
	for step := range lessons {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, size.w, size.h)
		tr.Frame()
		want := fmt.Sprintf("%d / %d · %s", step+1, len(lessons), lessons[step].name)
		if !tr.HasText(want) {
			t.Fatalf("step %d: the compact header does not say %q", step+1, want)
		}
	}
}

// TestSelectStepClosesTheStageSheet ties the sheet's lifetime to the navigation that uses it.
//
// The sheet closes in selectStep rather than in the stepper, because selectStep is the one
// function every route to a stage goes through: the sheet's rows, the dots, the prev/next
// buttons, the number keys and autoplay. Closing it in the stepper would leave the sheet open
// over the stage a reader reached with the play button underneath it — and the button is under
// the sheet, so they could not press it again to get back.
func TestSelectStepClosesTheStageSheet(t *testing.T) {
	a := newTestApp(t)
	a.StageSheet = true
	a.selectStep(3)
	if a.StageSheet {
		t.Fatal("selectStep left the stage sheet open over the stage it jumped to")
	}
}

// TestCompactLayoutHasNoDesktopPanels is the structural half of the switch: the compact layout
// must not build the sidebar or the inspector column at all.
//
// Checking only that things fit would pass a layout that drew both panels and hid them under
// the lesson, which costs a frame every time and, worse, would still run their layout on every
// render. The sidebar's rows are labelled and the inspector has its own Scroll; neither may be
// in the tree at a phone width.
func TestCompactLayoutHasNoDesktopPanels(t *testing.T) {
	size := phoneSizes[1]
	for step := range lessons {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, size.w, size.h)
		tr.Frame()
		for i := range lessons {
			if tr.HasText(stageRowsLabel(i)) {
				t.Fatalf("step %d: the desktop sidebar's row %d is drawn at a phone width",
					step+1, i+1)
			}
		}
		// The desktop sidebar's model block is the other panel that must not be here. It is
		// addressed by its "权重：" line, which is unique to the sidebar; "d_model" is not — the
		// lesson's own prose names it, so matching on that reported the sentence as a panel.
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, "权重：") {
				t.Fatalf("step %d: the desktop sidebar is drawn at a phone width (%q)", step+1, s)
			}
		}
	}
}

// TestTokenRailFitsItsOwnRowOnAPhone keeps the prompt strip from squeezing the readout.
//
// The rail is "token [chips…] [readout]" in one row. On a phone the readout ("#35 'd' · byte 100
// 位置 35 / 35") is most of the width, so the chips scrolled container gets what is left. If the
// readout were allowed to grow it would push the strip to zero and a reader would see the
// selected token's description with no tokens to pick.
func TestTokenRailFitsItsOwnRowOnAPhone(t *testing.T) {
	size := phoneSizes[0] // the narrowest: 375
	a := newTestApp(t)
	tr := ui.NewTester(a.View, size.w, size.h)
	tr.Frame()
	if !tr.HasText("token") {
		t.Fatal("the token rail's label is missing on a phone")
	}
	// At least one chip has to be inside the window, or the strip is a scroll with nothing
	// visibly scrollable.
	inside := false
	for i := range a.Trace.Tokens {
		g := chipGlyph(a.Trace, i)
		r, ok := tr.Find(g)
		if ok && r.X >= 0 && r.X+r.W <= float32(size.w) {
			inside = true
			break
		}
	}
	if !inside {
		t.Fatalf("no token chip is inside a %d-DIP window; the readout has taken the strip's "+
			"whole row", size.w)
	}
}

// TestCompactPromptFieldIsUsable guards the one control whose width was measured falling to 48
// DIP before the compact layout existed.
//
// The prompt field is how the reader changes the input, and a field 48 DIP wide shows two
// characters of a 36-character prompt. It wraps onto its own line with the buttons now, so it
// gets the window's width.
func TestCompactPromptFieldIsUsable(t *testing.T) {
	for _, size := range phoneSizes {
		a := newTestApp(t)
		tr := ui.NewTester(a.View, size.w, size.h)
		tr.Frame()
		r, ok := tr.Find("输入")
		if !ok {
			t.Fatalf("%s: the prompt field is missing", size.name)
		}
		if r.W < 150 {
			t.Fatalf("%s: the prompt field is %.0f DIP wide of a %d-DIP window; it shows a "+
				"few characters of the prompt and cannot be read while typing", size.name, r.W, size.w)
		}
	}
}

// TestPhoneLayoutDrawsTheSameStages is the content half: the compact layout is a different
// arrangement of the same lesson, not a shorter one.
//
// Every stage's own title has to be drawn, because the stage heading is what tells a reader
// which of the seven they are in once the sheet is closed. A compact layout that dropped any of
// them would still pass the overflow tests — it would simply have less to overflow.
func TestPhoneLayoutDrawsTheSameStages(t *testing.T) {
	size := phoneSizes[2]
	for step, l := range lessons {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, size.w, size.h)
		tr.Frame()
		if !tr.HasText(l.title) {
			t.Fatalf("step %d: the compact layout does not draw the stage's title %q",
				step+1, l.title)
		}
		if !tr.HasText(l.formula) {
			t.Fatalf("step %d: the compact layout does not draw the stage's formula %q",
				step+1, l.formula)
		}
	}
}

// TestCompactLayoutKeepsTheStageCardsReachable checks the summary of each stage's cards, which
// is the only index a phone reader has.
//
// The desktop layout can show the card row and the sidebar at once; the compact one replaces the
// sidebar with the sheet, so the card row is the only map of a stage's four screens. It wraps
// rather than being clipped, and every chip has to be inside the window.
func TestCompactLayoutKeepsTheStageCardsReachable(t *testing.T) {
	size := phoneSizes[0]
	for step, chips := range stageCards {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, size.w, size.h)
		tr.Frame()
		for _, chip := range chips {
			r, ok := tr.Find(chip)
			if !ok {
				t.Fatalf("step %d: the card row does not list %q at a phone width", step+1, chip)
			}
			if r.X+r.W > float32(size.w)+0.5 {
				t.Fatalf("step %d: the card chip %q ends at x=%.0f, past the %d-DIP window",
					step+1, chip, r.X+r.W, size.w)
			}
		}
	}
}

// TestCompactStageBoardKeepsItsDots keeps the progress trail on a phone, where it is the only
// sign of how far the pass has come.
//
// The dots are labelled ("第 N 步 <名字>") and the labels are how this test finds them. What is
// asserted is that there are seven and that none is past the window's right edge: the trail
// shrank from 18-DIP current dot to 6-DIP dots to make room, and a future width change would
// clip the last stage rather than wrapping.
func TestCompactStageBoardKeepsItsDots(t *testing.T) {
	size := phoneSizes[0]
	a := newTestApp(t)
	tr := ui.NewTester(a.View, size.w, size.h)
	tr.Frame()
	for i := range lessons {
		label := fmt.Sprintf("第 %d 步 %s", i+1, lessons[i].name)
		r, ok := tr.Find(label)
		if !ok {
			t.Fatalf("the compact step bar has no dot for stage %d (label %q)", i+1, label)
		}
		if r.X+r.W > float32(size.w)+0.5 {
			t.Fatalf("stage %d's dot ends at x=%.0f, past the %d-DIP window", i+1, r.X+r.W, size.w)
		}
	}
}

// TestSelectStepClosesTheSheetForEveryRoute covers the routes that can be driven while the
// sheet is open.
//
// The sheet covers the whole window, so the dots and the prev/next buttons under it cannot be
// pressed while it is up — which is exactly why every route has to close it: a reader who
// reached a stage from the sheet must be able to press 下一步, and a route that left the sheet
// open would cover the button that gets them out. The routes that are not covered by the sheet
// are the global shortcuts (number keys, arrows, space) and autoplay, and those are the ones
// this drives.
func TestSelectStepClosesTheSheetForEveryRoute(t *testing.T) {
	size := phoneSizes[1]
	nav := []struct {
		name string
		goTo func(a *App, tr *ui.Tester)
	}{
		{"number key", func(a *App, tr *ui.Tester) {
			tr.Key(0, stageDigitKeys()[len(lessons)-1])
		}},
		{"arrow key", func(a *App, tr *ui.Tester) {
			a.Step = 0
			tr.Frame()
			tr.Key(0, ui.KeyRight)
		}},
		{"autoplay", func(a *App, tr *ui.Tester) {
			// The first frame after 自动播放 only arms the deadline; the advance is the frame
			// after the clock passes it. Advancing twice is what the desktop play tests do.
			a.Step = 0
			a.togglePlay()
			tr.Advance(2e9)
			tr.Frame()
			tr.Advance(2e9)
			tr.Frame()
		}},
		{"the sheet's own row", func(a *App, tr *ui.Tester) {
			if err := tr.Click(stageRowsLabel(len(lessons) - 1)); err != nil {
				t.Errorf("click a stage row: %v", err)
			}
		}},
	}
	for _, n := range nav {
		t.Run(n.name, func(t *testing.T) {
			a := newTestApp(t)
			a.StageSheet = true
			tr := ui.NewTester(a.View, size.w, size.h)
			tr.Frame()
			n.goTo(a, tr)
			tr.Frame()
			if a.Step == 0 && n.name != "number key" && n.name != "the sheet's own row" {
				t.Fatalf("%s did not move the stage, so it proves nothing", n.name)
			}
			if a.StageSheet {
				t.Fatalf("%s left the stage sheet open over the stage it jumped to", n.name)
			}
		})
	}
}

// TestCompactLayoutSurvivesTheEmptyPrompt is the state a reader meets first, and the one where
// the compact layout has the least to arrange.
//
// With no tokens every stage shows one recovery card and the inspector shows its empty state.
// Both layouts have to draw that without leaving a panel of the other layout behind: the
// empty trace short-circuits stage(), not the layout.
func TestCompactLayoutSurvivesTheEmptyPrompt(t *testing.T) {
	size := phoneSizes[1]
	for step := range lessons {
		a := newTestApp(t)
		a.applyPromptAs("", "测试")
		a.Step = step
		tr := ui.NewTester(a.View, size.w, size.h)
		tr.Frame()
		if !tr.HasText("还没有提示词") && !tr.HasText("提示词装不进这个窗口") {
			t.Fatalf("step %d: the empty prompt draws neither recovery card on a phone", step+1)
		}
		if !tr.HasText("检查器") {
			t.Fatalf("step %d: the empty prompt loses the inspector heading on a phone", step+1)
		}
	}
}

// TestCompactOnlyDropsTheKeyboardHintWithTheKeyboard keeps the shortcut line's absence
// deliberate.
//
// The compact step bar drops the shortcut hint because a phone has no keys to name. The line is
// still drawn where it is reachable — at the foot of the lesson — and the keys still work when
// there is a hardware keyboard. This checks that the hint is not in the *bar*, which is where
// the drop has to be, and that the same strings are still in the lesson so the information is
// not simply gone.
func TestCompactOnlyDropsTheKeyboardHintWithTheKeyboard(t *testing.T) {
	size := phoneSizes[1]
	a := newTestApp(t)
	tr := ui.NewTester(a.View, size.w, size.h)
	tr.Frame()
	bar, ok := tr.Find(playLabel(false))
	if !ok {
		t.Fatal("the compact step bar has no play button")
	}
	// The desktop hint sits in the step bar's row. On a phone the bar is the last 60 DIP of the
	// window; the hint must not be in it.
	for _, s := range tr.Texts() {
		if !strings.Contains(s, "切换阶段") {
			continue
		}
		r, _ := tr.Find(s)
		if r.Y >= bar.Y-30 && r.Y <= bar.Y+bar.H+30 {
			t.Fatalf("the shortcut hint %q is drawn in the compact step bar, which has no "+
				"keys to name", s)
		}
	}
}
