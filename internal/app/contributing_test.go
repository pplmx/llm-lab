package app

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"
)

// TestContributingNamesRealTests keeps the contributor guide from citing tests
// that no longer exist. The guide was written after several rounds of fixing
// "the number does not support the sentence" defects and points at the tests that
// lock each one down; a rename would leave a reader hunting for names that are
// gone.
func TestContributingNamesRealTests(t *testing.T) {
	doc, err := os.ReadFile(filepath.Join("..", "..", "CONTRIBUTING.md"))
	if err != nil {
		t.Fatalf("read CONTRIBUTING.md: %v", err)
	}
	text := string(doc)

	// Every TestXxx the guide mentions must exist in this package or in model.
	sources, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	modelSources, _ := filepath.Glob("../model/*.go")
	sources = append(sources, modelSources...)
	var all strings.Builder
	for _, f := range sources {
		b, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		all.Write(b)
	}
	declared := all.String()

	// Every TestXxx the guide names, without guessing at suffixes: the point is
	// to catch renames, and a pattern that only matches certain endings would
	// silently skip most of them.
	re := regexp.MustCompile(`Test[A-Z][A-Za-z0-9]*`)
	named := map[string]bool{}
	for _, m := range re.FindAllString(text, -1) {
		named[m] = true
	}
	if len(named) == 0 {
		t.Fatal("the guide names no tests; this test is no longer checking anything")
	}
	for name := range named {
		if !strings.Contains(declared, "func "+name+"(") {
			t.Fatalf("CONTRIBUTING.md cites %s, which no longer exists", name)
		}
	}
}

// TestContributingCoversTheTeachingChecks checks the guide still carries the
// list of figure-reading mistakes, since that list is the part of the document
// that a test cannot enforce on its own.
func TestContributingCoversTheTeachingChecks(t *testing.T) {
	doc, err := os.ReadFile(filepath.Join("..", "..", "CONTRIBUTING.md"))
	if err != nil {
		t.Fatalf("read CONTRIBUTING.md: %v", err)
	}
	text := string(doc)
	for _, want := range []string{
		"同一组图必须用同一个尺度",
		"颜色深浅要说清上限",
		"图形要标出参照点",
		"数字要支撑它旁边那句话",
		"孤立数字要有量纲参照",
		"切片要说清总体",
		"文档里的数字必须从代码派生",
		"成对的东西要由一个来源决定",
		"状态的生命周期要定义完整",
		"「发生时对不对」和「结束后对不对」是两件事",
		"断言要从**渲染出来的文本**里读数字",
		"破坏性验证要用**非默认样本**",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("the guide no longer states %q", want)
		}
	}
}

// TestContributingDoesNotRestateTheColourRange keeps one copy of a number that has to move.
//
// The score-colour ceiling depends on the prompt, the layer and the head. README states the measured per-layer spans and
// a test pins them to the model; CONTRIBUTING carried a second, independent copy reading "本模型 5.4–8.3". Neither figure
// matched anything the app produces, and because the two live in different files, changing one never flagged the other.
// The guide now points at the README instead of repeating it.
func TestContributingDoesNotRestateTheColourRange(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "CONTRIBUTING.md"))
	if err != nil {
		t.Fatalf("read CONTRIBUTING.md: %v", err)
	}
	doc := string(b)
	if strings.Contains(doc, "5.4–8.3") {
		t.Fatal("CONTRIBUTING restates the colour-ceiling range that README owns; the two will drift apart")
	}
	// It should still describe the rule, just not with its own numbers.
	if !strings.Contains(doc, "色深") {
		t.Fatal("the colour-depth point disappeared from CONTRIBUTING")
	}
	if !strings.Contains(doc, "README") {
		t.Fatal("CONTRIBUTING no longer points at README for the measured range")
	}
}

// TestContributingCountsEveryTooltip keeps the guide's "九处 .Tooltip(...)" honest.
//
// The guide tells a reader that tooltips cannot be observed headless and must be checked by eye, and gives the count so
// they know how much that is. A tenth tooltip with the guide still saying nine would send someone away believing they had
// checked them all.
//
// The count is computed from every non-test file in the package. It used to be computed from a
// hard-coded list of three, which had silently lost theme.go — the workings block's 复制 button
// lives there. The guard counted 8 while the package had 9, the guide had been written to agree
// with the guard, and both were wrong together for as long as the check stayed green. Counting
// "the files I remember" is not counting.
func TestContributingCountsEveryTooltip(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "CONTRIBUTING.md"))
	if err != nil {
		t.Fatalf("read CONTRIBUTING.md: %v", err)
	}
	doc := string(b)
	if !strings.Contains(doc, "tooltip") {
		t.Fatal("the guide no longer mentions tooltips; if they became testable, remove this test")
	}

	// Every non-test file in the package, discovered rather than listed.
	//
	// The list used to be hard-coded to app.go, stages.go and widgets.go, and it had
	// silently lost a file: theme.go builds the workings block's 复制 button and calls
	// .Tooltip on it. The guard therefore counted 8 while the package had 9, and because
	// the guide's count had been written to agree with the guard, both were wrong together
	// and the check stayed green. A count is only a check if the count is of everything.
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, name := range files {
		if strings.HasSuffix(name, "_test.go") {
			continue
		}
		src, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		n += strings.Count(string(src), ".Tooltip(")
	}
	if n == 0 {
		t.Fatal("no tooltips left; the guide's paragraph is stale")
	}

	// The guide spells the count in Chinese numerals up to ten.
	cn := [...]string{"零", "一", "两", "三", "四", "五", "六", "七", "八", "九", "十"}
	if strings.Contains(doc, cn[n]) && strings.Contains(doc, cn[n]+"处") {
		return
	}
	t.Fatalf("there are %d tooltips (%s处) but the guide does not say so", n, cn[n])
}

// TestContributingFocusTableMatchesTheApp measures the Tab cycle and compares it to the table.
//
// The guide's table is where a reader learns what Tab reaches, and it had rotted into something
// false: it listed six stops where the attention stage has eleven, and every coordinate past the
// third was wrong, because it described an interface from before the layer and head segmented
// controls and the card-index chips existed.
//
// The old version of this test could not notice. It checked that six control names appeared in
// the document, that a heading said 六个, and that the table had seven lines — all true while the
// numbers beside them described a different app — and it only required Tab to reach five stops,
// so a cycle of eleven passed. The check is now against the pixels.
func TestContributingFocusTableMatchesTheApp(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "CONTRIBUTING.md"))
	if err != nil {
		t.Fatalf("read CONTRIBUTING.md: %v", err)
	}
	doc := string(b)

	if !strings.Contains(doc, "1180×720") {
		t.Fatal("the guide no longer states the window the focus table was measured at")
	}

	// Stage 3 (index 2): the attention stage, where layer and head are both live and the stage
	// carries a three-chip card row, so the cycle includes every kind of stop.
	rings := focusRings(t, 2, 1180, 720)
	if len(rings) < 6 {
		t.Fatalf("Tab found only %d focus rings on stage 3; the cycle has collapsed", len(rings))
	}

	// The heading states the cycle's length. A count in a heading is a claim like any other and
	// goes stale the same way — it once read 五个 over a six-row table.
	cn := [...]string{"零", "一", "两", "三", "四", "五", "六", "七", "八", "九", "十",
		"十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十"}
	if len(rings) >= len(cn) {
		t.Fatalf("%d focus rings is past what this test can spell", len(rings))
	}
	if !strings.Contains(doc, cn[len(rings)]+" 个 Tab 停点") &&
		!strings.Contains(doc, cn[len(rings)]+"个 Tab 停点") {
		t.Fatalf("the attention stage has %d Tab stops but the guide's heading does not say %q",
			len(rings), cn[len(rings)]+"个 Tab 停点")
	}

	// Every measured ring is in the table, box and all. A ring missing from the document is a
	// control the reader is not told they can reach; a box that disagrees is a stale row.
	for i, r := range rings {
		want := fmt.Sprintf("(%d,%d)-(%d,%d)", int(r[0]), int(r[1]), int(r[2]), int(r[3]))
		if !strings.Contains(doc, want) {
			t.Fatalf("Tab stop %d draws its ring at %s, which the guide's table does not list; "+
				"re-measure the table rather than guessing the numbers", i+1, want)
		}
	}

	// The table's data rows must equal the cycle. This catches a row added to the document for
	// a control that does not exist, which the box check alone cannot see.
	i := strings.Index(doc, "| Tab | 控件 | 焦点环位置 |")
	if i < 0 {
		t.Fatal("the guide no longer has a focus table")
	}
	rest := doc[i:]
	if j := strings.Index(rest, "\n\n"); j >= 0 {
		rest = rest[:j]
	}
	rows := 0
	for _, line := range strings.Split(rest, "\n") {
		if strings.HasPrefix(line, "| ") {
			rows++
		}
	}
	if got := rows - 2; got != len(rings) {
		t.Fatalf("the focus table has %d data rows for a cycle of %d stops", got, len(rings))
	}
}

// focusRings walks the Tab cycle on one stage and returns each stop's focus-ring bounding box.
//
// The ring is found by differencing each frame against the *unfocused* frame: it exists only
// while a control has focus. Diffing against the previous stop instead — the obvious thing to
// write — measures two rings at once and yields a box spanning both, which is how a first draft
// reported boxes wider than any control on screen.
//
// Which widget each ring belongs to is deliberately not attempted. Pressing a key to see what
// changes cannot identify it: Right moves no app field in the text input, walks the value in the
// segmented controls, and does nothing on a button, so several distinct stops all look like
// "nothing happened". The box is the only signal that answers both "is there a ring" and "where".
func focusRings(t *testing.T, step, w, h int) [][4]float32 {
	t.Helper()
	a := newTestApp(t)
	a.Step = step
	tr := ui.NewTester(a.View, w, h)
	tr.Frame()
	base := append([]byte(nil), tr.Image().Pix...)
	var rings [][4]float32
	for n := 0; n < 24; n++ {
		tr.Key(0, ui.KeyTab)
		tr.Frame()
		img := tr.Image()
		minX, minY, maxX, maxY := 1<<30, 1<<30, -1, -1
		for y := 0; y < h; y++ {
			row := y * img.Stride
			for x := 0; x < w; x++ {
				o := row + x*4
				if o+2 >= len(img.Pix) || o+2 >= len(base) {
					continue
				}
				if img.Pix[o] == base[o] && img.Pix[o+1] == base[o+1] && img.Pix[o+2] == base[o+2] {
					continue
				}
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
		if maxX < 0 {
			continue
		}
		ring := [4]float32{float32(minX), float32(minY), float32(maxX), float32(maxY)}
		for _, prev := range rings {
			if prev == ring {
				return rings
			}
		}
		rings = append(rings, ring)
	}
	t.Fatal("the Tab cycle did not repeat within 24 presses, so it has no length to state")
	return rings
}

// TestContributingTestCountStaysAboveItsStatedBound keeps the guide's "五百多个测试函数"
// from becoming false in the direction that matters.
//
// The guide states this count so a contributor can judge what running the package costs.
// It used to state it exactly — "489 个测试函数" — and that was wrong within a couple of
// commits, because nothing re-derived it: the package had 510 by the time anyone looked.
// An exact count in prose is a number with no owner. A lower bound can be checked, and,
// being a bound, it does not need editing every time a test is added.
//
// Only the lower bound is enforced. Growth is the normal case and needs no action; a count
// that falls below what the guide promises means either tests were deleted or the sentence
// is now an overstatement, and both are worth a failure.
func TestContributingTestCountStaysAboveItsStatedBound(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "CONTRIBUTING.md"))
	if err != nil {
		t.Fatalf("read CONTRIBUTING.md: %v", err)
	}
	doc := string(b)

	// "五百多" promises strictly more than 500.
	const bound = 500
	if !strings.Contains(doc, "五百多") {
		t.Fatal("CONTRIBUTING no longer states the order of magnitude of the app's test count; " +
			"update this test with the wording you used")
	}
	// And no copy of the number may state a different bound. The guide mentions it twice —
	// the timing paragraph and the note explaining why it is a bound — and checking only
	// that *a* "五百多" exists passes while the other copy says something else. Verified by
	// mutation: changing one of the two left this test green until every occurrence was
	// required to agree.
	if got := strings.Count(doc, "五百多"); got != 2 {
		t.Fatalf("the guide states the test-count bound %d time(s); it should state it twice, "+
			"in the timing paragraph and in the note about why it is a bound", got)
	}

	files, err := filepath.Glob("*_test.go")
	if err != nil {
		t.Fatal(err)
	}
	decl := regexp.MustCompile(`(?m)^func Test`)
	n := 0
	for _, f := range files {
		src, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		n += len(decl.FindAllIndex(src, -1))
	}
	if n == 0 {
		t.Fatal("found no test functions; this guard is no longer checking anything")
	}
	if n <= bound {
		t.Fatalf("the guide says 五百多个测试函数, but this package has %d; either tests were "+
			"removed or the sentence needs correcting", n)
	}
}
