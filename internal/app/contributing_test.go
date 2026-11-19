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

	// The table's control names, in order, are the invariant: they are what tells a keyboard
	// reader which control Tab will reach next, and they do not depend on the machine.
	//
	// The boxes do. They used to be checked too, and a box that disagreed was called a stale
	// row — but the boxes are pixels of text, so the same window measures them differently
	// wherever the fonts differ. Measured on the three CI runners, stop 1 came out 390, 394 and
	// 387 DIP wide against the 390 recorded here, and every one of the three failed while the
	// table's order and names were right. CONTRIBUTING still carries the boxes, labelled with
	// the window and the fact that they were measured there, because a reader comparing their
	// own screen wants them; what this test holds is the part that is the same everywhere.
	// Which control each stop lands on. The table names the card chips by position ("卡片索引
	// chip 1"), the app knows them by the card they open ("Q / K / V"), and both describe the
	// same stop, so a chip row is satisfied by either.
	stops := focusStops(t, 2, 1180, 720)
	for i, label := range stops {
		if label == "" {
			t.Fatalf("Tab stop %d lands on no control this test knows; add it to focusStops", i+1)
		}
		// The guide may name a control by what the reader sees rather than by its
		// accessibility label; both name the same stop.
		aliases := map[string]string{
			"输入":     "提示词输入框",
			"◀ 采样":   "采样",
			"层 分段控件": "层 分段控件",
			"头 分段控件": "头 分段控件",
		}
		named := strings.Contains(doc, "| "+label+" |")
		if a, ok := aliases[label]; ok {
			named = named || strings.Contains(doc, "| "+a+" |")
		}
		if !named && strings.HasPrefix(label, "卡片索引 chip") {
			named = strings.Contains(doc, "| "+label+" |")
		}
		if !named {
			// A card chip is listed by its position in the table and known here by its card.
			if i+1 <= len(stops) && i >= 8 {
				if strings.Contains(doc, fmt.Sprintf("| %d | 卡片索引 chip %d |", i+1, i-7)) {
					continue
				}
			}
			t.Fatalf("Tab stop %d lands on %q, which the guide's table does not name; a control "+
				"the reader is not told they can reach is one they will not find", i+1, label)
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

// focusStops names the control each Tab stop lands on, in cycle order.
//
// It asks the tester which element has the focus after each Tab, which is a question about the
// view's state rather than about pixels, so the answer does not depend on font metrics at all.
// The earlier version of this test matched the recorded ring coordinates against a table in
// CONTRIBUTING.md, and the table's boxes are pixels of text: all three CI runners failed on it
// (stop 1 measured 387, 390 and 394 DIP wide against the 390 written down) while the cycle it
// describes — which control comes next — was right on every one of them.
//
// The candidates are the app's real control texts. Ordering matters because Focused also
// reports true for a container: the prompt field is labelled "输入", a segmented control's
// value ("0", "H0") is inside the focused segment, and a card chip's ring is drawn around the
// card's own title. The longest name whose element reports focus is the one recorded.
func focusStops(t *testing.T, step, w, h int) []string {
	t.Helper()
	a := newTestApp(t)
	a.Step = step
	tr := ui.NewTester(a.View, w, h)
	tr.Frame()

	controls := []string{
		"输入", "重置", "撤销", "采样下一个 token",
		"▶ 自动播放", "⏸ 暂停", "← 上一步", "下一步 →",
		"温度",
		"Q / K / V", "分数矩阵", "掩码反事实", "加权求和",
		"注意力矩阵", "四个头", "头之间重合吗",
		"嵌入向量", "嵌入有含义吗", "位置向量", "位置编码的形状",
		"残差流", "每层说什么", "每层改多少", "LayerNorm", "残差为何不爆",
		"FFN 内部", "FFN 写出了答案", "哪个单元在干活",
		"最后位置", "每位置预测", "绑定权重", "Top-12", "采样",
	}
	// The segmented controls report through their values, which are not unique across the
	// window ("1" is both a layer and a head), so they are named by the label the reader sees
	// beside them and resolved together with that label's row.
	segments := []struct{ name, value string }{
		{"层 分段控件", "0"}, {"头 分段控件", "H0"},
	}

	rings := focusRings(t, step, w, h)
	out := make([]string, 0, len(rings))
	for n := range rings {
		// Replay the cycle up to stop n, then ask what has the focus: Focused answers about
		// the current frame, and the frame depends on how many Tabs came before it.
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, w, h)
		tr.Frame()
		for i := 0; i <= n; i++ {
			tr.Key(0, ui.KeyTab)
			tr.Frame()
		}
		best := ""
		for _, label := range controls {
			if tr.Focused(label) && len(label) > len(best) {
				best = label
			}
		}
		if best == "" {
			for _, seg := range segments {
				if tr.Focused(seg.value) {
					best = seg.name
					break
				}
			}
		}
		out = append(out, best)
	}
	return out
}

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
