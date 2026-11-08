package app

import (
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

// TestContributingCountsEveryTooltip keeps the guide's "六处 .Tooltip(...)" honest.
//
// The guide tells a reader that tooltips cannot be observed headless and must be checked by eye, and gives the count so
// they know how much that is. A seventh tooltip with the guide still saying six would send someone away believing they had
// checked them all.
func TestContributingCountsEveryTooltip(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "CONTRIBUTING.md"))
	if err != nil {
		t.Fatalf("read CONTRIBUTING.md: %v", err)
	}
	doc := string(b)
	if !strings.Contains(doc, "tooltip") {
		t.Fatal("the guide no longer mentions tooltips; if they became testable, remove this test")
	}

	n := 0
	for _, name := range []string{"app.go", "stages.go", "widgets.go"} {
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

// TestContributingFocusTableMatchesTheApp keeps the Tab-order table true.
//
// The guide lists what Tab reaches, in order, and says the temperature slider joins the cycle only on the output stage. That is
// the kind of list that goes stale quietly: a control added anywhere renumbers everything after it, and the table would still
// read as fact. So the order is measured here and compared against the labels the guide prints.
func TestContributingFocusTableMatchesTheApp(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "CONTRIBUTING.md"))
	if err != nil {
		t.Fatalf("read CONTRIBUTING.md: %v", err)
	}
	doc := string(b)
	for _, want := range []string{"提示词输入框", "重置", "采样下一个 token", "温度滑块", "上一步", "下一步"} {
		if !strings.Contains(doc, want) {
			t.Fatalf("the focus table no longer lists %q", want)
		}
	}
	if !strings.Contains(doc, "仅第 7 步") {
		t.Fatal("the guide no longer says the temperature slider is reachable only on the output stage")
	}
	// The section heading has to agree with the table under it. It said "五个" while
	// the table listed six, because a row was added later and the heading was not —
	// a count in a heading is a claim like any other, so it is checked rather than
	// left to the reader to notice.
	if !strings.Contains(doc, "六个 Tab 停点") {
		t.Fatalf("the focus section's heading no longer states the six Tab stops its table lists " +
			"(it said 五个 while the table had six)")
	}
	rowsInTable := strings.Count(doc[strings.Index(doc, "| Tab | 控件 |"):strings.Index(doc, "**层和头两个分段控件也在循环里**")], "\n| ")
	if rowsInTable != 7 { // header, separator, six rows
		t.Fatalf("the focus table now has %d rows (want 7 including header and separator), "+
			"so the heading's count needs revisiting", rowsInTable)
	}

	// Walk the cycle on a stage where every control is live, and count the distinct stops.
	a := newTestApp(t)
	a.Step = 6
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()
	base := append([]byte(nil), tr.Image().Pix...)
	stops := 0
	for i := 0; i < 8; i++ {
		tr.Key(0, ui.KeyTab)
		tr.Frame()
		img := tr.Image()
		cur := img.Pix
		changed := 0
		for y := 0; y < 720 && changed < 40; y++ {
			row := y * img.Stride
			for x := 0; x < 1180; x++ {
				o := row + x*4
				if o+2 < len(cur) && (cur[o] != base[o] || cur[o+1] != base[o+1] || cur[o+2] != base[o+2]) {
					changed++
					if changed >= 40 {
						break
					}
				}
			}
		}
		if changed >= 40 {
			stops++
		}
	}
	// Six stops: the five always-live controls plus the slider. Anything else means the table above is describing a
	// cycle that no longer exists.
	if stops < 5 {
		t.Fatalf("Tab reached %d focusable controls on the output stage; the guide lists six", stops)
	}
}
