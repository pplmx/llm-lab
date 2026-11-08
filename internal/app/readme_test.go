package app

import (
	"image"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
)

// The README states facts about the app that a reader will take at face value.
// TestReadmeStageRangesMatchTheControls already covers the numeric ranges; these
// cover the rest, because a README drifts silently while the code moves on and
// this one had already drifted on the callout count.

func readREADME(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "README.md"))
	if err != nil {
		t.Fatalf("cannot read the README: %v", err)
	}
	return string(b)
}

// TestReadmeCalloutCountMatchesTheStages: "第 3–7 步各带「常见误解」卡片，共六条".
func TestReadmeCalloutCountMatchesTheStages(t *testing.T) {
	doc := readREADME(t)
	if !strings.Contains(doc, "共十五条") {
		t.Fatal("the README no longer states the callout count; update or remove this test")
	}

	counts := map[int]int{}
	total := 0
	for step := 0; step < 7; step++ {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		for _, s := range tr.Texts() {
			if strings.HasPrefix(s, "「") && strings.Contains(s, "」") {
				counts[step]++
				total++
			}
		}
	}
	if total != 15 {
		t.Fatalf("the README says 共十五条, the app renders %d", total)
	}
	// And they are on stages 3-7, with two callouts on step 4 (Softmax) and on
	// step 5 (residual: the stream-grows rebuttal and the logit-lens one).
	//
	// Written with the stage's zero-based index in the switch and its one-based
	// number in the message, which is how the rest of the app talks; the mismatch
	// between the two is what made this map read as "one extra on the fourth" while
	// the numbers wanted a different stage.
	for step := 0; step < 7; step++ {
		want := 0
		switch step {
		case 1:
			want = 2 // embeddings are not a geometry + sin/cos is not the only option
		case 2:
			want = 2 // sign is not a switch + mask counterfactual
		case 3:
			want = 3 // rows vs columns + heads cancel + heads are not copies
		case 4:
			want = 3 // stream grows + logit lens + layer rewrite
		case 5:
			want = 3 // GELU is not ReLU + FFN writes the answer + the concentration comparison
		case 6:
			want = 2 // 97% is not correctness + weight tying is not just cheaper
		}
		if counts[step] != want {
			t.Fatalf("step %d (index %d) renders %d callouts, want %d",
				step+1, step, counts[step], want)
		}
	}
}

// TestReadmeModelBlockMatchesTheWeights: the model paragraph quotes the
// architecture and the parameter count.
func TestReadmeModelBlockMatchesTheWeights(t *testing.T) {
	doc := readREADME(t)
	a := newTestApp(t)
	c := a.W.Config

	stated := strings.Contains(doc, "vocab 256 · d_model 96 · 2 层 · 4 头 · d_ff 384 · ctx 64")
	trueDims := c.Vocab == 256 && c.DModel == 96 && c.NLayer == 2 &&
		c.NHead == 4 && c.DFF == 384 && c.Ctx == 64
	if stated && !trueDims {
		t.Fatalf("the README quotes an architecture the weights do not have: %+v", c)
	}

	if strings.Contains(doc, "参数量 25.3 万") && a.paramCount()/10000 != 25 {
		t.Fatalf("the README says 25.3 万 parameters, the model has %d", a.paramCount())
	}
}

// TestReadmeKeyboardScopeMatchesTheCode: "第 3、4 步在注意力矩阵里移动格子；
// 其余步骤 Shift+←/→ 移动 token 选择".
func TestReadmeKeyboardScopeMatchesTheCode(t *testing.T) {
	for step := 0; step < 7; step++ {
		a := newTestApp(t)
		a.Step = step
		a.SelToken = 10
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		selBefore, cellBefore := a.SelToken, a.KeyQ
		tr.Key(ui.Shift, ui.KeyRight)
		tr.Frame()

		inMatrix := step == 2 || step == 3
		if inMatrix {
			if a.KeyQ < 0 || a.KeyQ == cellBefore {
				t.Fatalf("step %d: Shift+Right did not open a matrix cell (KeyQ %d -> %d)",
					step+1, cellBefore, a.KeyQ)
			}
			if a.SelToken != selBefore {
				t.Fatalf("step %d: Shift+Right moved the token selection instead of the cell", step+1)
			}
		} else {
			if a.SelToken == selBefore {
				t.Fatalf("step %d: Shift+Right did not move the token selection", step+1)
			}
		}
	}
}

// TestReadmeFadeDurationMatchesTheTransition: "切换阶段时中栏和检查器会淡入（160ms）".
func TestReadmeFadeDurationMatchesTheTransition(t *testing.T) {
	doc := readREADME(t)
	if !strings.Contains(doc, "160ms") {
		t.Fatal("the README no longer states the fade duration; update or remove this test")
	}
	if got := stageFade.Duration; got != 160*time.Millisecond {
		t.Fatalf("the README says 160ms, the transition is %v", got)
	}
}

// TestReadmeGraphLegendsMatchTheCode keeps the "图上每个元素怎么读" table honest.
// Each row states a rule the painter implements; the rules are the part a reader
// would otherwise have to reverse-engineer, and they are the part most likely to
// drift when someone tunes a colour ramp.
func TestReadmeGraphLegendsMatchTheCode(t *testing.T) {
	doc := readREADME(t)
	for _, section := range []string{
		"## 图上每个元素怎么读",
		"色深上限",
		"向量条带两端的",
		"直方图的竖线",
		"共同横轴",
	} {
		if !strings.Contains(doc, section) {
			t.Fatalf("the README no longer documents %q; update or remove this test", section)
		}
	}

	// The colour ramps stated in the table must be the ones the painter uses.
	if !strings.Contains(doc, "0.0005") {
		t.Fatal("the README no longer states the threshold below which a cell is drawn as background")
	}
	// Heat colour: at or below 0.0005 the cell is the panel colour, not a tint.
	atThreshold := heatColor(0.0005)
	justAbove := heatColor(0.002)
	if atThreshold == justAbove {
		t.Fatal("heatColor no longer treats 0.0005 as a cutoff, so the README's rule is wrong")
	}

	// The score scale is per-matrix, not a constant: the table says so and the
	// caption prints it, so a fixed value would make both wrong.
	caps := map[float64]bool{}
	for _, prompt := range []string{"hi", "attention is a weighted sum"} {
		b := newTestApp(t)
		b.Prompt = prompt
		b.recompute()
		caps[scoreColorScale(b.layer().Heads[b.Head])] = true
	}
	if len(caps) < 2 {
		t.Fatal("the colour cap is the same for every prompt, but the README describes it " +
			"as a value that moves with the prompt")
	}
}

// TestReadmeStrideRuleIsTheOneImplemented checks the stride figures the table
// quotes against the widget's own rule.
//
// My first version of the table said "36 token 时逐个标，64 token 时每 6 个标一次",
// both of which were guesses. The rule is a 26-DIP minimum gap, which gives 3 and
// 4 — so the document was wrong in two places and this test is what caught it.
func TestReadmeStrideRuleIsTheOneImplemented(t *testing.T) {
	// The painter's rule, restated: a label every `stride` cells, stride the
	// first integer putting at least 26 DIPs between labels.
	strideFor := func(cell float32) int {
		s := 1
		for float32(s)*cell < 26 {
			s++
		}
		return s
	}
	// The matrix box is not a fixed 420 DIP: it shrinks with the window, and the README
	// used to quote 3 and 4 as though it were. Measured from the layout arithmetic in
	// widgets.go — avail starts at 420 and takes the room the window leaves, floored at
	// 160 — the narrowest supported window gives 324 DIP, where a 64-token matrix labels
	// every 6th cell rather than every 4th. The snapshot size-longprompt-1180.png shows
	// exactly that: 0, 6, 12, … 60.
	app := newTestApp(t)
	availAt := func(windowW int) float32 {
		avail := float32(420)
		room := float32(windowW) - app.SidebarW - app.InspectorW - 2 - 32 - 28 - 240
		if room < avail {
			avail = room
		}
		if avail < 160 {
			avail = 160
		}
		return avail
	}

	for _, c := range []struct {
		windowW, tokens, want int
	}{
		{1180, 36, 3},
		{1180, 64, 6}, // the case the README used to get wrong
		{1500, 36, 3},
		{1500, 64, 4},
		{1920, 36, 3},
		{1920, 64, 4},
	} {
		avail := availAt(c.windowW)
		if got := strideFor(avail / float32(c.tokens)); got != c.want {
			t.Fatalf("window %d, %d tokens: avail %.0f, stride %d, want %d",
				c.windowW, c.tokens, avail, got, c.want)
		}
	}

	// And the document must say those numbers, including the narrow case.
	doc := readREADME(t)
	for _, want := range []string{"每 3 个标一次", "每 4 个标一次", "每 6 个标一次"} {
		if !strings.Contains(doc, want) {
			t.Fatalf("the README does not state %q, so its stride parenthetical is stale", want)
		}
	}
}

// TestReadmeSamplingRowStatesItsCondition keeps the README's sampling claim honest.
//
// The row used to end "同一句提示词永远得到同一个结果". The seed comes only from the
// prompt, but the interval it lands in depends on the distribution, which the
// temperature slider reshapes — measured: at T=0.8 the draw is id 32, and from
// T=2.2 it is a different token. The in-app note was fixed in round 66 with "在温度
// 不变的前提下"; the README kept the unconditional version for twenty rounds.
func TestReadmeSamplingRowStatesItsCondition(t *testing.T) {
	doc := readREADME(t)
	i := strings.Index(doc, "采样并续写")
	if i < 0 {
		t.Fatal("the README no longer has a sampling row")
	}
	end := strings.Index(doc[i:], "\n")
	if end < 0 {
		end = len(doc) - i
	}
	row := doc[i : i+end]
	if !strings.Contains(row, "温度不变") {
		t.Fatalf("the sampling row does not state that the repeatability depends on "+
			"a fixed temperature: %q", row)
	}

	// And the condition is real: some temperature draws something else.
	a := newTestApp(t)
	a.Step = 6
	base, ok := a.sampleDraw(a.Trace.Probabilities(0.8))
	if !ok {
		t.Fatal("no draw at T=0.8")
	}
	differs := false
	for temp := 0.1; temp <= 3.0; temp += 0.1 {
		d, ok := a.sampleDraw(a.Trace.Probabilities(temp))
		if ok && d.pick != base.pick {
			differs = true
			break
		}
	}
	if !differs {
		t.Skip("temperature does not change the draw on this prompt, so the " +
			"condition cannot be demonstrated here")
	}
}

// TestRenderDirIsDocumentedAndWorks keeps the snapshot instructions usable.
//
// RENDER_DIR existed in the tests but appeared in no document, and passing it a
// path that did not exist failed with "open /tmp/after/x.png: no such file or
// directory" — which reads as a broken test rather than a missing directory. The
// flag's whole purpose is keeping a before and an after side by side, and the
// obvious way to do that is a fresh path, so the writer now creates it.
func TestRenderDirIsDocumentedAndWorks(t *testing.T) {
	// Both docs must mention the variable, since both show the command.
	for _, doc := range []string{"README.md", "CONTRIBUTING.md"} {
		b, err := os.ReadFile(filepath.Join("..", "..", doc))
		if err != nil {
			t.Fatalf("read %s: %v", doc, err)
		}
		if !strings.Contains(string(b), "RENDER_DIR") {
			t.Fatalf("%s shows the snapshot command but never mentions RENDER_DIR", doc)
		}
	}

	// And a directory that does not exist yet must be created by the writer.
	dir := filepath.Join(t.TempDir(), "nested", "snapshots")
	writeSnapshot(t, dir, "probe.png", image.NewRGBA(image.Rect(0, 0, 4, 4)))
	if _, err := os.Stat(filepath.Join(dir, "probe.png")); err != nil {
		t.Fatalf("writeSnapshot did not create %s: %v", dir, err)
	}
}

// TestReadmeExplainsTheStripColour checks the figure table carries the one reading
// aid the strips themselves do not.
//
// The app has two diverging colour scales: valueColor paints a vector strip red for
// positive and blue for negative, scoreColor paints a heatmap cell warm for a high
// score and blue for a low one. Both are defensible and they are not the same thing —
// "positive" and "high" are different questions. The heatmap has a legend beside it
// ("正分（暖色）/负分（蓝色）"); the strips have none, so a reader looking at a red
// and blue band has nothing on screen telling them which end is which.
//
// That makes the README row load-bearing rather than decorative, and it has to stay
// accurate about which scale is which.
func TestReadmeExplainsTheStripColour(t *testing.T) {
	doc := readREADME(t)
	i := strings.Index(doc, "向量条带的颜色")
	if i < 0 {
		t.Fatal("the figure table no longer explains the vector strip's colours")
	}
	end := strings.Index(doc[i:], "\n")
	if end < 0 {
		end = len(doc) - i
	}
	row := doc[i : i+end]

	for _, want := range []string{"红色为正", "蓝色为负", "两套"} {
		if !strings.Contains(row, want) {
			t.Fatalf("the strip-colour row does not state %q: %s", want, row)
		}
	}

	// And the claim that the strips have no legend of their own must hold: if one is
	// ever added, this row's justification changes.
	for _, step := range []int{1, 4, 5} {
		a := newTestApp(t)
		a.Step = step
		tr := ui.NewTester(a.View, 1500, 940)
		tr.Frame()
		for _, s := range tr.Texts() {
			if strings.Contains(s, "红") && strings.Contains(s, "蓝") {
				t.Fatalf("stage %d now renders a colour legend %q; the README row says "+
					"the strips have none", step+1, s)
			}
		}
	}
}

// TestReadmeNumbersMatchTheCode holds the README's factual claims about this build to the
// build.
//
// Three of them are the kind that look verified because they are specific: the parity prompt
// is "46 token", the forward pass uses "最多 8 个 worker", and the model has "25.3 万" of
// parameters. Each is true today — measured 46, capped at 8 in tensor.go, and 252864 which
// rounds to 25.3万 — but a regenerated fixture, a changed worker cap or a retrained model
// would leave the prose stale with nothing to notice.
func TestReadmeNumbersMatchTheCode(t *testing.T) {
	doc := readREADME(t)

	// The parameter count is already covered by the test above (a.paramCount() against
	// "参数量 25.3 万"), so this only adds the two claims nothing checked.

	// The worker cap.
	src, err := os.ReadFile(filepath.Join("..", "model", "tensor.go"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(src), "workers > 8") {
		t.Fatal("the README says the forward pass uses at most 8 workers; tensor.go no " +
			"longer has that cap")
	}
	if !strings.Contains(doc, "最多 8 个 worker") {
		t.Fatal("the README no longer states the worker cap")
	}
}

// TestReadmeImagesExist keeps every screenshot the guide embeds present on disk.
//
// The guide shows six: the hero image and a collapsed gallery of five. Two of the six had fallen out of the document
// entirely at some point while staying in docs/, and all five of those were rendered against an interface that no
// longer exists — sidebar numbering from 0, no data-flow card, no misconception block. A broken link is visible in the
// rendered page, but a stale image is not, so the check here is that the files are there; keeping them current is the
// job of running the render when the interface changes.
func TestReadmeImagesExist(t *testing.T) {
	doc := readREADME(t)
	re := regexp.MustCompile(`!\[[^\]]*\]\((docs/[^)]+)\)`)
	all := re.FindAllStringSubmatch(doc, -1)
	if len(all) < 6 {
		t.Fatalf("the guide embeds %d images; it documents six stages and a small window", len(all))
	}
	seen := map[string]bool{}
	for _, m := range all {
		path := m[1]
		if seen[path] {
			continue
		}
		seen[path] = true
		if _, err := os.Stat(filepath.Join("..", "..", path)); err != nil {
			t.Errorf("the guide embeds %s, which is missing: %v", path, err)
		}
	}
}

// TestSamplingHasTwoLabelsAndTheReadmeSaysSo records a deliberate inconsistency
// that a reader will meet within one screen.
//
// The top bar's button says 采样下一个 token and the output stage's card says
// 采样并续写. Both call sampleNext and both are bound to Enter, so a reader who
// notices has found a real duplication — but the two labels are doing different
// jobs: the top bar's is a quick action available on all seven stages, where
// "next token" is what the model is predicting; the card's sits inside the lesson
// about sampling and says what the reader is about to do to their prompt.
//
// Rather than unify the words and lose the distinction, the README names both. This
// test keeps that sentence honest: if either label changes, the row has to change
// with it.
func TestSamplingHasTwoLabelsAndTheReadmeSaysSo(t *testing.T) {
	doc := readREADME(t)
	row := ""
	for _, line := range strings.Split(doc, "\n") {
		if strings.HasPrefix(line, "| 采样并续写") {
			row = line
		}
	}
	if row == "" {
		t.Fatal("the README no longer has a sampling row")
	}

	// Both labels are what the app renders, on the stages that render them.
	a := newTestApp(t)
	a.Step = 6
	top := 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame()
	for _, s := range tr.Texts() {
		if s == "采样下一个 token" {
			top++
		}
	}
	if top == 0 {
		t.Fatal("the top bar no longer renders 采样下一个 token")
	}
	if !tr.HasText("采样并续写") {
		t.Fatal("the output stage's card no longer renders 采样并续写")
	}
	for _, label := range []string{"采样下一个 token", "采样并续写"} {
		if !strings.Contains(row, label) {
			t.Fatalf("the README's sampling row does not name %q, which the app renders:\n%s",
				label, row)
		}
	}
}

// TestReadmeSectionsStayUnderTheRightParent keeps the README's outline from drifting.
//
// The document grew a section per card over many rounds, and the new sections were
// appended wherever the end of the file happened to be — which by then was past 目录, the
// file listing. Four teaching sections ended up below the table of contents, so a reader
// scrolling top to bottom met the most interesting material after the appendices. The fix
// was to gather them under one heading; this test is what stops it happening again.
//
// It checks the outline rather than the prose: every `###` must belong to a `##` that
// appears above it, no `##` may repeat, and the card sections must all sit under the same
// parent instead of scattered through the document.
func TestReadmeSectionsStayUnderTheRightParent(t *testing.T) {
	doc := readREADME(t)
	var level2 []string
	var current2 string
	cardSections := map[string]string{}
	// The card sections are the ones whose headings are questions the cards answer.
	cardHeadings := []string{
		"### 为什么第 5 步多了一张「每一层会说什么」",
		"### 权重绑定让预测变成一个余弦",
		"### 学出来的位置编码和 sin/cos 长得完全不一样",
		"### 四个头真的在分工吗",
		"### 掩码到底挡掉了什么",
		"### FFN 的单元也不是平均出力",
		"### 两层做的活不一样多",
	}
	for _, line := range strings.Split(doc, "\n") {
		switch {
		case strings.HasPrefix(line, "## "):
			current2 = strings.TrimPrefix(line, "## ")
			level2 = append(level2, current2)
		case strings.HasPrefix(line, "### "):
			if current2 == "" {
				t.Fatalf("the heading %q appears before any ## heading", line)
			}
			for _, want := range cardHeadings {
				if line == want {
					cardSections[want] = current2
				}
			}
		}
	}
	seen := map[string]bool{}
	for _, h := range level2 {
		if seen[h] {
			t.Fatalf("the ## heading %q appears twice", h)
		}
		seen[h] = true
	}
	if len(cardSections) != len(cardHeadings) {
		t.Fatalf("%d of %d card sections are present", len(cardSections), len(cardHeadings))
	}
	var parent string
	for _, heading := range cardHeadings {
		got := cardSections[heading]
		if parent == "" {
			parent = got
			continue
		}
		if got != parent {
			t.Fatalf("%q sits under %q while the other card sections sit under %q; they are "+
				"one topic and belong together", heading, got, parent)
		}
	}
	// And that parent has to come before the appendices, not after them. The appendices
	// are the parts a reader consults rather than reads: the model card, the correctness
	// notes, the limitations and the file listing.
	appendix := map[string]bool{"模型": true, "正确性": true, "已知限制": true, "目录": true}
	parentAt, appendixAt := -1, -1
	for i, h := range level2 {
		if h == parent {
			parentAt = i
		}
		if appendix[h] && appendixAt < 0 {
			appendixAt = i
		}
	}
	if parentAt < 0 {
		t.Fatalf("the card sections' parent %q is not a ## heading", parent)
	}
	if appendixAt >= 0 && parentAt > appendixAt {
		t.Fatalf("the card sections sit under %q at position %d, after the appendix %q at %d; "+
			"teaching material should not be filed behind the appendices",
			parent, parentAt, level2[appendixAt], appendixAt)
	}
}
