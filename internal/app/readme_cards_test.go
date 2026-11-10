package app

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"llmlab/internal/model"
)

// The README's "这些卡片量到了什么" section quotes measured figures for eight cards. Those
// numbers were all correct when written, and — unlike the numbers in the interaction tables
// and the gate timings — nothing re-derived them. The card tests in workings_test.go check
// that each card prints *its own* computed values, which is a different property: they would
// all pass happily while the README described a previous set of weights.
//
// So each test here parses the figure out of the README and compares it with the same
// computation the card uses. The weights are bundled and the default prompt is fixed, so the
// numbers are stable; a retrain that moves one of them fails with both figures in the message.
//
// One finding while writing these: the logit-lens paragraph quoted T = 1.00 numbers (62.6%
// and 93.0%) for a card that is rendered at the reader's temperature, and the app opens at
// 0.80. The same shape as the per-position loss figure corrected in the previous round. The
// README now states the temperature, and the test pins both.

// readmeCardSections returns the README's body, or fails if the section it guards is gone.
func readmeCardSections(t *testing.T) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "README.md"))
	if err != nil {
		t.Fatalf("read README.md: %v", err)
	}
	doc := string(b)
	if !strings.Contains(doc, "## 这些卡片量到了什么") {
		t.Fatal("the README no longer has the card section these tests guard; if it was " +
			"renamed or removed, update or delete this file together with it")
	}
	return doc
}

// mustNumber parses a decimal that a regex captured, so a formatting change in the README
// fails with the pattern rather than with a silently skipped comparison.
func mustNumber(t *testing.T, s string) float64 {
	t.Helper()
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		t.Fatalf("cannot parse %q as a number: %v", s, err)
	}
	return v
}

// near reports whether a measured value rounds to the figure the README states.
//
// The README quotes different precisions per figure (0.040, 103.4, 62.6%), so rather than
// invent a tolerance per line, the comparison is done at the precision the README used: the
// measured value is formatted with the same number of decimals and must match exactly. That
// makes "0.040" mean "rounds to 0.040", which is what a reader takes it to mean.
func near(t *testing.T, what, stated string, measured float64) {
	t.Helper()
	decimals := 0
	if i := strings.IndexByte(stated, '.'); i >= 0 {
		decimals = len(stated) - i - 1
	}
	got := strconv.FormatFloat(measured, 'f', decimals, 64)
	if got != stated {
		t.Fatalf("%s: the README says %s, the code computes %s", what, stated, got)
	}
}

// TestReadmeLogitLensFiguresMatchesTheLadder guards the logit-lens paragraph.
//
// Two temperatures appear in it on purpose. The ladder's own probabilities move with the
// reader's temperature and the app opens at 0.80, so 66.6% and 97.0% are what a reader sees on
// arrival; 62.6% and 93.0% are the same rungs at T = 1.00. The paragraph previously quoted only
// the second pair while the card drew the first, which is the defect this test now holds shut.
//
// The prompt-wide counts (92 of 180 rungs, 76 confidently wrong, all 36 positions) are also
// T = 1.00 figures: they are 77 at the opened temperature, because one rung sits near the 90%
// threshold. The README states the temperature for these too.
func TestReadmeLogitLensFiguresMatchesTheLadder(t *testing.T) {
	doc := readmeCardSections(t)

	// "层 1 注意力后回落到 62.6%，最终 93.0%" — the T = 1.00 pair.
	re := regexp.MustCompile(`层 1 注意力后回落到 ([0-9.]+)%，最终 ([0-9.]+)%`)
	m := re.FindStringSubmatch(doc)
	if m == nil {
		t.Fatal("the README no longer states the logit-lens rung probabilities in the form " +
			"this test knows; update the test with the sentence you wrote")
	}

	a := newTestApp(t)
	if a.Temp != 0.8 {
		t.Fatalf("the app opens at T = %.2f, but this test assumes 0.80", a.Temp)
	}
	last := len(a.Trace.Tokens) - 1
	read := func(temp float64) (attnOne, final float64) {
		lens := computeLogitLens(a.W, a.Trace, last, temp)
		for _, r := range lens.Rungs {
			switch {
			case r.Name == "层 1 注意力后":
				attnOne = 100 * float64(r.Probs[r.Top])
			case r.IsFinal:
				final = 100 * float64(r.Probs[r.Top])
			}
		}
		return attnOne, final
	}

	atOne, finalAtOne := read(1.0)
	near(t, "logit lens, layer 1 attention at T=1.00", m[1], atOne)
	near(t, "logit lens, final rung at T=1.00", m[2], finalAtOne)

	// The same rungs at the temperature the app actually opens on. The card is rendered at
	// the reader's temperature, so quoting only the T=1.00 pair describes a screen nobody
	// sees on arrival; the paragraph has to name both, and this holds it to that.
	opened, finalOpened := read(a.Temp)
	reOpened := regexp.MustCompile(`默认打开的 T = 0\.80 下是 ([0-9.]+)% 与 ([0-9.]+)%`)
	mo := reOpened.FindStringSubmatch(doc)
	if mo == nil {
		t.Fatal("the README quotes the logit-lens rungs only at T=1.00, or no longer states " +
			"the opened temperature's pair in the form this test knows; the card draws at the " +
			"reader's temperature and the app opens at 0.80")
	}
	near(t, "logit lens, layer 1 attention at the opened temperature", mo[1], opened)
	near(t, "logit lens, final rung at the opened temperature", mo[2], finalOpened)

	// The prompt-wide counts. "180 个中间层里有 92 个", "76 个对自己的错误答案有 90% 以上的
	// 把握", "36 个位置里每一个".
	reCounts := regexp.MustCompile(`(\d+) 个中间层里有 (\d+) 个的答案与最终答案不同\*?\*?，其中 \*\*?(\d+) 个` +
		``)
	mc := reCounts.FindStringSubmatch(doc)
	if mc == nil {
		t.Fatal("the README no longer states the logit-lens callout counts in the form this " +
			"test knows; update the test with the sentence you wrote")
	}
	wantRungs := mustNumber(t, mc[1])
	wantDisagree := mustNumber(t, mc[2])
	wantConfident := mustNumber(t, mc[3])

	atOneSummary := computePromptLensSummary(a.W, a.Trace, 1.0)
	if float64(atOneSummary.Rungs) != wantRungs {
		t.Fatalf("the README says %v intermediate rungs; the ladder has %d",
			wantRungs, atOneSummary.Rungs)
	}
	if float64(atOneSummary.Disagree) != wantDisagree {
		t.Fatalf("the README says %v rungs disagree with the final answer; the ladder has %d",
			wantDisagree, atOneSummary.Disagree)
	}
	if float64(atOneSummary.ConfidentWrong) != wantConfident {
		t.Fatalf("the README says %v rungs are confidently wrong (T=1.00); the ladder has %d",
			wantConfident, atOneSummary.ConfidentWrong)
	}
	if atOneSummary.PositionsWithConfidentWrong != atOneSummary.Positions {
		t.Fatalf("the README says every one of the %d positions has such a rung; only %d do",
			atOneSummary.Positions, atOneSummary.PositionsWithConfidentWrong)
	}
}

// TestReadmeWeightTyingFiguresMatchTheDecomposition guards the weight-tying section.
//
// "logit 25.770 = ‖E[b]‖ 9.973 × ‖x‖ 9.351 × cos 0.2763", the two ranges in the table, their
// correlations, and "563 / 32640（1.7%）".
func TestReadmeWeightTyingFiguresMatchTheDecomposition(t *testing.T) {
	doc := readmeCardSections(t)

	re := regexp.MustCompile(`logit ([0-9.]+) = ‖E\[b\]‖ ([0-9.]+) × ‖x‖ ([0-9.]+) × cos ([0-9.]+)`)
	m := re.FindStringSubmatch(doc)
	if m == nil {
		t.Fatal("the README no longer states the weight-tying decomposition in the form this " +
			"test knows; update the test with the sentence you wrote")
	}

	a := newTestApp(t)
	wt := computeWeightTying(a.W, a.Trace, a.Temp)
	if !wt.OK {
		t.Fatal("the weight-tying measurement failed on the bundled weights")
	}
	near(t, "weight tying, logit", m[1], wt.Logit)
	near(t, "weight tying, ‖E[b]‖", m[2], wt.EmbedNorm)
	near(t, "weight tying, ‖x‖", m[3], wt.FinalNorm)
	near(t, "weight tying, cosine", m[4], wt.Cosine)

	// The table's two rows.
	reNorm := regexp.MustCompile(`\| ‖E\[b\]‖ \| ([0-9.]+) – ([0-9.]+) \| \*\*([−-][0-9.]+)\*\*`)
	if mm := reNorm.FindStringSubmatch(doc); mm == nil {
		t.Fatal("the README no longer has the ‖E[b]‖ row with its range and correlation")
	} else {
		near(t, "weight tying, ‖E[b]‖ range low", mm[1], wt.EmbedNormMin)
		near(t, "weight tying, ‖E[b]‖ range high", mm[2], wt.EmbedNormMax)
		stated := strings.ReplaceAll(mm[3], "−", "-")
		near(t, "weight tying, ‖E[b]‖ correlation", strings.TrimPrefix(stated, "-"),
			absFloat(wt.LogitNormCorrelation))
	}

	reCos := regexp.MustCompile(`\| cos\(E\[b\], x\) \| ([−-][0-9.]+) – \+([0-9.]+) \| \*\*\+([0-9.]+)\*\*`)
	if mm := reCos.FindStringSubmatch(doc); mm == nil {
		t.Fatal("the README no longer has the cos(E[b], x) row with its range and correlation")
	} else {
		near(t, "weight tying, cosine range low", strings.TrimPrefix(
			strings.ReplaceAll(mm[1], "−", "-"), "-"), absFloat(wt.CosineMin))
		near(t, "weight tying, cosine range high", mm[2], wt.CosineMax)
		near(t, "weight tying, cosine correlation", mm[3], wt.LogitCosineCorrelation)
	}

	rePairs := regexp.MustCompile(`只有 \*\*(\d+) / (\d+)（([0-9.]+)%）\*\*`)
	if mm := rePairs.FindStringSubmatch(doc); mm == nil {
		t.Fatal("the README no longer states how many byte pairs rank differently")
	} else {
		if got := mustNumber(t, mm[1]); int(got) != wt.PairsOrderedDifferently {
			t.Fatalf("the README says %v pairs rank differently by logit than by cosine; the "+
				"code counts %d", got, wt.PairsOrderedDifferently)
		}
		if got := mustNumber(t, mm[2]); int(got) != wt.Pairs {
			t.Fatalf("the README says there are %v byte pairs; the code counts %d", got, wt.Pairs)
		}
		pct := 100 * float64(wt.PairsOrderedDifferently) / float64(wt.Pairs)
		near(t, "weight tying, share of differently-ranked pairs", mm[3], pct)
	}

	// The identity the card's whole argument rests on.
	product := wt.EmbedNorm * wt.FinalNorm * wt.Cosine
	if diff := absFloat(product - wt.Logit); diff > 1e-3 {
		t.Fatalf("‖E[b]‖·‖x‖·cos = %.6f but the logit is %.6f; the decomposition is not exact",
			product, wt.Logit)
	}
}

// absFloat is math.Abs without pulling the math import into every call site here.
func absFloat(v float64) float64 {
	if v < 0 {
		return -v
	}
	return v
}

// TestReadmeHeadIndependenceFiguresMatchTheLayer guards the head-independence table:
// "0 | +0.027 – +0.144 | 19% – 32%" and "1 | +0.098 – +0.537 | 21% – 28%".
func TestReadmeHeadIndependenceFiguresMatchTheLayer(t *testing.T) {
	doc := readmeCardSections(t)

	re := regexp.MustCompile(`\| (\d) \| \*?\*?\+([0-9.]+) – \+([0-9.]+)\*?\*? \| (\d+)% – (\d+)% \|`)
	rows := re.FindAllStringSubmatch(doc, -1)
	if len(rows) != 2 {
		t.Fatalf("the README's head-independence table has %d matching rows, want 2; update "+
			"this test with the table you wrote", len(rows))
	}

	a := newTestApp(t)
	for _, m := range rows {
		layer := int(mustNumber(t, m[1]))
		if layer < 0 || layer >= len(a.Trace.Layers) {
			t.Fatalf("the README tabulates layer %d, but the model has %d layers",
				layer, len(a.Trace.Layers))
		}
		hi := computeHeadIndependence(a.Trace.Layers[layer], len(a.Trace.Tokens))
		if !hi.OK {
			t.Fatalf("layer %d's head-independence measurement failed", layer)
		}
		near(t, fmt.Sprintf("layer %d, smallest pairwise cosine", layer), m[2], hi.MinPairCosine)
		near(t, fmt.Sprintf("layer %d, largest pairwise cosine", layer), m[3], hi.MaxPairCosine)
		near(t, fmt.Sprintf("layer %d, smallest head share", layer), m[4],
			100*hi.MinShare)
		near(t, fmt.Sprintf("layer %d, largest head share", layer), m[5],
			100*hi.MaxShare)
	}
}

// TestReadmeFFNAnswerDirectionFiguresMatchTheRanking guards the FFN table:
// "| 层 0 | **103.4** | **4.5** | 30 / 35 |" and "| 层 1 | 3.7 | 10.7 | 20 / 35 |".
func TestReadmeFFNAnswerDirectionFiguresMatchTheRanking(t *testing.T) {
	doc := readmeCardSections(t)

	re := regexp.MustCompile(`\| 层 (\d) \| \*?\*?([0-9.]+)\*?\*? \| \*?\*?([0-9.]+)\*?\*? \| (\d+) / (\d+) \|`)
	rows := re.FindAllStringSubmatch(doc, -1)
	if len(rows) < 2 {
		t.Fatalf("the README's FFN answer-direction table has %d rows, want at least 2; "+
			"update this test with the table you wrote", len(rows))
	}

	a := newTestApp(t)
	seen := map[int]bool{}
	for _, m := range rows {
		layer := int(mustNumber(t, m[1]))
		if seen[layer] {
			continue
		}
		seen[layer] = true
		if layer >= len(a.Trace.Layers) {
			t.Fatalf("the README tabulates layer %d, but the model has %d layers",
				layer, len(a.Trace.Layers))
		}
		d := computeFFNAnswerDirection(a.W, a.Trace, layer)
		if !d.OK {
			t.Fatalf("layer %d's answer-direction measurement failed", layer)
		}
		near(t, fmt.Sprintf("layer %d, rank of the answer in the input", layer), m[2],
			d.InputMeanRank)
		near(t, fmt.Sprintf("layer %d, rank after the FFN", layer), m[3], d.OutputMeanRank)
		if got := int(mustNumber(t, m[4])); got != d.OutputTop10 {
			t.Fatalf("layer %d: the README says %d positions put the answer in the top 10 "+
				"after the FFN; the code counts %d", layer, got, d.OutputTop10)
		}
		if got := int(mustNumber(t, m[5])); got != d.Positions {
			t.Fatalf("layer %d: the README says %d positions were scored; the code scores %d",
				layer, got, d.Positions)
		}
	}
}

// TestReadmeFFNConcentrationFiguresMatchTheUnits guards the unit-sharing table:
// "| 层 0 | **130 / 384** | **0** | 0 |" and "| 层 1 | **28 / 384** | **17** | 27 |".
func TestReadmeFFNConcentrationFiguresMatchTheUnits(t *testing.T) {
	doc := readmeCardSections(t)

	re := regexp.MustCompile(`\| 层 (\d) \| \*\*(\d+) / (\d+)\*\* \| \*\*(\d+)\*\* \| (\d+) \|`)
	rows := re.FindAllStringSubmatch(doc, -1)
	if len(rows) < 2 {
		t.Fatalf("the README's FFN unit table has %d rows, want at least 2; update this test "+
			"with the table you wrote", len(rows))
	}

	a := newTestApp(t)
	all := computeAllFFNConcentration(a.Trace)
	if len(all) != len(a.Trace.Layers) {
		t.Fatalf("measured %d layers, the trace has %d", len(all), len(a.Trace.Layers))
	}
	seen := map[int]bool{}
	for _, m := range rows {
		layer := int(mustNumber(t, m[1]))
		if seen[layer] {
			continue
		}
		seen[layer] = true
		if layer >= len(all) {
			t.Fatalf("the README tabulates layer %d, but only %d were measured",
				layer, len(all))
		}
		c := all[layer]
		if got := int(mustNumber(t, m[2])); got != c.HalfMass {
			t.Fatalf("layer %d: the README says %d units carry half the activation; the code "+
				"counts %d", layer, got, c.HalfMass)
		}
		if got := int(mustNumber(t, m[3])); got != c.Units {
			t.Fatalf("layer %d: the README says %d units; the model has %d", layer, got, c.Units)
		}
		if got := int(mustNumber(t, m[4])); got != c.AlwaysOn {
			t.Fatalf("layer %d: the README says %d units are on at every position; the code "+
				"counts %d", layer, got, c.AlwaysOn)
		}
		if got := int(mustNumber(t, m[5])); got != c.NeverOn {
			t.Fatalf("layer %d: the README says %d units never open; the code counts %d",
				layer, got, c.NeverOn)
		}
	}
}

// TestReadmeLayerRewriteFiguresMatchTheTrace guards the layer-rewrite table's first row — the
// default prompt, which is the one the app opens on.
//
// The other two rows are other prompts, so they are covered separately by
// TestReadmeLayerRewriteOtherPrompts below.
func TestReadmeLayerRewriteFiguresMatchTheTrace(t *testing.T) {
	doc := readmeCardSections(t)

	re := regexp.MustCompile(`\| ` + regexp.QuoteMeta("`The transformer reads the tokens and`") +
		` \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|`)
	m := re.FindStringSubmatch(doc)
	if m == nil {
		t.Fatal("the README no longer has the default prompt's row in the layer-rewrite table")
	}

	a := newTestApp(t)
	rewrites := computeLayerRewrite(a.Trace)
	if len(rewrites) != 2 {
		t.Fatalf("expected 2 layers, measured %d", len(rewrites))
	}
	for i, stated := range m[1:] {
		layer := i / 2
		field := "cosine in/out"
		if i%2 == 1 {
			field = "‖ΔFFN‖/‖x‖"
		}
		r := rewrites[layer]
		measured := r.CosineInOut
		if i%2 == 1 {
			measured = r.FFNRatio
		}
		near(t, fmt.Sprintf("layer %d, %s", layer, field), stated, measured)
	}
}

// TestReadmeLayerRewriteOtherPrompts checks the two non-default prompts in the same table.
//
// The README claims the pattern holds across prompts, which is the part that makes it a
// statement about the model rather than about one sentence. Checking only the default row
// would leave the other two unverified.
func TestReadmeLayerRewriteOtherPrompts(t *testing.T) {
	doc := readmeCardSections(t)

	const (
		attention = "attention is all you need"
		lets      = "Attention lets every token…"
	)
	re := regexp.MustCompile(`\| ` + regexp.QuoteMeta("`attention is all you need`") +
		` \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|`)
	m := re.FindStringSubmatch(doc)
	if m == nil {
		t.Fatal("the README no longer has the `attention is all you need` row")
	}
	if !strings.Contains(doc, "`"+lets+"`") {
		t.Fatal("the README no longer has the third prompt's row in the layer-rewrite table")
	}

	w := mustBundled(t)
	tr := model.Forward(w, promptIDs(attention))
	rewrites := computeLayerRewrite(tr)
	if len(rewrites) != 2 {
		t.Fatalf("expected 2 layers, measured %d", len(rewrites))
	}
	near(t, "attention prompt, layer 0 cosine in/out", m[1], rewrites[0].CosineInOut)
	near(t, "attention prompt, layer 0 ‖ΔFFN‖/‖x‖", m[2], rewrites[0].FFNRatio)
	near(t, "attention prompt, layer 1 cosine in/out", m[3], rewrites[1].CosineInOut)
	near(t, "attention prompt, layer 1 ‖ΔFFN‖/‖x‖", m[4], rewrites[1].FFNRatio)
}

// TestReadmePosEncodingFiguresMatchTheShape guards the positional-encoding table:
// "这个模型学出来的 | **+0.040** | **−0.052**", "经典固定正弦编码 | +0.969 | +0.562", and the
// 1/√d reference of 0.102.
func TestReadmePosEncodingFiguresMatchTheShape(t *testing.T) {
	doc := readmeCardSections(t)

	re := regexp.MustCompile(`\| 这个模型学出来的 \| \*\*\+([0-9.]+)\*\* \| \*\*−([0-9.]+)\*\* \|`)
	m := re.FindStringSubmatch(doc)
	if m == nil {
		t.Fatal("the README no longer has the learned-encoding row in its positional table")
	}
	reSin := regexp.MustCompile(`\| 经典固定正弦编码 \| \+([0-9.]+) \| \+([0-9.]+) \|`)
	ms := reSin.FindStringSubmatch(doc)
	if ms == nil {
		t.Fatal("the README no longer has the sinusoidal row in its positional table")
	}
	reRand := regexp.MustCompile(`\| (\d+) 维里两个随机向量 \| ≈([0-9.]+)（期望）`)
	mr := reRand.FindStringSubmatch(doc)
	if mr == nil {
		t.Fatal("the README no longer has the random-vector row in its positional table")
	}

	a := newTestApp(t)
	pe := computePosEncodingShape(a.W)
	if !pe.OK {
		t.Fatal("the positional-encoding measurement failed on the bundled weights")
	}
	near(t, "learned encoding, adjacent cosine", m[1], pe.AdjacentCos)
	near(t, "learned encoding, distant cosine", m[2], absFloat(pe.DistantCos))
	if pe.DistantCos > 0 {
		t.Fatalf("the README writes the distant cosine as negative (%s) but it is +%.4f",
			m[2], pe.DistantCos)
	}
	near(t, "sinusoidal encoding, adjacent cosine", ms[1], pe.SinusoidalAdjacentCos)
	near(t, "sinusoidal encoding, distant cosine", ms[2], pe.SinusoidalDistantCos)
	if got := int(mustNumber(t, mr[1])); got != pe.Dims {
		t.Fatalf("the README says %d dimensions; the model has %d", got, pe.Dims)
	}
	near(t, "random-vector reference cosine", mr[2], pe.RandomCos)

	// The paragraph's conclusion rests on the learned adjacent cosine being *below* the
	// random reference, not merely different from the sinusoidal one.
	if pe.AdjacentCos >= pe.RandomCos {
		t.Fatalf("the learned encoding's adjacent cosine is %.4f, not below the %.4f a random "+
			"pair would give; the README's \"几乎互不相干\" no longer holds", pe.AdjacentCos, pe.RandomCos)
	}
}

// TestReadmeEmbeddingFiguresMatchTheTable guards the embedding-structure table:
// "文本里相邻的 35 对 | **+0.0213**", "随便两组（32640 对） | +0.0033（标准差 0.1017）",
// and "元音之间 / 元音与辅音之间 | +0.0335 / 0.0000".
func TestReadmeEmbeddingFiguresMatchTheTable(t *testing.T) {
	doc := readmeCardSections(t)

	re := regexp.MustCompile(`\| 文本里相邻的 (\d+) 对 \| \*\*\+([0-9.]+)\*\* \|`)
	m := re.FindStringSubmatch(doc)
	if m == nil {
		t.Fatal("the README no longer has the adjacent-pairs row in its embedding table")
	}
	reRand := regexp.MustCompile(`\| 随便两组（(\d+) 对） \| \+([0-9.]+)（标准差 ([0-9.]+)） \|`)
	mr := reRand.FindStringSubmatch(doc)
	if mr == nil {
		t.Fatal("the README no longer has the random-pairs row in its embedding table")
	}
	reVowel := regexp.MustCompile(`\| 元音之间 / 元音与辅音之间 \| \+([0-9.]+) / (-?[0-9.]+) \|`)
	mv := reVowel.FindStringSubmatch(doc)
	if mv == nil {
		t.Fatal("the README no longer has the vowel row in its embedding table")
	}

	a := newTestApp(t)
	e := computeEmbeddingStructure(a.W, a.Prompt)
	if !e.OK {
		t.Fatal("the embedding-structure measurement failed on the bundled weights")
	}
	if got := int(mustNumber(t, m[1])); got != e.AdjacentPairs {
		t.Fatalf("the README says %d adjacent pairs; the code measures %d", got, e.AdjacentPairs)
	}
	near(t, "embedding, adjacent-pair cosine", m[2], e.AdjacentCos)
	if got := int(mustNumber(t, mr[1])); got != e.RandomPairs {
		t.Fatalf("the README says %d random pairs; the code measures %d", got, e.RandomPairs)
	}
	near(t, "embedding, random-pair cosine", mr[2], e.RandomCos)
	near(t, "embedding, random-pair spread", mr[3], e.SDRandom)
	near(t, "embedding, vowel cosine", mv[1], e.VowelCos)
	near(t, "embedding, vowel-consonant cosine", strings.TrimPrefix(mv[2], "-"),
		absFloat(e.ConsonantCos))

	// "最像的 20 对字节里只有 1 对两边都是可读字符", and the top pair is `I` with `\x9a`.
	reTop := regexp.MustCompile(`最像的 20 对字节里只有 (\d+) 对两边都是可读字符`)
	if mt := reTop.FindStringSubmatch(doc); mt == nil {
		t.Fatal("the README no longer states how many of the top 20 pairs are printable")
	} else if got := int(mustNumber(t, mt[1])); got != e.SimilarPrintableOf20 {
		t.Fatalf("the README says %d of the top 20 pairs are printable; the code counts %d",
			got, e.SimilarPrintableOf20)
	}
	if !strings.Contains(doc, "`I` 和一个控制字节 `\\x9a`") {
		t.Fatal("the README no longer names the most similar pair; update this test with the " +
			"pair it names now")
	}
	if len(e.TopPairs) == 0 {
		t.Fatal("the embedding measurement found no similar pairs to name")
	}
	top := e.TopPairs[0]
	if top.A != 'I' || top.B != 0x9a {
		t.Fatalf("the README says the most similar pair is I with \\x9a, but the measurement "+
			"says %d with %d", top.A, top.B)
	}

	// The conclusion: adjacent bytes are not meaningfully closer than random ones.
	if e.AdjacentCos > e.RandomCos+2*e.SDRandom {
		t.Fatalf("the adjacent-pair cosine %.4f is more than two standard deviations above "+
			"the random mean %.4f; the README's \"差别落在噪声里\" no longer holds",
			e.AdjacentCos, e.RandomCos)
	}
}

// TestReadmeMaskCounterfactualFiguresMatchTheHeads guards the mask paragraph:
// "每个头都会把 51%–84% 的注意力放到被掩码遮住的位置上" and row 0's 100% / k34 47%.
//
// The per-head figure is a mean over rows 0..T-2 (meanFutureShare skips the last row, which
// has no future to attend to). Measuring extremes over every cell instead gives 0–100% and
// would look like a contradiction; the note in the README is what tells a reader which one is
// meant, so the test uses the same helper the card does.
func TestReadmeMaskCounterfactualFiguresMatchTheHeads(t *testing.T) {
	doc := readmeCardSections(t)

	re := regexp.MustCompile(`每个头都会把 (\d+)%–(\d+)% 的注意力放到被掩码遮住的位置上`)
	m := re.FindStringSubmatch(doc)
	if m == nil {
		t.Fatal("the README no longer states the per-head masked share range in the form this " +
			"test knows")
	}

	a := newTestApp(t)
	n := len(a.Trace.Tokens)
	dh := a.W.Config.DHead()
	lo, hi := 100.0, 0.0
	for li := range a.Trace.Layers {
		for hi2 := range a.Trace.Layers[li].Heads {
			mean, rows := meanFutureShare(a.Trace.Layers[li].Heads[hi2], n, dh)
			if rows == 0 {
				t.Fatalf("L%dH%d: no rows measured", li, hi2)
			}
			if 100*mean < lo {
				lo = 100 * mean
			}
			if 100*mean > hi {
				hi = 100 * mean
			}
		}
	}
	// The README quotes whole percentages.
	if stated, want := int(mustNumber(t, m[1])), int(lo+0.5); stated != want {
		t.Fatalf("the README says the smallest per-head mean is %d%%, the code measures %d%%",
			stated, want)
	}
	if stated, want := int(mustNumber(t, m[2])), int(hi+0.5); stated != want {
		t.Fatalf("the README says the largest per-head mean is %d%%, the code measures %d%%",
			stated, want)
	}

	// Row 0: "去掉掩码后 100% 的注意力跑到后面，其中最大的一格是 k34 的 47%".
	reRow := regexp.MustCompile(`第 0 行最极端[^。]*去掉掩码后 \*\*(\d+)%\*\*[^。]*最大的一格是 k(\d+) 的 (\d+)%`)
	mr := reRow.FindStringSubmatch(doc)
	if mr == nil {
		t.Fatal("the README no longer describes row 0's unmasked distribution in the form this " +
			"test knows")
	}
	c := computeMaskCounterfactual(a.Trace.Layers[0].Heads[0], 0, n, dh)
	if !c.OK {
		t.Fatal("row 0's counterfactual failed to compute")
	}
	if got := int(mustNumber(t, mr[1])); int(100*c.FutureShare+0.5) != got {
		t.Fatalf("the README says row 0 sends %d%% of its unmasked attention to the future; "+
			"the code measures %.0f%%", got, 100*c.FutureShare)
	}
	if got := int(mustNumber(t, mr[2])); got != c.FutureTop {
		t.Fatalf("the README says row 0's largest future key is k%d; the code says k%d",
			got, c.FutureTop)
	}
	if got := int(mustNumber(t, mr[3])); int(100*c.FutureTopShare+0.5) != got {
		t.Fatalf("the README says k%d takes %d%% of row 0's unmasked attention; the code "+
			"measures %.0f%%", c.FutureTop, got, 100*c.FutureTopShare)
	}
}

// TestReadmeHeadProfileTableMatchesTheHeads guards the per-head table: "0 | 66% | 22% | 37% |
// 61% | 差别明显（跨度 44 个百分点）" and the second layer's row.
//
// The percentages are PrevShare — the mean weight each head puts on the token before it — and
// the parenthetical span is the difference between the row's largest and smallest.
func TestReadmeHeadProfileTableMatchesTheHeads(t *testing.T) {
	doc := readmeCardSections(t)

	re := regexp.MustCompile(`\| (\d) \| (\d+)% \| (\d+)% \| (\d+)% \| (\d+)% \| [^|]*跨度 (\d+) 个百分点`)
	rows := re.FindAllStringSubmatch(doc, -1)
	if len(rows) != 2 {
		t.Fatalf("the README's per-head table has %d matching rows, want 2; update this test "+
			"with the table you wrote", len(rows))
	}

	a := newTestApp(t)
	for _, m := range rows {
		layer := int(mustNumber(t, m[1]))
		if layer >= len(a.Trace.Layers) {
			t.Fatalf("the README tabulates layer %d, but the model has %d layers",
				layer, len(a.Trace.Layers))
		}
		profiles := computeHeadProfiles(a.Trace.Layers[layer], len(a.Trace.Tokens))
		if len(profiles) != 4 {
			t.Fatalf("layer %d has %d head profiles, want 4", layer, len(profiles))
		}
		lo, hi := 100.0, 0.0
		for h, p := range profiles {
			want := int(mustNumber(t, m[2+h]))
			// The app prints the share with %.0f, which rounds; compare the same way.
			got := int(100*p.PrevShare + 0.5)
			if got != want {
				t.Fatalf("layer %d head %d: the README says %d%%, the code measures %.2f%%",
					layer, h, want, 100*p.PrevShare)
			}
			if 100*p.PrevShare < lo {
				lo = 100 * p.PrevShare
			}
			if 100*p.PrevShare > hi {
				hi = 100 * p.PrevShare
			}
		}
		// The span is the difference of the unrounded shares, rounded once — the same order
		// of operations the card uses (`spread := hi - lo`, printed with %.0f). Rounding each
		// end first would give layer 1 a span of 2 instead of 3.
		span := int(hi - lo + 0.5) // hi and lo are already percentages
		if got := int(mustNumber(t, m[6])); got != span {
			t.Fatalf("layer %d: the README says the span is %d percentage points; the code "+
				"gives %d", layer, got, span)
		}
	}
}
