package app

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/egoist/mygo/ui"

	"llmlab/internal/model"
)

type lesson struct {
	name    string
	title   string
	body    string
	formula string
	// bodyFn, when set, replaces body at render time. A sentence that quotes a number
	// from the loaded weights, or that describes what those weights actually do, cannot
	// be a constant: change the checkpoint and the prose would keep claiming the old
	// figures — or, worse, describe behaviour the new weights do not exhibit.
	bodyFn func(*model.Trace) string
}

var lessons = []lesson{
	{
		name:  "分词",
		title: "把文字切成 token",
		body: "模型不认识字母，只认识整数。这里用最朴素的字节级分词：把提示词的 UTF-8 字节逐个映射成 0–255 的 id。" +
			"英文一个字符一个 token，一个汉字要占三个 token——右边显示的数字就是 id。" +
			"往上数还有：一个 emoji 通常四个字节；而由「男人 + 零宽连接符 + 女人 + 零宽连接符 + 女孩」拼成的家庭符号，" +
			"你看到的是一个图形，模型看到的是 18 个 token（三个 emoji 各四字节，两个连接符各三字节）。" +
			"真实的 LLM 用 BPE 把高频字节对合并成子词，词表通常三万到十五万条，道理相同：先离散化，再查表。",
		formula: "id_t = byte(prompt)[t]",
	},
	{
		name:    "嵌入",
		title:   "查表得到向量",
		body:    "嵌入表是一张 vocab × d_model 的矩阵，每一行就是一个 token 的「意思」。这张表不是人工写的，是训练出来的——和第 3 步那三个投影矩阵、第 5 步的 γ/β 一样，都是梯度下降调出来的数字。查表得到向量后，再加上位置向量：注意力的打分只看 q·k，本质上是对一组 key 加权求和，同一批词换个顺序，这组权重本身不会有任何不同——所以位置信息必须显式注入，否则“谁在前谁在后”就没有任何表示。因果掩码让每个位置只看得到自己及之前的内容，那是另一回事：它限制的是“能看到几个”，不是“这些词排在第几位”。",
		formula: "x_t = E[token_t] + P[t]",
	},
	{
		name:  "注意力分数",
		title: "Query 与 Key 的点积",
		body: "每个位置的向量被三个矩阵投影成 Query（我在找什么）、Key（我能提供什么）、Value（我携带的内容）——这三个矩阵和嵌入表一样，是训练出来的，不是人为设计的。第 t 个位置用它的 Query 去点乘前面每个位置的 Key，得到「相关度」。除以 √d_head 是为了让方差稳定，否则维度一高分数就会爆炸，softmax 直接饱和。注意右上三角是空的：因果掩码让模型看不到未来。" +
			"这个掩码是代码写死的，不是训练出来的——它属于架构，和那些学到的权重不是一回事。",
		formula: "score[t][s] = (q_t · k_s) / √d_head,  s ≤ t",
	},
	{
		name:  "Softmax",
		title: "变成权重再混合",
		bodyFn: func(tr *model.Trace) string {
			const base = "一行分数经过 softmax 变成概率分布，每一行的和恰好是 1。" +
				"用这些权重去加权求和 Value，就得到这个位置「从上下文里取回的信息」。"
			// Below three measurable rows nothing can be called a pattern, and "切换头上的分段控件
			// 就能看出来" needs at least two positions to differ. A one-token prompt has no row with a
			// predecessor at all, which used to print the fraction "0/0 行". A two-token prompt has
			// exactly one such row, which is not a sample.
			var prev, rows int
			if len(tr.Layers) > 0 && len(tr.Layers[0].Heads) > 0 {
				prev, rows = previousTokenShare(tr.Layers[0].Heads[0].Attn, len(tr.Tokens))
			}
			if rows < 3 {
				return base + "多个头并行做同一件事，各自学到的模式不同。" +
					fmt.Sprintf("这个提示词只有 %d 行可以比较（每个位置只看得到它自己和之前的 key），"+
						"样本太少，说不上有什么模式，也还看不出头与头的差别——多打几个字再看。", rows)
			}
			shared := "多个头并行做同一件事，但学到的模式并不一样，切换头上的分段控件就能看出来。"
			// The claim that follows holds for the trained weights — measured 31 of 35 rows — and
			// not for random ones, where the same count is 5 of 35. The app renders random weights
			// whenever the bundled file is missing, so the sentence has to follow the measurement.
			if prev*4 >= rows*3 {
				return base + shared + fmt.Sprintf("这一份权重里，第 0 层的头 0 在 %d/%d 行上把最大的权重"+
					"给了紧邻的前一个 token；第 1 层的几个头则反复回到同一小批位置上。"+
					"热力图里同一列发亮，就是这个头在「一直看着那里」。", prev, rows)
			}
			return base + shared + fmt.Sprintf("这一份权重里还看不出这种分工：第 0 层的头 0 只在 %d/%d 行上"+
				"把最大的权重给了紧邻的前一个 token，其余分散在各处。训练过的模型通常会出现"+
				"「盯着前一个 token」或「盯着一小批位置」这类模式，这里没有。", prev, rows)
		},
		formula: "α = softmax(score),  out_t = Σ_s α[t][s] · v_s",
	},
	{
		name:    "残差 / LN",
		title:   "残差与层归一化",
		body:    "子层的输出不会覆盖原向量，而是加回去——这就是残差连接，它让梯度有一条高速公路，也让「原始信息」一路保留到最后。进入子层之前先做 LayerNorm（pre-norm）：减均值、除标准差，再乘可学习的 γ、加 β。看直方图会发现归一化后分布被拉回零附近，这就是数值稳定的来源。一个块里有两次这样的加法：一次接注意力，一次接 FFN。",
		formula: "x ← x + Attn(LN₁(x));  x ← x + FFN(LN₂(x))",
	},
	{
		name:  "FFN",
		title: "逐位置的前馈网络",
		bodyFn: func(tr *model.Trace) string {
			c := tr.Config
			ofAll, _ := ffnParamShare(c)
			width := float64(c.DFF) / float64(c.DModel)
			return fmt.Sprintf(
				"注意力负责在位置之间搬运信息，FFN 负责在每个位置上单独做非线性变换："+
					"先升维到 %.0f 倍宽，过 GELU，再降回原维度。只看 transformer 块内部，"+
					"三分之二的参数都在这里（算上嵌入表则是 %.0f%%）。"+
					"GELU 是固定的数学函数，不含任何可学参数——被训练的是它前后的 W₁ 和 W₂。"+
					"它在负值区不是硬零，而是平滑地趋近零，所以激活会呈现稀疏但连续的分布。",
				width, ofAll*100)
		},
		formula: "FFN(x) = W₂ · GELU(W₁ · LN(x))",
	},
	{
		name:    "输出 / 采样",
		title:   "从向量到下一个字",
		body:    "最后一层的输出经过最终 LayerNorm，再与嵌入矩阵相乘（权重绑定），只把**最后一个位置**的向量投影到词表上（形状从 [T, 96] 变成 [1, 256]），得到下一个 token 的 logit——前面的位置也算得出来，但不参与这次预测。logit 不是概率，要除以温度再做 softmax。温度趋近 0 等于贪心，温度大于 1 会让分布变平、输出更随机。",
		formula: "p = softmax(logits / T)",
	},
}

func (a *App) stage(c *ui.Context) {
	tr := a.Trace
	if len(tr.Tokens) == 0 {
		a.emptyPromptCard(c)
		return
	}
	switch a.Step {
	case 0:
		a.stageTokens(c)
	case 1:
		a.stageEmbed(c)
	case 2:
		a.stageScores(c)
	case 3:
		a.stageSoftmax(c)
	case 4:
		a.stageResidual(c)
	case 5:
		a.stageFFN(c)
	case 6:
		a.stageOutput(c)
	}
}

// emptyPromptCard is the way out of the empty state. The presets used to live
// only in the tokenising stage's card, so a reader who emptied the prompt on
// step 5 was told "输入一些文字开始" with no visible way to do it: the presets
// were four steps away and the input field, while empty, gives no hint that a
// byte-level tokenizer will accept anything at all.
func (a *App) emptyPromptCard(c *ui.Context) {
	card(c, func() {
		sectionTitle(c, "还没有提示词", "整条流水线从这里开始")
		mutedText(c, "向上方的输入框随便打几个字——英文、中文、标点都行。"+
			"这是一个字节级分词器：每个 UTF-8 字节是一个 token，所以任何文字都能进来，"+
			"不需要先出现在词表里。")
		ui.Row(c).Gap(6).Wrap().AlignItems(ui.Center).Children(func() {
			ui.Text(c, "或者从一句现成的开始").FontSize(fontMicro).TextColor(colFaint).Shrink(0)
			for _, p := range promptPresets {
				// The empty preset is the state the reader is already in: a
				// chip for it would be a button that does nothing.
				if p.prompt == "" {
					continue
				}
				a.promptChip(c, p, false)
			}
		})
		ui.Row(c).Gap(6).AlignItems(ui.Center).Children(func() {
			ui.Text(c, "提示：").FontSize(fontMicro).TextColor(colFaint).Shrink(0)
			// The old wording was "输入后按任意阶段键（1–7）继续，顶部的 token 条会立刻出现",
			// which reads as though the view waits for a key. It does not: promptChanged
			// recomputes on every keystroke, so the strip appears as the reader types and
			// the stage keys only switch stage. Measured: typing "hi" from empty reaches two
			// tokens with no key pressed at all.
			ui.Text(c, "边打边重算，顶部的 token 条会立刻出现；1–7 或方向键可以切到别的阶段。").
				FontSize(fontMicro).TextColor(colFaint)
		})
	})
}

// promptChip draws one preset prompt. Both places that offer the presets — the
// tokenising card and the empty-prompt card — go through here: they had drifted
// into two different looks (10.5px/4-8-5 versus 11px/5-9-6) for the same
// control, which reads as two different things rather than one.
//
// showSelected marks the preset currently in the field; the empty-prompt card
// passes false because it only ever lists the presets the reader is not using.
func (a *App) promptChip(c *ui.Context, p promptPreset, showSelected bool) {
	chip := ui.Row(c).Padding(4, 9).Radius(6).AlignItems(ui.Center).
		Background(colPanel2).Border(1, colBorder).Children(func() {
		ui.Text(c, p.name).FontSize(fontSmall).TextColor(colMuted).NoWrap()
	})
	// Every other clickable chip in the app lights up under the pointer — the
	// token strip and the sidebar's stepper rows both go to colPanel2. These
	// presets never did, so the one control that fills the prompt field looked
	// less clickable than a token you can only select. The selected state takes
	// precedence, since it is the more specific fact.
	if showSelected && a.Prompt == p.prompt {
		chip = chip.Background(colAccent.Alpha(0.14)).Border(1, colAccent.Alpha(0.6))
	} else if chip.Hovered() {
		chip = chip.Background(colPanel).Border(1, colMuted.Alpha(0.5))
	}
	if chip.Clicked() {
		a.applyPrompt(p.prompt)
	}
	if p.prompt != "" {
		chip.Tooltip(truncateRunes(p.prompt, 48))
	}
}

// truncateRunes shortens s to at most n runes, for a tooltip about a prompt
// that is by definition longer than the chip that names it.
func truncateRunes(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

func (a *App) stageTokens(c *ui.Context) {
	tr := a.Trace
	// The chip grid that used to open this stage now lives in the strip above
	// every stage, so this card states the counts and leaves the reading of
	// the sequence itself to that strip.
	card(c, func() {
		sectionTitle(c, "提示词变成 token 序列", fmt.Sprintf("T = %d", len(tr.Tokens)))
		// The counts describe the sequence the model actually ran on, which is
		// not the whole prompt once truncation kicks in. Saying "字节数 63" for a
		// 270-byte prompt reads as a fact about the prompt, so a truncated run
		// states both numbers.
		dropped := len(a.Prompt) - len(tr.Tokens)
		ui.Row(c).Gap(16).Wrap().Children(func() {
			if dropped > 0 {
				ui.Text(c, fmt.Sprintf("字节数 %d（原文 %d）", len(tr.Tokens), len(a.Prompt))).
					FontSize(fontSmall).TextColor(colMuted)
				ui.Text(c, fmt.Sprintf("丢弃前 %d 字节", dropped)).FontSize(fontSmall).TextColor(colWarm)
			} else {
				ui.Text(c, fmt.Sprintf("字节数 %d", len(tr.Tokens))).FontSize(fontSmall).TextColor(colFaint)
			}
			ui.Text(c, fmt.Sprintf("字符数 %d", utf8.RuneCountInString(tr.Text))).FontSize(fontSmall).TextColor(colFaint)
			ui.Text(c, fmt.Sprintf("上下文上限 %d", tr.Config.Ctx)).FontSize(fontSmall).TextColor(colFaint)
			if extra := len(tr.Tokens) - utf8.RuneCountInString(tr.Text); extra > 0 {
				ui.Text(c, fmt.Sprintf("多字节字符多占 %d 个 token", extra)).FontSize(fontSmall).TextColor(colAccent2)
			}
		})
		mutedText(c, "窗口顶部的 token 条就是这条序列：每个格子是一个 token，上面的字是它的字面形态，"+
			"下面的数字是它的 id。点任意一格，右侧检查器会显示这个 token 在当前阶段的数值。")
		ui.Row(c).Gap(6).Wrap().AlignItems(ui.Center).Children(func() {
			ui.Text(c, "试试").FontSize(fontMicro).TextColor(colFaint).Shrink(0)
			for _, p := range promptPresets {
				a.promptChip(c, p, true)
			}
		})
	})

	card(c, func() {
		sectionTitle(c, "字符 → 字节", "为什么一个汉字要占三个 token")
		type group struct {
			r     rune
			bytes []byte
		}
		groups := make([]group, 0, 10)
		for i := 0; i < len(tr.Tokens) && len(groups) < 10; {
			start := tr.Tokens[i].Start
			g := group{r: tr.Tokens[i].Rune}
			for ; i < len(tr.Tokens) && tr.Tokens[i].Start == start; i++ {
				g.bytes = append(g.bytes, tr.Tokens[i].Byte)
			}
			groups = append(groups, g)
		}
		ui.Row(c).Wrap().Gap(10).Children(func() {
			for _, g := range groups {
				g := g
				label := fmt.Sprintf("%q", g.r)
				if g.r == 0 && len(g.bytes) > 0 {
					label = fmt.Sprintf("0x%02X", g.bytes[0])
				}
				ui.Column(c).Gap(3).Children(func() {
					ui.Text(c, label).FontSize(fontData).TextColor(colText)
					ui.Row(c).Gap(2).Children(func() {
						for _, b := range g.bytes {
							// colFaint on colPanel2 is 4.46:1, just under the
							// 4.5 this 10px digit needs. colMuted clears 6.5:1
							// and still reads as a caption beside the rune.
							ui.Text(c, fmt.Sprintf("%d", b)).FontSize(fontMicro).TextColor(colMuted).
								Padding(2, 3).Radius(3).Background(colPanel2)
						}
					})
				})
			}
		})
		// Work through the selected token's own character, so the claim "one
		// hanzi is three tokens" stops being a fact to accept and becomes a
		// conversion to check: the bytes are shown with their UTF-8 prefix
		// bits, and the payload bits are added back into the codepoint.
		if i := clampIndex(a.SelToken, len(tr.Tokens)); i >= 0 {
			if rb := runeAt(tr, tr.Tokens[i].Start); rb.OK {
				var parts []string
				for _, b := range rb.Bytes {
					parts = append(parts, fmt.Sprintf("%s %08b", fmt.Sprintf("%02X", b), b))
				}
				if rb.Rune == 0 && len(rb.Bytes) == 1 {
					// A byte that is not valid UTF-8. Saying it "decodes to
					// U+FFFD" would describe the display, not the input: the
					// byte is what the model sees, and the tokenizer accepted it
					// while the text view had to replace it to stay printable.
					workingsLines(c, []string{
						fmt.Sprintf("0x%02X %08b：不是合法的 UTF-8 字节，单独作为一个 token", rb.Bytes[0], rb.Bytes[0]),
						fmt.Sprintf("文本视图里显示为 �，但模型看到的 id 就是 %d", rb.Bytes[0]),
					})
				} else {
					workingsLines(c, []string{
						fmt.Sprintf("%q = U+%04X，占 %d 个字节：%s",
							rb.Rune, rb.Codepoint, len(rb.Bytes), strings.Join(parts, "  ")),
						fmt.Sprintf("还原码点：%s → U+%04X", utf8PayloadChain(rb.Bytes), rb.Codepoint),
					})
				}
			}
		}
	})

	// The belief is the one a reader arrives with from other tokenisers, and the correction cites
	// this prompt's own counts so it can be checked against the strip above rather than believed.
	if bytes, runes := len(tr.Text), utf8.RuneCountInString(tr.Text); bytes > runes {
		misconception(c,
			"「一个汉字就是一个 token」",
			fmt.Sprintf("这里 %d 个字符编成了 %d 个 token：一个汉字在 UTF-8 里占三个字节，"+
				"字节级分词就把这三个字节各算一个 token。往上看 token 条，每个汉字下面都跟三个格子。",
				runes, len(tr.Tokens)))
	}

	card(c, func() {
		sectionTitle(c, "数据流与形状", "点击任意阶段跳转")
		T, d := len(tr.Tokens), tr.Config.DModel
		rows := []struct {
			step        int
			name, shape string
		}{
			{0, "分词", fmt.Sprintf("[%d]", T)},
			{1, "嵌入", fmt.Sprintf("[%d, %d]", T, d)},
			{2, "注意力", fmt.Sprintf("[%d, %d]", T, d)},
			{3, "Softmax", fmt.Sprintf("[%d, %d]", T, T)},
			{4, "残差 / LN", fmt.Sprintf("[%d, %d]", T, d)},
			{5, "FFN", fmt.Sprintf("[%d, %d]", T, tr.Config.DFF)},
			// The first six shapes keep the token axis; this one drops it. Writing [256]
			// beside six [36, …] entries leaves a reader asking where the 36 went — the
			// answer is that only the last position is projected, so the axis becomes 1.
			{6, "输出", fmt.Sprintf("[1, %d]", tr.Config.Vocab)},
		}
		ui.Row(c).Wrap().Gap(6).AlignItems(ui.Center).Children(func() {
			for i, r := range rows {
				i, r := i, r
				chip := ui.Column(c).Gap(1).Padding(5, 9).Radius(6).
					Background(colPanel2).Border(1, colBorder).Children(func() {
					ui.Text(c, r.name).FontSize(fontMicro).TextColor(colMuted).NoWrap()
					ui.Text(c, r.shape).FontSize(fontMicro).TextColor(colAccent2).NoWrap()
				})
				if r.step == a.Step {
					chip = chip.Background(colAccent.Alpha(0.12)).Border(1, colAccent.Alpha(0.6))
				}
				if chip.Clicked() {
					a.selectStep(r.step)
				}
				if i < len(rows)-1 {
					ui.Text(c, "→").FontSize(fontSmall).TextColor(colFaint)
				}
			}
		})
	})
}

func (a *App) stageEmbed(c *ui.Context) {
	tr := a.Trace
	i := a.SelToken
	if i < 0 {
		i = 0
	}
	card(c, func() {
		sectionTitle(c, fmt.Sprintf("token %d「%s」的输入向量", i, tr.Tokens[i].Text), fmt.Sprintf("d_model = %d", tr.Config.DModel))
		ui.Text(c, "token embedding").FontSize(fontMicro).TextColor(colFaint)
		vectorStrip(c, tr.Embed[i], 20, &a.HoverDimIn)
		ui.Text(c, "position embedding").FontSize(fontMicro).TextColor(colFaint)
		vectorStrip(c, tr.Pos[i], 20, &a.HoverDimIn)
		ui.Text(c, "x = embedding + position").FontSize(fontMicro).TextColor(colAccent2)
		vectorStrip(c, tr.Input[i], 24, &a.HoverDimIn)
		// One dimension of that addition, with this token's numbers. Hovering a
		// strip above changes which dimension. The sum is exact in float32, so a
		// reader adding the two printed values gets the third one back.
		dim := 0
		if a.HoverDimIn >= 0 && a.HoverDimIn < len(tr.Embed[i]) {
			dim = a.HoverDimIn
		}
		e, pos, x := tr.Embed[i][dim], tr.Pos[i][dim], tr.Input[i][dim]
		// Both terms are parenthesised, not just the negative one. Round 167 fixed
		// "+0.5674 + -0.6724"; the positive case has the same problem the other way round,
		// printing "+0.1247 + +0.4062" — two plus signs in a row that read as a typo. With
		// both bracketed the two lines share one shape:
		//   (+0.1247) + (+0.4062) = +0.5309
		//   (+0.5674) + (-0.6724) = -0.1050
		workingsLines(c, []string{
			fmt.Sprintf("d%d：E[%d][%d] = %+.4f · P[%d] = %+.4f",
				dim, tr.Tokens[i].ID, dim, e, i, pos),
			fmt.Sprintf("%s + %s = %+.4f", signed(e), signed(pos), x),
		})
		// The two terms are added, so the obvious question is how their sizes
		// compare: if the position vector were much larger it would swamp the
		// word, and much smaller would leave order barely encoded. Measured on
		// this model they come out almost equal, which is a fact about this
		// model — a reader should be able to see it rather than assume.
		eNorm, pNorm := norm(tr.Embed[i]), norm(tr.Pos[i])
		if pNorm > 0 {
			kv(c, "两项的量级", fmt.Sprintf("‖E‖ %.2f · ‖P‖ %.2f · P/E = %.2f",
				eNorm, pNorm, pNorm/eNorm))
		}
	})

	card(c, func() {
		// The heading and the note both described four rows whatever the prompt was. With a
		// single-token prompt the card showed one row under "随位置变化" and a note promising
		// four positions — a comparison with nothing to compare and a count that was not
		// there. The table has one row per position the prompt uses, so say how many.
		shown := len(tr.Pos)
		if shown > 4 {
			shown = 4
		}
		switch {
		case shown <= 1:
			sectionTitle(c, "位置向量是每个位置一行",
				fmt.Sprintf("这个提示词只有 %d 个位置", shown))
		case shown < 4:
			sectionTitle(c, "位置向量随位置变化",
				fmt.Sprintf("这个提示词只有 %d 个位置，都列在这里", shown))
		default:
			sectionTitle(c, "位置向量随位置变化", "取前四个位置的 P[t]")
		}
		for t := 0; t < 4 && t < len(tr.Pos); t++ {
			ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
				ui.Text(c, fmt.Sprintf("t=%d", t)).FontSize(fontMicro).TextColor(colFaint).Width(26)
				ui.Box(c).Grow(1).Children(func() {
					vectorStrip(c, tr.Pos[t], 14, &a.HoverDimIn)
				})
			})
		}
	})
}

// signed formats v with its sign, wrapping a negative in parentheses so a sum
// reads as "+0.5674 + (-0.6724)" rather than the easily misread "+0.5674 + -0.6724".
// signed formats one addend of the embedding sum, always in parentheses.
//
// The sign is part of the number, so printing it bare puts two signs next to the
// operator: "+0.5674 + -0.6724" on one line, "+0.1247 + +0.4062" on the other. Both read
// as typos. Bracketing every addend, negative or not, gives the line one shape whatever
// the signs turn out to be.
func signed(v float32) string {
	return fmt.Sprintf("(%+.4f)", v)
}

// kvScale states the 1/sqrt(d_head) scaling, which both the score stage's legend and
// its inspector quote. It was the same line of code in two places, so the two panels
// could have drifted to different precision — the workings line already computes the
// same quantity as float32, and the three copies had two different types.
func kvScale(c *ui.Context, dHead int) {
	kv(c, "缩放", fmt.Sprintf("1/√%d = %.3f", dHead, 1/math.Sqrt(float64(dHead))))
}

func (a *App) stageScores(c *ui.Context) {
	tr := a.Trace
	l := a.layer()
	i := clampIndex(a.SelToken, len(tr.Tokens))
	h := l.Heads[a.Head]
	card(c, func() {
		// The Key and Value strips follow the cell under the cursor, not the selected
		// token. Walking the matrix to (q, k) makes the workings line below print
		// q_q · k_k while the selection stays on the row, so drawing K[q] beside it
		// showed a different vector from the one the arithmetic used — measured, the
		// two had opposite signs in the first three dimensions.
		key := i
		if q, k := a.cellCursor(); q >= 0 && k >= 0 && q == i && k < len(h.K) {
			key = k
		}
		sectionTitle(c, fmt.Sprintf("token %d 的 Q / K / V", i), fmt.Sprintf("头 %d · d_head = %d", a.Head, tr.Config.DHead()))
		// Where these three vectors come from. The stage body says it, but that card
		// sits below the matrix; a reader looking at three strips of numbers should not
		// have to scroll past the heatmap to learn they are projections of x.
		ui.Text(c, "q = x·W_Q，k = x·W_K，v = x·W_V（同一个 x，三个不同的投影矩阵）").
			FontSize(fontMicro).TextColor(colFaint)
		ui.Text(c, "Query").FontSize(fontMicro).TextColor(colFaint)
		vectorStrip(c, h.Q[i], 18, &a.HoverDimQ)
		keyLabel := "Key（同位置）"
		valLabel := "Value（同位置）"
		if key != i {
			// Name the position the way every other label in the app does — "key 4",
			// zero-based, matching the matrix's column header — rather than "第 4 个",
			// which reads as one-based and would send the reader to the wrong column.
			keyLabel = fmt.Sprintf("Key（位置 %d）", key)
			valLabel = fmt.Sprintf("Value（位置 %d）", key)
		}
		ui.Text(c, keyLabel).FontSize(fontMicro).TextColor(colFaint)
		vectorStrip(c, h.K[key], 18, &a.HoverDimK)
		ui.Text(c, valLabel).FontSize(fontMicro).TextColor(colFaint)
		vectorStrip(c, h.V[key], 18, &a.HoverDimV)
	})

	card(c, func() {
		// Name the layer in the heading, not only on the control at the top of
		// the window: once the page is scrolled the control is out of sight, and
		// every number below belongs to one specific layer.
		sectionTitle(c, "分数矩阵", fmt.Sprintf("层 %d · 行 = query，列 = key，右上三角被掩码", a.Layer))
		ui.Row(c).Gap(14).AlignItems(ui.Start).Children(func() {
			ui.Column(c).Gap(4).Children(func() {
				a.heatmapAxed(c, h.Scores, "score", true)
			})
			ui.Column(c).Gap(6).Grow(1).Children(func() {
				mutedText(c, "点积越大，说明这个位置的 Query 与那个位置的 Key 越匹配。除法因子让分数的量级不随维度增长。")
				// The old legend said "negative: suppresses attention", which
				// is false here: sign and relative weight disagree in 27% of
				// this model's cells, and 212 negative scores still land in
				// their row's top 3. Sign is only a convenient colour split;
				// what decides attention is the ranking within a row.
				// List only the sign the matrix actually contains. A legend swatch for
				// warm cells beside a matrix with none of them advertises a category
				// the reader cannot find — measured on a two-token prompt, where every
				// score is negative and the whole grid is blue.
				hasPos, hasNeg := false, false
				for q := range h.Scores {
					for k := 0; k <= q && k < len(h.Scores[q]); k++ {
						if h.Scores[q][k] > -1e30 {
							if h.Scores[q][k] >= 0 {
								hasPos = true
							} else {
								hasNeg = true
							}
						}
					}
				}
				ui.Row(c).Gap(12).Children(func() {
					if hasPos {
						ui.Row(c).Gap(5).AlignItems(ui.Center).Children(func() {
							ui.Box(c).Size(9, 9).Radius(2).Background(colWarm)
							ui.Text(c, "正分（暖色）").FontSize(fontMicro).TextColor(colMuted)
						})
					}
					if hasNeg {
						ui.Row(c).Gap(5).AlignItems(ui.Center).Children(func() {
							ui.Box(c).Size(9, 9).Radius(2).Background(colNeg)
							ui.Text(c, "负分（蓝色）").FontSize(fontMicro).TextColor(colMuted)
						})
					}
				})
				if !hasPos {
					// The misconception below says sign does not matter, only the
					// ranking within a row. Saying so here explains why a wholly
					// negative matrix is still a perfectly ordinary one.
					mutedText(c, "这一头在这个提示词下所有分数都是负的——分数可以整体平移，"+
						"决定注意力的是同一行里的大小顺序，不是正负。")
				}
				kvScale(c, tr.Config.DHead())
				kv(c, "可见", fmt.Sprintf("%d 个 key", i+1))
				// The colour ramp is |score| / saturation, so its top is a fact
				// about this matrix: state it, since it moves with the prompt and
				// with the layer — measured 4.6 to 12.5 on layer 0 and 8.1 to 15.2
				// on layer 1 across the bundled examples — and nothing else on the
				// stage implies it.
				if sc := scoreColorScale(h); sc > 0 {
					kv(c, "色深上限", fmt.Sprintf("|分数| = %.2f", sc))
					mutedText(c, "颜色深浅按 |分数| ÷ 上限 取值，所以最亮的那格就是这个上限。")
				}
			})
		})
		// One cell of the matrix, worked out: which q·k products make it up, and
		// the scaling that follows. The cell is whatever the reader is pointing
		// at or has pinned, so it follows the same cursor as the inspector.
		cq, ck := a.cellCursor()
		if cq >= 0 && ck >= 0 && cq < len(tr.Tokens) && ck < len(tr.Tokens) && ck <= cq {
			if p := computeScore(h.Q[cq], h.K[ck], tr.Config.DHead(), h.Scores[cq][ck]); p.OK {
				scoreWorkings(c, cq, ck, tr.Config.DHead(), p)
			}
		} else {
			mutedText(c, "把指针移到矩阵的某个格子上（或点一下固定它），这里会写出这一格的分数是怎么算出来的。")
		}
		if i == 0 {
			// Row 0 holds a single score, so "compare the scores inside this row" has nothing to
			// compare — the callout claimed "把这一行 1 个分数统统减去 5", which is arithmetic on a
			// set of one. Only the mask half of the point applies here.
			misconception(c,
				"「负分就是把这个 key 屏蔽掉了」",
				"第 0 行只有一个分数，行内比较还谈不上。真正让 key 彻底出局的只有右上角的因果掩码："+
					"那里是 -Inf，softmax 之后才是 0。选中后面的 token 就能看到负分与权重并不对应。")
		} else {
			misconception(c,
				"「负分就是把这个 key 屏蔽掉了」",
				fmt.Sprintf("分数的绝对值不重要，只有同一行内的大小关系重要。把这一行 %d 个分数"+
					"统统减去 5，哪怕全部变成负数，softmax 之后的权重一模一样。"+
					"真正让 key 彻底出局的只有右上角的因果掩码，那里是 -Inf，softmax 之后才是 0。",
					i+1))
		}
	})
}

// utf8PayloadChain reassembles the codepoint from the bytes the way a reader
// would by hand: strip each byte's length prefix, then shift the remaining bits
// into place. Written as (payload<<shift) added together, because the bit-twiddling
// form "E6&0F<<12|B3&3F<<6|A8&3F" reads as arithmetic noise.
func utf8PayloadChain(bs []byte) string {
	if len(bs) == 0 {
		return ""
	}
	shifts := map[int][]int{1: {0}, 2: {6, 0}, 3: {12, 6, 0}, 4: {18, 12, 6, 0}}
	w, ok := shifts[len(bs)]
	if !ok {
		return ""
	}
	parts := make([]string, 0, len(bs))
	for i, b := range bs {
		if w[i] == 0 {
			parts = append(parts, fmt.Sprintf("%02X", utf8Payload(b)))
			continue
		}
		parts = append(parts, fmt.Sprintf("(%02X<<%d)", utf8Payload(b), w[i]))
	}
	return strings.Join(parts, " + ")
}

// scoreWorkings prints how one score is computed: the largest contributions to
// q·k, the exact remainder that closes the dot product, and the division by
// √d_head. The dot product is accumulated in float32 like model.dot, so the
// number on the last line is the one the heatmap colours, not a near match.
func scoreWorkings(c *ui.Context, q, k, dHead int, p scoreParts) {
	var terms []string
	for _, t := range p.Top {
		terms = append(terms, fmt.Sprintf("d%d %+.3f", t.D, t.V))
	}
	// The scale is shown at the precision the model multiplies with: float32. A
	// float64 1/sqrt(d) prints as 0.204124145, but multiplying the printed dot by
	// it gives 0.932751018 while the printed score is 0.932751060 — a reader
	// checking the line by hand would find the last digits do not close.
	scale := float64(float32(1 / math.Sqrt(float64(dHead))))
	workingsLines(c, []string{
		fmt.Sprintf("q%d · k%d 的三项最大贡献：%s", q, k, strings.Join(terms, "  ")),
		fmt.Sprintf("+ 其余 %d 维 %+.4f = 点积 %.4f", dHead-len(p.Top), p.Rest, p.Dot),
		fmt.Sprintf("score = 点积 / √%d = %.4f × %.4f = %+.4f", dHead, p.Dot, scale, p.Score),
	})
}

func (a *App) stageSoftmax(c *ui.Context) {
	tr := a.Trace
	l := a.layer()
	i := clampIndex(a.SelToken, len(tr.Tokens))
	h := l.Heads[a.Head]

	card(c, func() {
		sectionTitle(c, "注意力矩阵", fmt.Sprintf("层 %d · 头 %d", a.Layer, a.Head))
		ui.Row(c).Gap(14).AlignItems(ui.Start).Children(func() {
			ui.Column(c).Gap(4).Children(func() {
				a.heatmapAxed(c, h.Attn, "attn", true)
			})
			ui.Column(c).Gap(6).Grow(1).MaxWidth(480).Children(func() {
				mutedText(c, "每一行加起来等于 1。颜色越深，这个 query 从那个 key 取走的信息越多。")
				rows := make([]barRow, 0, 8)
				var max float32
				for j := 0; j <= i; j++ {
					if h.Attn[i][j] > max {
						max = h.Attn[i][j]
					}
				}
				for j := 0; j <= i; j++ {
					rows = append(rows, barRow{
						Label:      fmt.Sprintf("%d %s", j, tr.Tokens[j].Text),
						Value:      h.Attn[i][j],
						ValueText:  fmt.Sprintf("%.3f", h.Attn[i][j]),
						Color:      colAccent,
						LabelWidth: 62,
						ValueWidth: 44,
						Index:      j,
					})
				}
				if len(rows) > 10 {
					rows = topBars(rows, 10)
				}
				bars(c, rows, max, colAccent)
			})
		})
		// The heatmap says "rows sum to 1", and the natural misreading is that
		// the whole matrix is one distribution, so the columns must sum to 1
		// too. They do not: a column is "how much everyone attends to me", and
		// nothing constrains that. The asymmetry callout this replaced was
		// mostly the causal mask talking about itself, which teaches less.
		colMax, colMin := columnSumRange(h.Attn)
		misconception(c,
			"「每行和为 1，那每列应该也是 1」",
			fmt.Sprintf("只有行有这个约束。这一层的列和从 %.2f 到 %.2f，"+
				"总共是 %.0f（等于 token 数，因为每行各贡献 1）。"+
				"行是「我在看谁」，天然是个分布；列是「谁在看我」，可以很高也可以接近 0——"+
				"矩阵右上角那些列的列和就是 0，纯粹是因果掩码。",
				colMin, colMax, float64(len(h.Attn))))
	})

	card(c, func() {
		sectionTitle(c, "加权求和", "本头的输出，以及它在残差里的份额")
		// Three strips, two different widths: the first is d_head, the other two are d_model,
		// and they are all drawn across the same space. Without a word about that a reader
		// compares them column by column, where cell 3 of the first sits over cell 12 of the
		// second.
		ui.Text(c, fmt.Sprintf("下面三条依次是：本头输出（%d 维）→ 投影回残差（%d 维）→ 四个头相加（%d 维）",
			tr.Config.DHead(), tr.Config.DModel, tr.Config.DModel)).
			FontSize(fontMicro).TextColor(colFaint)
		ui.Text(c, fmt.Sprintf("Σ α·v（头内输出，d_head = %d）", tr.Config.DHead())).FontSize(fontMicro).TextColor(colFaint)
		vectorStrip(c, h.Out[i], 18, &a.HoverDimHeadOut)
		// Head h only multiplies the rows of Wo that belong to its own slice,
		// so this is the head's actual contribution, not the sum of all heads.
		ui.Text(c, "· Wo[h] 切片（本头贡献 Δ_h，d_model 宽）").FontSize(fontMicro).TextColor(colFaint)
		vectorStrip(c, l.HeadProj[a.Head][i], 18, &a.HoverDimProj)
		ui.Text(c, "全部头相加 = 本层注意力输出").FontSize(fontMicro).TextColor(colAccent2)
		vectorStrip(c, l.AttnProj[i], 18, &a.HoverDimProj)
		// How α turns into the head output, one dimension at a time. Without
		// this the stage shows a weight and a resulting vector with nothing
		// connecting them.
		dim := 0
		if a.HoverDimHeadOut >= 0 && a.HoverDimHeadOut < tr.Config.DHead() {
			dim = a.HoverDimHeadOut
		}
		if p := computeWeighted(h.Attn[i], h.V, i, dim, len(tr.Tokens)); p.OK {
			weightedWorkings(c, i, dim, p)
		}
		// What each head writes into the residual, and the fact that those
		// writes partly cancel. Without this the card shows one head's
		// projection and the merged vector with no sense of scale between them.
		if hc := computeHeadContributions(*l, i); hc.OK {
			ui.Text(c, "各头写进残差流的长度 ‖HeadProj[h]‖").FontSize(fontMicro).TextColor(colFaint)
			rows := make([]barRow, 0, len(hc.Heads))
			for _, hh := range hc.Heads {
				label := fmt.Sprintf("H%d", hh.Head)
				colour := colGreen
				if hh.Head == a.Head {
					label = fmt.Sprintf("H%d ◂", hh.Head)
					colour = colAccent2
				}
				rows = append(rows, barRow{
					Label:      label,
					Value:      float32(hh.Norm),
					ValueText:  fmt.Sprintf("%.2f", hh.Norm),
					Color:      colour,
					LabelWidth: 40,
					ValueWidth: 40,
					Tip:        fmt.Sprintf("头 %d 写入残差的 L2 长度 %.3f（占四头之和 %.0f%%）", hh.Head, hh.Norm, hh.Share),
				})
			}
			bars(c, rows, float32(hc.SumNorm), colGreen)
			st := a.cancellation()
			misconception(c,
				"「四个头各写一块，加起来就是它们长度的和」",
				fmt.Sprintf("范数不能直接相加。四个头的长度之和是 %.2f，但它们相加后的长度只有 %.2f，"+
					"比值 %.2f——也就是说四个向量只有一部分指向同一方向，其余互相抵消了。"+
					"这不是这一行碰巧如此：本模型 %d 行里 %d 行都是这样（比值 %.2f–%.2f）。"+
					"所以「注意力输出 = 各头之和」是一次向量加法，不是把四份贡献堆起来。",
					hc.SumNorm, hc.TotalNorm, hc.Cancellation,
					st.Rows, st.Below, st.Lo, st.Hi))
		}
	})
}

func (a *App) stageResidual(c *ui.Context) {
	tr := a.Trace
	l := a.layer()
	i := clampIndex(a.SelToken, len(tr.Tokens))

	card(c, func() {
		sectionTitle(c, "残差流", fmt.Sprintf("层 %d · 两次加法，信息都绕过了子层", a.Layer))
		// The three strips below are the residual chain, and it is easy to label
		// them as if the sublayers consumed x directly. They do not: each sublayer
		// reads a normalised copy (pre-norm), which is what makes the residual
		// stream itself stay unnormalised and grow. Naming the normalised row
		// makes that visible at the point where the reader is looking at both.
		ui.Text(c, "残差流 x（进入本层，未归一化）").FontSize(fontMicro).TextColor(colFaint)
		vectorStrip(c, l.In[i], 16, &a.HoverDimIn)
		// LN(x) is what the normalisation produced, so it takes colAccent2 — the palette's
		// "what came out of the step". It was colWarm, which is a data colour for a scored
		// matrix: on a text label it reads as a warning and, at 11.09:1, as the most
		// important line on a card where it is the middle quantity.
		ui.Text(c, "LN(x) — 子层实际读到的输入").FontSize(fontMicro).TextColor(colAccent2)
		vectorStrip(c, l.Norm1[i], 16, &a.HoverDimIn)
		ui.Text(c, "注意力子层输出 Δ_attn").FontSize(fontMicro).TextColor(colFaint)
		vectorStrip(c, l.AttnProj[i], 16, &a.HoverDimIn)
		ui.Text(c, "x + Δ_attn").FontSize(fontMicro).TextColor(colAccent2)
		vectorStrip(c, l.AfterAttn[i], 20, &a.HoverDimIn)
		ui.Text(c, "FFN 子层输出 Δ_ffn").FontSize(fontMicro).TextColor(colFaint)
		vectorStrip(c, l.FFNOut[i], 16, &a.HoverDimIn)
		ui.Text(c, "x + Δ_attn + Δ_ffn（送往下一层）").FontSize(fontMicro).TextColor(colAccent2)
		vectorStrip(c, l.AfterFFN[i], 20, &a.HoverDimIn)
		// One dimension of the chain, so "the stream accumulates" stops being a
		// metaphor: each line is one addInto, and the running total is the
		// vector the strip above draws. Hovering any strip picks the dimension.
		dim := 0
		if a.HoverDimIn >= 0 && a.HoverDimIn < len(l.In[i]) {
			dim = a.HoverDimIn
		}
		if steps, ok := residualChain(*l, i, dim); ok {
			residualWorkings(c, dim, steps, l == &tr.Layers[len(tr.Layers)-1], a.Layer)
		}
	})

	card(c, func() {
		before := l.AfterAttn[i]
		after := l.Norm2[i]
		mb, sb := meanStd(before)
		ma, sa := meanStd(after)
		sectionTitle(c, "LayerNorm", fmt.Sprintf("μ %.3f → %.3f · σ %.3f → %.3f", mb, ma, sb, sa))
		// Both charts share one axis. Scaled separately they looked alike even
		// when LayerNorm had narrowed the row by half, and a reader comparing
		// the two pictures — which is the whole point of stacking them — was
		// comparing two different x-ranges.
		axLo, axHi := minMaxFloat(before)
		aLo, aHi := minMaxFloat(after)
		if aLo < axLo {
			axLo = aLo
		}
		if aHi > axHi {
			axHi = aHi
		}
		ui.Text(c, "归一化前").FontSize(fontMicro).TextColor(colFaint)
		histogramRange(c, before, 32, colWarm, axLo, axHi)
		ui.Text(c, "归一化后").FontSize(fontMicro).TextColor(colAccent2)
		histogramRange(c, after, 32, colAccent2, axLo, axHi)
		// "γ 和 β 是训练出来的，不是 1 和 0" is true of the bundled weights and
		// false of the random fallback, where they are exactly the identity. The
		// sentence is chosen from the weights rather than asserted.
		lnNote := "这里的 γ、β 是训练出来的，不是 1 和 0——下面那行算式里的数值就是它们在这个维度上的真实取值。"
		if !lnParamsAreLearned(a.W) {
			lnNote = "这份权重没有训练过，γ 恰好是 1、β 恰好是 0，所以这一步的缩放其实没起作用——训练之后它们才会离开 1 和 0。"
		}
		mutedText(c, "LayerNorm 先减均值、除标准差，再乘可学习的 γ、加 β；直方图展示最终的 γ/β 输出，所以方差不必恰好为 1。"+lnNote)
		// The same treatment the attention stage's cell card got: instead of
		// showing a number and leaving it to be trusted, print the arithmetic
		// that produced one entry of this row. γ and β are shown because they
		// are learned and not 1/0 here, which the previous text implied.
		dim := 0
		if a.HoverDimIn >= 0 && a.HoverDimIn < len(before) {
			dim = a.HoverDimIn
		}
		w := a.W.Layers[a.Layer]
		if p := computeLN(before, w.LN2G, w.LN2B, dim); p.ok {
			lnWorkings(c, dim, p)
		}
		// The two numbers that make the point: γ/β leave the variance a little
		// off 1, but the LENGTH of a normalized row is pinned near √d_model no
		// matter how large the stream feeding it has grown.
		lnLo, lnHi := lnNormRange(tr.Layers, len(tr.Tokens), tr.FinalNorm)
		pin := math.Sqrt(float64(tr.Config.DModel))
		inMean, outMean := residualNormMeans(tr)
		misconception(c,
			"「残差是加法，所以每加一次，数值就更大一点」",
			// %.0f is wrong for a model whose residual stream is short: random weights put inMean
			// at 0.2233 and outMean at 0.6322, which rounds to "从 0 长到 1" — a sentence that reads
			// as "it barely moved" immediately before the same sentence says "约 2.8 倍，而且每一层
			// 都在变大". Scale the precision to the number.
			fmt.Sprintf("残差流确实会一路变长：各位置平均 ‖x‖ 从块输入处的 %s 长到最后一层输出的 %s，"+
				"约 %.1f 倍，而且每一层都在变大。但每个子层之前都先做 LayerNorm，"+
				"它的输出长度几乎是常数：√d_model = %.2f，实测所有行落在 %.2f–%.2f，也就是 %.0f%%–%.0f%%。"+
				"所以子层看到的输入幅度始终稳定，一路膨胀的只是「原始信息」那条旁路本身。",
				normText(inMean), normText(outMean), outMean/inMean,
				pin, lnLo, lnHi, 100*lnLo/pin, 100*lnHi/pin))
	})
}

// cancellationStats summarises the heads' cancellation over every row of the
// trace: how many rows were measured, how many of them cancel (ratio below 1),
// and the smallest and largest ratio seen.
type cancellationStats struct {
	Rows   int
	Below  int
	Lo, Hi float64
}

// measureCancellation walks every (layer, token) row once. The stage quotes
// three numbers from it — the row count, how many cancel, and the ratio range —
// and computing them in one pass matters: this runs per frame, and two separate
// walks of 72 rows with a norm each was measurable in BenchmarkView (about 4%).
func measureCancellation(tr *model.Trace) cancellationStats {
	st := cancellationStats{Lo: math.Inf(1)}
	for li := range tr.Layers {
		l := tr.Layers[li]
		for i := 0; i < len(tr.Tokens) && i < len(l.AttnProj); i++ {
			c, ok := headCancellation(l, i)
			if !ok {
				continue
			}
			st.Rows++
			if c < 1 {
				st.Below++
			}
			if c < st.Lo {
				st.Lo = c
			}
			if c > st.Hi {
				st.Hi = c
			}
		}
	}
	return st
}

// residualWorkings prints the additions that build one dimension of the residual
// stream. It is the one place in the lab where the point is the running total
// rather than any single term, so every line ends on it.
func residualWorkings(c *ui.Context, dim int, steps []residualStep, isLast bool, layer int) {
	lines := make([]string, 0, len(steps))
	// The named layer is 0-based everywhere the reader can see it: the segmented control above the
	// stage reads "层 0 | 1", the inspector header reads "层 1", and every card heading quotes
	// a.Layer. This line used to say "第 2 层" for layer 1 — a second numbering scheme on the same
	// screen, so a reader comparing the control to the arithmetic found two different layer numbers.
	lines = append(lines, fmt.Sprintf("d%d 的残差流（层 %d）：%s = %+.4f",
		dim, layer, steps[0].Label, steps[0].Term))
	running := steps[0].Sum
	for _, s := range steps[1:] {
		// Parenthesise the term when it is negative, so the line reads
		// "-0.1050 + (-1.0295) = -1.1344" rather than the ambiguous
		// "-0.1050 + -1.0295".
		term := fmt.Sprintf("%+.4f", s.Term)
		if s.Term < 0 {
			term = fmt.Sprintf("(%+.4f)", s.Term)
		}
		lines = append(lines, fmt.Sprintf("%+.4f %s %s = %+.4f", running, s.Label, term, s.Sum))
		running = s.Sum
	}
	next := "送往下一层"
	if isLast {
		next = "送入最终 LayerNorm"
	}
	lines = append(lines, fmt.Sprintf("每一次加法都不覆盖原值，所以层 %d 的输出完整%s", layer, next))
	workingsLines(c, lines)
}

// residualNormMeans returns the mean L2 norm of the residual stream at the block
// input and at the last block's output, averaged over positions. The residual
// stage quotes both to show that the stream grows while its normalized view of
// it does not.
func residualNormMeans(tr *model.Trace) (in, out float64) {
	n := len(tr.Tokens)
	if n == 0 || len(tr.Layers) == 0 {
		return 0, 0
	}
	first, last := tr.Layers[0], tr.Layers[len(tr.Layers)-1]
	for i := 0; i < n; i++ {
		in += float64(norm(first.In[i]))
		out += float64(norm(last.AfterFFN[i]))
	}
	return in / float64(n), out / float64(n)
}

// lnWorkings prints how one entry of the normalized row was computed. Hovering
// a vector strip above changes dim, so the reader can walk the row and watch
// the same formula produce each value.
func lnWorkings(c *ui.Context, dim int, p lnParts) {
	// γ and β carry five decimals, not three: at three, (x − μ)·inv · γ picks up a
	// 3e-4 error and the printed result lands in the wrong fourth decimal — 90 of
	// 96 dimensions disagreed with the line as printed. Five cuts that roughly in
	// half, and the line marks the result "≈" with a note underneath, because
	// rounding five factors to a few decimals cannot close exactly and pretending
	// otherwise sends the reader hunting for an error that is not there.
	// The mean is the subtrahend and is negative on about half the rows, so it needs the
	// same bracket the softmax line uses: "(x − -0.0958)" runs two signs together and makes
	// the reader decode the expression before checking it.
	mu := fmt.Sprintf("%.4f", p.mean)
	if p.mean < 0 {
		mu = "(" + mu + ")"
	}
	workingsLines(c, []string{
		// The line travels with its own subject. By the time a reader scrolls a tall card down
		// to the arithmetic, the card's title is off screen — measured at 1180×720, the
		// LayerNorm line sits 700 DIP below its heading. The residual chain's first line
		// already opens with "d7 的残差流在第 1 层"; this one started at "d7：μ = …" and a
		// reader arriving at it could not tell which of the card's two subjects it belonged
		// to.
		fmt.Sprintf("LayerNorm d%d：μ = %.4f，σ = %.4f，1/√(σ²+ε) = %.4f", dim, p.mean, 1/p.invStd, p.invStd),
		fmt.Sprintf("(%.4f − %s) × %.4f × γ %.5f + β %.5f ≈ %.4f",
			p.x, mu, p.invStd, p.gamma, p.beta, p.out),
	})
	mutedText(c, "算式按模型的 float32 精度算；这里把 μ、σ 舍到四位、γ 和 β 舍到五位，"+
		"所以自己代入时末位可能有几个单位的出入。")
}

// lnNormRange returns the smallest and largest L2 norm of every LayerNorm output
// in the trace, final norm included. It backs the residual stage's claim that
// normalization pins a row's length even while the residual stream grows.
func lnNormRange(layers []model.LayerTrace, tokens int, final [][]float32) (lo, hi float64) {
	lo = math.Inf(1)
	for li := range layers {
		for i := 0; i < tokens && i < len(layers[li].Norm1); i++ {
			for _, row := range [][]float32{layers[li].Norm1[i], layers[li].Norm2[i]} {
				if v := float64(norm(row)); v < lo {
					lo = v
				} else if v > hi {
					hi = v
				}
			}
		}
	}
	for i := 0; i < tokens && i < len(final); i++ {
		if v := float64(norm(final[i])); v < lo {
			lo = v
		} else if v > hi {
			hi = v
		}
	}
	return lo, hi
}

// geluFloor is GELU's global minimum, about -0.16997 at x ≈ -0.7518. It is the
// number that makes "GELU has a negative output range" concrete: ReLU bottoms
// out at exactly 0, GELU bottoms out here.
const geluFloor = -0.170

// zeroDims counts the entries of v that are exactly zero, which is what makes
// "sparse" a statement about magnitude rather than about zeros.
func zeroDims(v []float32) int {
	n := 0
	for _, x := range v {
		if x == 0 {
			n++
		}
	}
	return n
}

// columnSumRange returns the smallest and largest column sums of an attention
// matrix. Rows sum to 1 by construction; columns have no such constraint, which
// is the fact the attention stage's callout teaches. Only the causal mask
// bounds a column: entries above the diagonal are zero.
func columnSumRange(attn [][]float32) (max, min float64) {
	n := len(attn)
	if n == 0 {
		return 0, 0
	}
	min = math.Inf(1)
	for j := 0; j < n; j++ {
		var sum float64
		for i := j; i < n; i++ {
			if j < len(attn[i]) {
				sum += float64(attn[i][j])
			}
		}
		if sum > max {
			max = sum
		}
		if sum < min {
			min = sum
		}
	}
	return max, min
}

// negDims counts the negative entries of v, the measurement the FFN stage
// quotes next to the GELU floor.
func negDims(v []float32) int {
	n := 0
	for _, x := range v {
		if x < 0 {
			n++
		}
	}
	return n
}

// ffnParamShare returns the FFN's share of all parameters, and its share of the
// parameters inside the transformer blocks. The lesson used to claim "two
// thirds of the parameters are here", which is only true of the second number:
// the embedding tables take 12% of this model, so the FFN is 58% of the whole.
func ffnParamShare(c model.Config) (ofAll, ofBlocks float64) {
	d := float64(c.DModel)
	emb := float64(c.Vocab)*d + float64(c.Ctx)*d
	attn := float64(c.NLayer) * 4 * d * d
	ffn := float64(c.NLayer) * 2 * d * float64(c.DFF)
	ln := float64(c.NLayer) * 4 * d
	final := 2 * d
	all := emb + attn + ffn + ln + final
	blocks := attn + ffn + ln
	return ffn / all, ffn / blocks
}

// activeThreshold is the cutoff behind the FFN card's "激活 > 0.01 的维度" line. The
// stage and its inspector both quote it, so it lives in one place: two copies of the
// number would let the two panels report different counts for the same row.
const activeThreshold = 0.01

// countActive returns how many of the hidden dimensions clear activeThreshold. Shared by
// the FFN stage card and its inspector, which state the same count.
func countActive(hidden []float32) int {
	n := 0
	for _, v := range hidden {
		if v > activeThreshold {
			n++
		}
	}
	return n
}

func (a *App) stageFFN(c *ui.Context) {
	l := a.layer()
	i := clampIndex(a.SelToken, len(a.Trace.Tokens))
	hidden := l.Hidden[i]
	active := countActive(hidden)
	card(c, func() {
		ofAll, ofBlocks := ffnParamShare(a.Trace.Config)
		sectionTitle(c, "升维 → GELU → 降维",
			fmt.Sprintf("层 %d · d_ff = %d · 占全部参数 %.0f%%（块内 %.0f%%）",
				a.Layer, len(hidden), ofAll*100, ofBlocks*100))
		ui.Text(c, "W₁·LN(x) 后过 GELU").FontSize(fontMicro).TextColor(colFaint)
		vectorStrip(c, hidden, 22, &a.HoverDimH)
		// "稀疏" is a claim about concentration, and a count of values above an
		// arbitrary 0.01 cut does not support it: measured on this model, none of
		// the 384 activations is near zero, so the "> 0.01" figure reads as 38%
		// sparse while the real structure is that the largest tenth carries half
		// the mass. Report the concentration instead, and keep the threshold
		// count beside it as what it is.
		share, topN := activationConcentration(hidden)
		ui.Row(c).Gap(16).Wrap().Children(func() {
			ui.Text(c, fmt.Sprintf("最大的 %d / %d 个维度承担 %.0f%% 的激活绝对值",
				topN, len(hidden), share*100)).FontSize(fontSmall).TextColor(colMuted)
			ui.Text(c, fmt.Sprintf("输出 > %.2f 的维度 %d / %d (%.0f%%)", activeThreshold,
				active, len(hidden), 100*float64(active)/float64(len(hidden)))).FontSize(fontSmall).TextColor(colFaint)
			_, hi := minMaxFloat(hidden)
			ui.Text(c, fmt.Sprintf("峰值 %+.2f", hi)).FontSize(fontSmall).TextColor(colFaint)
		})
		histogram(c, hidden, 36, colGreen)
		// The negative region holds most of the values in a sliver of the axis
		// width, so the histogram cannot show its shape there — it is one bar.
		// Stating the proportion is more honest than letting a reader conclude
		// from a single tall column that nothing is there.
		hLo, hHi := minMaxFloat(hidden)
		if hLo < 0 && hHi > hLo {
			neg := 0
			for _, v := range hidden {
				if v < 0 {
					neg++
				}
			}
			axisShare := 100 * (0 - float64(hLo)) / (float64(hHi) - float64(hLo))
			// "只剩一根柱子" describes what happens when the negative region is squeezed into a
			// sliver of the axis: 4.0% on the trained weights. Random weights give it 28.9%, where
			// the bins are wide enough to read, so the claim has to follow the measurement.
			squeeze := fmt.Sprintf("全挤在 0 左边、只占 %.1f%% 的轴宽，这一侧的直方图因此只剩一根柱子——"+
				"它们的具体分布看上面的色带，不要从这张图判断。", axisShare)
			if axisShare >= 10 {
				squeeze = fmt.Sprintf("都落在 0 左边、占 %.1f%% 的轴宽，这一侧还能看出分布的形状——"+
					"更细的结构看上面的色带。", axisShare)
			}
			mutedText(c, fmt.Sprintf("GELU 不是 ReLU：负值不会被截成 0，只是被平滑压向 0，所以分布右侧拖出一条长尾，这就是「稀疏但连续」。"+
				"竖线标出 0 的位置：%d 个负值（占 %.0f%%）%s",
				neg, 100*float64(neg)/float64(len(hidden)), squeeze))
		} else {
			mutedText(c, "GELU 不是 ReLU：负值不会被截成 0，只是被平滑压向 0，所以分布右侧拖出一条长尾，这就是「稀疏但连续」。")
		}
		misconception(c,
			"「GELU 只是 ReLU 的平滑版，输出照样非负」",
			fmt.Sprintf("%d 个维度里就有 %d 个是负数——GELU 有真实的负输出，最小值 %.3f（在 x ≈ -0.75 处）；"+
				"直方图最左端那个 %.2f 就是它：负区是先掉进一个小坑再回到 0，不是被截平。"+
				"所以这里的「稀疏」指「大部分维度接近 0」，而不是「大部分维度恰好等于 0」——"+
				"这一行里恰好等于 0 的有 %d 个。",
				len(hidden), negDims(hidden), geluFloor, geluFloor, zeroDims(hidden)))
		ui.Text(c, "× W₂ 回到 d_model").FontSize(fontMicro).TextColor(colFaint)
		vectorStrip(c, l.FFNOut[i], 18, &a.HoverDimOut)
		// One output dimension of the sum, with the largest contributions and
		// the exact remainder. The output strip is d_model wide while the
		// hidden strip above is d_ff wide, so the formula follows its own
		// cursor: sharing one index meant pointing at hidden d200 (a real
		// dimension, since d_ff is 384) silently showed out[0] instead.
		d := a.W.Config.DModel
		dim := 0
		if a.HoverDimOut >= 0 && a.HoverDimOut < d {
			dim = a.HoverDimOut
		}
		if p := computeFFN(hidden, a.W.Layers[a.Layer].W2, len(hidden), d, dim); p.OK {
			ffnWorkings(c, dim, len(hidden), p)
		}
	})
}

// ffnWorkings prints how one FFN output dimension is summed. The remainder is
// shown because it is most of the answer: the three largest terms carry only
// 14–16% of the absolute mass in this model, so a reader who saw just those
// would reasonably conclude the rest does not matter.
func ffnWorkings(c *ui.Context, dim, dff int, p ffnParts) {
	// Four decimals, like the total: at three the terms and the remainder each
	// carried up to 5e-4 and the reader's sum missed the printed result on 88 of
	// 96 dimensions (worst 1.5e-3). Four pieces of a sum being re-added have to
	// share one precision.
	var terms []string
	for _, t := range p.Top {
		terms = append(terms, fmt.Sprintf("k%d %+.4f", t.K, t.V))
	}
	workingsLines(c, []string{
		fmt.Sprintf("out[%d] 的三项最大贡献：%s", dim, strings.Join(terms, "  ")),
		fmt.Sprintf("+ 其余 %d 项 %+.4f ≈ %+.4f", dff-len(p.Top), p.Rest, p.Out),
	})
	ui.Text(c, fmt.Sprintf("余项是精确补足的和，不是估算值——三个最大项并不主导结果，只占 %.0f%%。"+
		"各行按四位小数显示，自己相加时末位可能差一两个单位。",
		ffnTopShare(p))).FontSize(fontMicro).TextColor(colFaint)
}

// ffnTopShare is the percentage of the sum's absolute mass carried by the terms
// shown. The denominator is Σ|term| over every k, not |out|: the signed sum
// cancels heavily, so dividing by it would report a share several times larger
// than the terms actually carry.
func ffnTopShare(p ffnParts) float64 {
	if p.AbsTotal == 0 {
		return 0
	}
	var top float64
	for _, t := range p.Top {
		top += math.Abs(t.V)
	}
	return 100 * top / p.AbsTotal
}

// topN is how many candidates the output stage lists. The card's title takes its name
// from it, so the heading and the slice cannot disagree about how many rows there are.
const topN = 12

func (a *App) stageOutput(c *ui.Context) {
	tr := a.Trace
	last := len(tr.Tokens) - 1
	probs := tr.Probabilities(a.Temp)
	top := model.TopK(probs, topN)
	var maxP float32
	for _, id := range top {
		if probs[id] > maxP {
			maxP = probs[id]
		}
	}

	card(c, func() {
		// Every other stage redraws when the selected token changes; this one does not,
		// because only the last position is projected. Selecting token 5 here leaves the
		// card unchanged, which reads as an interaction that did nothing. Say which
		// position the strip is showing whenever it is not the selected one.
		note := "最终 LayerNorm 之后"
		if a.SelToken != last && last >= 0 {
			note = fmt.Sprintf("t=%d · 最终 LayerNorm 之后；只有这个位置会被投影，选中的 t=%d 不参与预测",
				last, a.SelToken)
		}
		sectionTitle(c, "最后位置的向量", note)
		vectorStrip(c, tr.FinalNorm[last], 22, &a.HoverDimIn)
		// An entropy of 0.16 nats means nothing without a scale. The ceiling is
		// log(vocab) — the entropy of a uniform distribution over all 256 bytes —
		// and the stage's whole point is how far below it this model sits, so
		// state both. The attention stage already anchors its entropy against
		// log(n); this is the same idea one stage later.
		ent := float64(model.Entropy(probs))
		ceiling := math.Log(float64(len(probs)))
		kv(c, "下一 token", fmt.Sprintf("温度 %.2f · 熵 %.2f nats", a.Temp, ent))
		kv(c, "熵的参考", fmt.Sprintf("均匀分布 log(%d) = %.2f nats，当前占 %.0f%%",
			len(probs), ceiling, 100*ent/ceiling))
	})

	card(c, func() {
		// The list is a slice, and on this model the slice is essentially the
		// whole distribution: the top 12 carry 99.99999% of the mass and the
		// other 244 candidates share the remaining 0.00002%. Saying only
		// "Top-12" left a reader to conclude the vocabulary has twelve entries
		// — the opposite of what a token-prediction stage should teach.
		var rest float32
		inTop := map[int]bool{}
		for _, id := range top {
			inTop[id] = true
		}
		for id, p := range probs {
			if !inTop[id] {
				rest += p
			}
		}
		// %.4f would print 0.0000 for a tail that shares 0.00002%, which reads as
		// nothing at all. The rows below already switch to scientific notation
		// when a probability gets small; the summary follows the same rule.
		// A tail that rounds to zero in float32 is zero, not "0.0e+00%". At T=0.1 the top
		// twelve carry the whole distribution and the remainder is 5.6e-17 — formatting that
		// with scientific notation printed a malformed-looking number for what is simply no
		// probability left.
		// Below this the printed figure is float32 residue, not a probability. At T=0.1 the top
		// twelve carry everything and the remainder lands at 5.6e-17; at T=0.3 it is 1.5e-19.
		// Formatting those in scientific notation presents arithmetic noise as a measurement —
		// "1.5e-17%" reads like a very small but real share.
		restText := fmt.Sprintf("%.4f%%", rest*100)
		switch {
		case rest*100 < 1e-9:
			restText = "0%"
		case rest*100 < 0.01:
			restText = fmt.Sprintf("%.1e%%", rest*100)
		}
		sectionTitle(c, fmt.Sprintf("Top-%d 概率", topN),
			fmt.Sprintf("softmax(logits / T) · 词表 %d 个，其余 %d 个合计 %s",
				len(probs), len(probs)-len(top), restText))
		rows := make([]barRow, 0, len(top))
		for rank, id := range top {
			// 0.0% for every tail row tells the reader nothing; keep the
			// significant digits where they still exist.
			pct := fmt.Sprintf("%.1f%%", probs[id]*100)
			switch {
			case probs[id] == 0:
				pct = "0%"
			case probs[id] < 0.0001:
				pct = fmt.Sprintf("%.1e%%", probs[id]*100)
			case probs[id] < 0.01:
				pct = fmt.Sprintf("%.2f%%", probs[id]*100)
			}
			rows = append(rows, barRow{
				Label:      fmt.Sprintf("%q", byteText(id)),
				Value:      probs[id],
				ValueText:  pct,
				Color:      probColor(probs[id], maxP),
				LabelWidth: 56,
				ValueWidth: 50,
				// "第 %d 名" rather than "#%d": the hash form means a token index everywhere
				// else in the app ("token #35", the rail's "#4", the inspector header), so
				// reusing it for a rank printed two different things under one symbol.
				Tip:   fmt.Sprintf("id %d · %q · 第 %d 名", id, byteText(id), rank+1),
				Index: rank,
			})
		}
		bars(c, rows, maxP, colAccent2)
		// The top-1 probability and the training budget both come from the
		// running model, so the callout stays true when the prompt, the
		// temperature or the weights change.
		trained := "没有训练信息"
		if a.W.Steps > 0 {
			trained = fmt.Sprintf("只训练了 %d 步", a.W.Steps)
		}
		// "这么高的把握" was a fixed phrase, and it stops making sense when the top-1 share is 1%:
		// random weights spread the distribution near-uniformly, and calling 1% high confidence
		// contradicts the number the same sentence just printed.
		confidence := "完全可以对一个字节有这么高的把握而依然写出乱码"
		if maxP < 0.2 {
			confidence = "根本没有那么大的把握，输出仍然可能是乱码"
		} else if maxP < 0.5 {
			confidence = "完全可以对一个字节有那么大的把握而依然写出乱码"
		}
		misconception(c,
			fmt.Sprintf("「概率 %.0f%% = 模型很确定下一个字就是它，所以它一定对」", maxP*100),
			fmt.Sprintf("这两件事没有关系。softmax 只把 logits 压成一个分布，%.0f%% 说的是"+
				"「在这个分布里它最大」，不是「它是对的」——一个 %d 万参数、%s的玩具模型"+
				"%s。把温度往下拖，这个比例会逼近 100%%，但输出只是更重复，并不会更正确。",
				maxP*100, a.paramCount()/10000, trained, confidence))
	})

	card(c, func() {
		sectionTitle(c, "采样", "从分布里抽一个 token")
		ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
			if ui.PrimaryButton(c, "采样并续写").Clicked() {
				a.sampleNext()
			}
			ui.Text(c, "续写结果会追加到提示词，并重新跑一遍前向。").FontSize(fontMicro).TextColor(colFaint)
		})
		if a.LastSample != "" {
			kv(c, "上一次", a.LastSample)
		}
		// Show where the draw lands, so "sampling" does not read as an
		// unreproducible random pick: the position comes from the prompt, and
		// the interval it falls in is exactly as wide as the drawn token's
		// probability.
		if d, ok := a.sampleDraw(probs); ok {
			workingsLines(c, []string{
				fmt.Sprintf("种子 = FNV-1a(提示词) = 0x%016X", d.seed),
				fmt.Sprintf("x = (种子 mod 1000003) ÷ 1000003 = %d ÷ 1000003 = %.6f",
					d.seed%1000003, d.x),
				fmt.Sprintf("x 落在 [%.4f, %.4f) 这一段 → 抽到 id %d %q（这一段宽度 %.4f 就是它的概率）",
					d.lo, d.hi, d.pick, byteText(d.pick), d.prob),
			}, workingsGapWide)
			// "同一句提示词永远抽出同一个 token" was too strong: the draw is a
			// function of the prompt AND the temperature, and the temperature
			// slider sits on this very card. At T=2.2 the same prompt draws a
			// different token, so the claim needs its condition attached.
			mutedText(c, "种子只来自提示词文字，所以在温度不变的前提下，同一句提示词永远抽出同一个 token；"+
				"改一个字种子就会变，拖动温度也会改变落在哪一段（分布的形状变了，同一个 x 可能落进别的区间）。")
		}
	})
}

func (a *App) lesson(c *ui.Context) {
	l := lessons[a.Step]
	card(c, func() {
		sectionTitle(c, "这一段在讲什么", l.name)
		body := l.body
		if l.bodyFn != nil && a.Trace != nil {
			body = l.bodyFn(a.Trace)
		}
		mutedText(c, body)
		// The top bar dims the controls this stage does not read; this line
		// says where they do apply, so the dimming is a signpost instead of a
		// dead end. Only the knobs that are off are named.
		var off []string
		if !inStages(a.Step, layerFrom, layerTo) {
			off = append(off, "层")
		}
		if !inStages(a.Step, headFrom, headTo) {
			off = append(off, "头")
		}
		if !inStages(a.Step, tempFrom, tempTo) {
			off = append(off, "温度")
		}
		if len(off) > 0 {
			ui.Text(c, fmt.Sprintf("顶部「%s」在这一步不生效：%s",
				strings.Join(off, " / "), controlScopeNote(off))).
				FontSize(fontMicro).TextColor(colFaint).LineHeight(1.4)
		}
		// The Shift+arrow line has to follow the stage: on the two attention
		// stages those keys walk the matrix cell, elsewhere they move the token.
		// One shared sentence told every stage "切换 token", which was simply
		// wrong on the two stages that draw a matrix — the ones where a reader is
		// most likely to try the keys.
		shift := "Shift+← → 切换 token"
		if a.hasMatrix() {
			shift = "Shift+← → ↑ ↓ 移动矩阵格子"
		}
		ui.Text(c, fmt.Sprintf("快捷键：← → 切换阶段 · %s · 空格 播放/暂停 · Enter 采样 · 1–7 跳转", shift)).
			FontSize(fontMicro).TextColor(colFaint)
	})
}

// controlScopeNote says where the controls a stage does not read take effect,
// measured in steps so it matches the numbered header above.
func controlScopeNote(off []string) string {
	var parts []string
	for _, name := range off {
		switch name {
		case "层":
			parts = append(parts, fmt.Sprintf("第 %d–%d 步选层", layerFrom+1, layerTo+1))
		case "头":
			parts = append(parts, fmt.Sprintf("第 %d–%d 步选头", headFrom+1, headTo+1))
		case "温度":
			parts = append(parts, fmt.Sprintf("第 %d 步改温度", tempFrom+1))
		}
	}
	return strings.Join(parts, " · ")
}

func clampIndex(i, n int) int {
	if n == 0 {
		return 0
	}
	if i < 0 {
		return 0
	}
	if i >= n {
		return n - 1
	}
	return i
}

func byteText(id int) string {
	b := byte(id)
	switch b {
	case ' ':
		return "␣"
	case '\n':
		return "\\n"
	case '\t':
		return "\\t"
	}
	if b >= 32 && b < 127 {
		return string(rune(b))
	}
	return fmt.Sprintf("\\x%02x", b)
}

// LessonNames lists the stages in order, for the command line.
func LessonNames() []string {
	names := make([]string, len(lessons))
	for i, l := range lessons {
		names[i] = l.name
	}
	return names
}

// --------------------------------------------------- inspector, per lesson

func (a *App) inspectorBytes(c *ui.Context, tr *model.Trace) {
	tk := tr.Tokens[a.SelToken]
	from, to := a.SelToken, a.SelToken
	for from > 0 && tr.Tokens[from-1].Start == tk.Start {
		from--
	}
	for to+1 < len(tr.Tokens) && tr.Tokens[to+1].Start == tk.Start {
		to++
	}
	card(c, func() {
		sectionTitle(c, "字节", "UTF-8 编码")
		ui.Row(c).Gap(4).Children(func() {
			for i := from; i <= to; i++ {
				i := i
				ui.Column(c).Gap(2).AlignItems(ui.Center).Children(func() {
					ui.Text(c, fmt.Sprintf("%d", tr.Tokens[i].Byte)).FontSize(fontData).TextColor(colText).
						Padding(3, 7).Radius(4).Background(colPanel2)
					ui.Text(c, fmt.Sprintf("#%d", i)).FontSize(fontMicro).TextColor(colFaint)
				})
			}
		})
		kv(c, "rune", fmt.Sprintf("%q", tk.Rune))
		kv(c, "位置", fmt.Sprintf("%d / %d", a.SelToken, len(tr.Tokens)-1))
	})
	card(c, func() {
		// The heading asks a question about the current prompt. With an ASCII-only prompt the
		// answer is "it does not", and leaving the question there made the card read as though
		// the reader had missed something — the figures underneath (36 字节 / 36 字符) were
		// actually evidence *against* the premise. Measured on an English prompt the heading
		// claimed a comparison the numbers did not show.
		bytes, runes := len(tr.Text), utf8.RuneCountInString(tr.Text)
		if bytes == runes {
			sectionTitle(c, "ASCII 下字节就是字符", "")
			mutedText(c, "这个提示词全是 ASCII：一个字符编成一个字节，也就是一个 token，"+
				"所以下面两个数相等。贴一句中文进来，它们就会分开。")
		} else {
			sectionTitle(c, "为什么中文更长", "")
			mutedText(c, "ASCII 一个字节就是一个 token；一个汉字在 UTF-8 里占 3 个字节，所以是 3 个 token。")
		}
		kv(c, "提示词", fmt.Sprintf("%d 字节 / %d 字符", bytes, runes))
	})
}

func (a *App) inspectorEmbed(c *ui.Context, tr *model.Trace) {
	i := clampIndex(a.SelToken, len(tr.Tokens))
	card(c, func() {
		sectionTitle(c, "向量统计", fmt.Sprintf("d_model = %d", tr.Config.DModel))
		em, es := meanStd(tr.Embed[i])
		pm, ps := meanStd(tr.Pos[i])
		xm, xs := meanStd(tr.Input[i])
		kv(c, "E 均值/σ", fmt.Sprintf("%+.3f / %.3f", em, es))
		kv(c, "P 均值/σ", fmt.Sprintf("%+.3f / %.3f", pm, ps))
		kv(c, "x 均值/σ", fmt.Sprintf("%+.3f / %.3f", xm, xs))
		kv(c, "x L2", fmt.Sprintf("%.3f", norm(tr.Input[i])))
		kv(c, "位置 t", fmt.Sprint(i))
	})
	card(c, func() {
		sectionTitle(c, "位置向量", "同一维度在不同 t 上的值")
		mutedText(c, "位置向量是学出来的，每个位置一行；它让注意力能区分先后。")
	})
}

func (a *App) inspectorScores(c *ui.Context, tr *model.Trace) {
	l := a.layer()
	i := clampIndex(a.SelToken, len(tr.Tokens))
	h := l.Heads[a.Head]
	card(c, func() {
		sectionTitle(c, "分数 top-5", fmt.Sprintf("query = token %d", i))
		rows := make([]barRow, 0, 8)
		var maxAbs float32
		for j := 0; j <= i; j++ {
			v := h.Scores[i][j]
			if av := float32(math.Abs(float64(v))); av > maxAbs {
				maxAbs = av
			}
			rows = append(rows, barRow{
				Label:      fmt.Sprintf("%d %s", j, tr.Tokens[j].Text),
				Value:      v,
				ValueText:  fmt.Sprintf("%+.2f", v),
				Color:      scoreColor(v, maxAbs),
				LabelWidth: 62,
				ValueWidth: 48,
				Index:      j,
			})
		}
		rows = topAbsBars(rows, 5)
		bars(c, rows, maxAbs, colWarm)
		kv(c, "可见 key", fmt.Sprintf("%d 个", i+1))
		kvScale(c, tr.Config.DHead())
	})
}

func (a *App) inspectorAttention(c *ui.Context, tr *model.Trace) {
	l := a.layer()
	i := clampIndex(a.SelToken, len(tr.Tokens))
	h := l.Heads[a.Head]
	card(c, func() {
		sectionTitle(c, "这一行的统计", fmt.Sprintf("层 %d · 头 %d", a.Layer, a.Head))
		row := h.Attn[i]
		var maxW float32
		best := 0
		over := 0
		for j := 0; j <= i; j++ {
			if row[j] > maxW {
				maxW, best = row[j], j
			}
			if row[j] > 0.05 {
				over++
			}
		}
		kv(c, "最大权重", fmt.Sprintf("%.3f → token %d %q", maxW, best, tr.Tokens[best].Text))
		kv(c, "权重 > 5%", fmt.Sprintf("%d 个 key", over))
		kv(c, "熵", fmt.Sprintf("%.2f nats", model.Entropy(row[:min(len(row), i+1)])))
		kv(c, "输出 L2", fmt.Sprintf("%.3f", norm(h.Out[i])))
		// How far this layer's attention is from the previous layer's for the
		// same row and head. Switching the layer control redraws the heatmap,
		// but two successive pictures cannot tell a reader whether the layers
		// are doing the same thing at a different strength or looking somewhere
		// else entirely.
		if d := computeLayerDivergence(tr.Layers, a.Layer, a.Head, i); d.Ok {
			// The layer control labels layers from 0, so name the previous
			// layer the same way. Writing a.Layer here would read "与第 1 层之差"
			// while showing layer 1 — a comparison with itself.
			kv(c, fmt.Sprintf("与第 %d 层之差", a.Layer-1),
				fmt.Sprintf("%.2f（0 = 相同，1 = 不重叠）", d.TV))
		}
		mutedText(c, "熵越接近 0 越专注；越接近 log(n) 越分散。")
		// Row 0 can only see itself, so every head's entropy there is exactly 0 and the
		// chart is four empty bars against a log(1) = 0 scale. State the reason instead of
		// drawing it: the causal mask, not the model, is what makes this row trivial.
		if i == 0 {
			mutedText(c, "第 0 行只能看到它自己，四个头的熵必然都是 0——从第 1 行起才有可比较的分配。")
			return
		}
		// The four heads are a segmented control above the stage, and switching
		// between them shows different bars but no way to compare. One compact
		// block per head, with the current one marked, makes "this head is the
		// focused one" a reading rather than a memory task.
		headRows := make([]barRow, 0, len(l.Heads))
		for hi := range l.Heads {
			hr := l.Heads[hi].Attn[i]
			e := model.Entropy(hr[:min(len(hr), i+1)])
			label := fmt.Sprintf("H%d", hi)
			colour := colGreen
			if hi == a.Head {
				label = fmt.Sprintf("H%d ◂", hi)
				colour = colAccent2
			}
			headRows = append(headRows, barRow{
				Label:      label,
				Value:      e,
				ValueText:  fmt.Sprintf("%.2f", e),
				Color:      colour,
				LabelWidth: 40,
				ValueWidth: 34,
				Tip:        fmt.Sprintf("头 %d 这一行的熵 %.2f nats", hi, e),
			})
		}
		ui.Text(c, "这一行在各头的熵（越小越专注）").FontSize(fontMicro).TextColor(colFaint)
		bars(c, headRows, float32(math.Log(float64(i+1))), colGreen)
	})
}

func (a *App) inspectorResidual(c *ui.Context) {
	l := a.layer()
	i := clampIndex(a.SelToken, len(a.Trace.Tokens))
	card(c, func() {
		sectionTitle(c, "残差流统计", "这一层的输出向量")
		vec := l.AfterFFN[i]
		mean, std := meanStd(vec)
		kv(c, "维度", fmt.Sprint(len(vec)))
		kv(c, "均值", fmt.Sprintf("%+.4f", mean))
		kv(c, "标准差", fmt.Sprintf("%.4f", std))
		kv(c, "L2", fmt.Sprintf("%.3f", norm(vec)))
		lo, hi := minMaxFloat(vec)
		kv(c, "范围", fmt.Sprintf("%+.2f … %+.2f", lo, hi))
	})
	card(c, func() {
		sectionTitle(c, "LayerNorm", "进入 FFN 之前")
		mb, sb := meanStd(l.AfterAttn[i])
		ma, sa := meanStd(l.Norm2[i])
		kv(c, "均值", fmt.Sprintf("%+.3f → %+.3f", mb, ma))
		kv(c, "标准差", fmt.Sprintf("%.3f → %.3f", sb, sa))
		body := "归一化后再乘 γ、加 β，所以最终方差不必恰好为 1。" +
			"γ、β 是训练出来的，不是 1 和 0；第 5 步的算式会把这个维度上的真实取值和结果一起写出来。"
		if !lnParamsAreLearned(a.W) {
			body = "归一化后再乘 γ、加 β，所以最终方差不必恰好为 1。" +
				"这份权重没有训练过，γ 是 1、β 是 0，所以这一步只是纯归一化；第 5 步的算式里能看到这两个值。"
		}
		mutedText(c, body)
	})
}

func (a *App) inspectorFFN(c *ui.Context) {
	l := a.layer()
	i := clampIndex(a.SelToken, len(a.Trace.Tokens))
	hidden := l.Hidden[i]
	active := countActive(hidden)
	mean, std := meanStd(hidden)
	_, hi := minMaxFloat(hidden)
	card(c, func() {
		sectionTitle(c, "隐藏层统计", fmt.Sprintf("d_ff = %d", len(hidden)))
		kv(c, fmt.Sprintf("激活 > %.2f", activeThreshold), fmt.Sprintf("%d / %d (%.0f%%)", active, len(hidden), 100*float64(active)/float64(len(hidden))))
		kv(c, "峰值", fmt.Sprintf("%+.2f", hi))
		kv(c, "均值/σ", fmt.Sprintf("%+.3f / %.3f", mean, std))
		mutedText(c, "GELU 把负值平滑压向 0，所以大部分维度接近零，只有少数被激活。")
	})
}

func (a *App) inspectorLogits(c *ui.Context, tr *model.Trace) {
	probs := tr.Probabilities(a.Temp)
	top := model.TopK(probs, 2)
	card(c, func() {
		sectionTitle(c, "分布统计", fmt.Sprintf("温度 %.2f", a.Temp))
		if len(top) > 0 {
			kv(c, "top-1", fmt.Sprintf("%q · %.1f%%", byteText(top[0]), probs[top[0]]*100))
		}
		if len(top) > 1 {
			kv(c, "领先", fmt.Sprintf("%.1f 个百分点", (probs[top[0]]-probs[top[1]])*100))
		}
		kv(c, "熵", fmt.Sprintf("%.2f nats", model.Entropy(probs)))
		// Temperature sits in the top bar next to the layer and head controls, which read the
		// model's own structure. It is not part of the model at all — nothing in internal/model
		// mentions it — so say which side of the line it is on, where the reader is looking at it.
		mutedText(c, "温度不是训练出来的权重，是推理时的旋钮：趋近 0 时分布变尖，等于贪心；"+
			"大于 1 时变平，输出更随机。")
	})
}

// topAbsBars keeps the n rows with the largest absolute value, then restores
// their original order.
func topAbsBars(rows []barRow, n int) []barRow {
	sorted := append([]barRow(nil), rows...)
	sort.SliceStable(sorted, func(i, j int) bool {
		return math.Abs(float64(sorted[i].Value)) > math.Abs(float64(sorted[j].Value))
	})
	if n > len(sorted) {
		n = len(sorted)
	}
	out := sorted[:n]
	sort.SliceStable(out, func(i, j int) bool { return out[i].Index < out[j].Index })
	return out
}

// previousTokenShare counts the rows whose largest attention weight lands on the token immediately
// before, and the number of rows that have a predecessor at all.
//
// Used by the softmax stage's prose, which used to assert the pattern unconditionally: it describes
// trained weights and is false of random ones, and the app renders both.
func previousTokenShare(attn [][]float32, tokens int) (prev, rows int) {
	for q := 1; q < tokens && q < len(attn); q++ {
		row := attn[q]
		if len(row) == 0 {
			continue
		}
		var maxW float32
		best := 0
		for k := 0; k <= q && k < len(row); k++ {
			if row[k] > maxW {
				maxW, best = row[k], k
			}
		}
		rows++
		if best == q-1 {
			prev++
		}
	}
	return prev, rows
}

// normText formats a residual-stream norm with enough precision to be meaningful.
//
// Two decimal places below 10, none above: the trained model's stream runs from 13 to 67, where the
// fraction is noise, while a random model's sits under 1, where "%.0f" would print 0 and 1 for values
// a factor of three apart.
func normText(v float64) string {
	if v < 10 {
		return fmt.Sprintf("%.2f", v)
	}
	return fmt.Sprintf("%.0f", v)
}
