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
					"它在负值区不是硬零，而是平滑地趋近零，所以激活是稀疏但连续的分布：%s。"+
					"「稀疏」指的是这种集中，不是「大部分维度恰好等于 0」。",
				width, ofAll*100, sparsityEvidence(tr))
		},
		formula: "FFN(x) = W₂ · GELU(W₁ · LN(x))",
	},
	{
		name:  "输出 / 采样",
		title: "从向量到下一个字",
		// The shape in this sentence names two numbers from the loaded weights, so per the
		// bodyFn contract above it cannot be a constant. It used to be: the text said
		// "[T, 96] 变成 [1, 256]" no matter what was loaded, and validate accepts any
		// d_model > 0, so a weights file with d_model 48 loaded fine and the lesson went on
		// describing a 96-wide model. Measured before this change: with d_model=48 the body
		// still read [T, 96].
		bodyFn: func(tr *model.Trace) string {
			return fmt.Sprintf(
				"最后一层的输出经过最终 LayerNorm，再与嵌入矩阵相乘（权重绑定），只把最后一个位置的向量投影到词表上"+
					"（形状从 [T, %d] 变成 [1, %d]），得到下一个 token 的 logit——但每个位置其实都算得出下一个字节"+
					"（下面「每个位置都在预测」那张卡片把它们逐个核对了）。logit 不是概率，要除以温度再做 softmax。"+
					"温度趋近 0 等于贪心，温度大于 1 会让分布变平、输出更随机。",
				tr.Config.DModel, tr.Config.Vocab)
		},
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
			ui.Text(c, fmt.Sprintf("边打边重算，顶部的 token 条会立刻出现；%s 或方向键可以切到别的阶段。",
				stageDigitRange())).
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
	a.cardNamed(c, "token 序列", func() {
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

	a.cardNamed(c, "字符 → 字节", func() {
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
					workingsLinesCopyable(c, []string{
						fmt.Sprintf("0x%02X %08b：不是合法的 UTF-8 字节，单独作为一个 token", rb.Bytes[0], rb.Bytes[0]),
						fmt.Sprintf("文本视图里显示为 �，但模型看到的 id 就是 %d", rb.Bytes[0]),
					})
				} else {
					workingsLinesCopyable(c, []string{
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

	a.cardNamed(c, "数据流与形状", func() {
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
	a.cardNamed(c, "嵌入向量", func() {
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
		workingsLinesCopyable(c, []string{
			fmt.Sprintf("d%d：E[%d][%d] = %+.4f · P[%d] = %+.4f",
				dim, tr.Tokens[i].ID, dim, e, i, pos),
			fmt.Sprintf("%s + %s = %+.4f", signed(float64(e)), signed(float64(pos)), x),
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

	a.stageEmbeddingStructure(c)

	a.cardNamed(c, "位置向量", func() {
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

	a.stagePosEncodingShape(c)
}

// stagePosEncodingShape answers what stage 2 leaves open: the position vector is
// learned, so what did the training produce?
//
// The four strips above show P[t] for four positions, which a reader can look at without
// being able to say anything about them. Two numbers do say something, and both are
// surprising next to the encoding most tutorials describe:
//
//	learned (this model)   cos(P[t], P[t+1]) = 0.040   cos at gap 32 = -0.052
//	sinusoidal (Vaswani)   cos(P[t], P[t+1]) = 0.969   cos at gap 32 =  0.562
//	two random vectors     1/√96             = 0.102
//
// So the learned table puts positions in nearly orthogonal directions — all of them about
// as unrelated as chance, with no smooth progression — while the fixed sinusoidal
// encoding is a slow walk along a curve. Neither is wrong, but they are opposite designs,
// and the point of the card is that nothing in this model's architecture chose the one it
// has: gradient descent did.
func (a *App) stagePosEncodingShape(c *ui.Context) {
	shape := computePosEncodingShape(a.W)
	if !shape.OK {
		return
	}
	a.cardNamed(c, "位置编码的形状", func() {
		sectionTitle(c, "学出来的位置长什么样", "与固定正弦编码的对比")
		mutedText(c, "上面四条色带只能看个热闹。两个余弦值能说清一件事：这张表把位置放成了"+
			"近乎互相垂直的方向，而不是沿一条平滑曲线依次移动——那正是经典正弦编码的样子。")
		// bars draws |value|/max, so a bar's length is a magnitude and cannot show a
		// sign; the signed figure stays in the value column, and the axis runs 0 to 1
		// where a cosine of 1 would fill the row. That is why DistantCos's bar is drawn
		// for 0.052 while its cell reads -0.052: the near-zero length is the point.
		rows := []barRow{
			{Label: "学过 t, t+1", Value: float32(math.Abs(shape.AdjacentCos)),
				ValueText: fmt.Sprintf("%+.3f", shape.AdjacentCos),
				Color:     colGreen, LabelWidth: 104, ValueWidth: 56,
				Tip: "这个模型训练出来的相邻位置余弦"},
			{Label: "正弦 t, t+1", Value: float32(math.Abs(shape.SinusoidalAdjacentCos)),
				ValueText: fmt.Sprintf("%+.3f", shape.SinusoidalAdjacentCos),
				Color:     colAccent2, LabelWidth: 104, ValueWidth: 56,
				Tip: "固定正弦编码的相邻位置余弦"},
			{Label: fmt.Sprintf("学过 t, t+%d", shape.Gap), Value: float32(math.Abs(shape.DistantCos)),
				ValueText: fmt.Sprintf("%+.3f", shape.DistantCos),
				Color:     colGreen, LabelWidth: 104, ValueWidth: 56,
				Tip: fmt.Sprintf("相隔 %d 个位置", shape.Gap)},
			{Label: fmt.Sprintf("正弦 t, t+%d", shape.Gap), Value: float32(math.Abs(shape.SinusoidalDistantCos)),
				ValueText: fmt.Sprintf("%+.3f", shape.SinusoidalDistantCos),
				Color:     colAccent2, LabelWidth: 104, ValueWidth: 56,
				Tip: "同样的间隔，固定正弦编码"},
		}
		bars(c, rows, 1, colGreen)
		workingsLinesCopyable(c, []string{
			fmt.Sprintf("学出来的：cos(P[t],P[t+1]) = %+.3f，隔 %d 个位置是 %+.3f",
				shape.AdjacentCos, shape.Gap, shape.DistantCos),
			fmt.Sprintf("固定正弦：同样两个间隔 = %+.3f 与 %+.3f",
				shape.SinusoidalAdjacentCos, shape.SinusoidalDistantCos),
			fmt.Sprintf("%d 维里两个随机向量的 |cos| 大约是 %.3f——所以学出来的这张表几乎就是正交的",
				shape.Dims, shape.RandomCos),
		})
		misconception(c,
			"「位置编码就是那套 sin/cos 公式，所有 transformer 都这样」",
			fmt.Sprintf("经典 transformer 用的是固定正弦编码，相邻位置余弦 %.2f——每行只是沿曲线走一小步。"+
				"这个模型的位置表是训练出来的，相邻余弦只有 %+.3f，比 %d 维随机向量的 %.3f 还小，"+
				fmt.Sprintf("也就是说它把 %d 个位置放成了 %d 个几乎互不相干的地址。",
					shape.Positions, shape.Positions)+
				"两种都能用：正弦编码让「近」在几何上有意义，学出来的表让模型自己决定哪个方向代表哪个位置。"+
				"这个仓库的架构没规定其中任何一种，是梯度下降选了这个。",
				shape.SinusoidalAdjacentCos, shape.AdjacentCos, shape.Dims, shape.RandomCos))
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
func signed(v float64) string {
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
	a.cardNamed(c, "Q / K / V", func() {
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

	a.cardNamed(c, "分数矩阵", func() {
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

	// After the matrix, not before: the mask card is secondary evidence, and putting
	// it first pushed the matrix itself to y=945 in a 720-tall window (measured) —
	// burying the stage's primary figure to show its explanation.
	a.stageMaskCounterfactual(c)
}

// stageMaskCounterfactual shows what the causal mask is actually preventing.
//
// The stage says the mask is architecture rather than a learned weight, and draws it
// as an empty upper triangle. That is honest but quiet: an empty region reads like a
// formality. Recomputing the same row without the mask answers the question the
// triangle invites — what would this position look at, if it could see everything? —
// and the answer is not marginal. Measured on this model with the default prompt,
// every head puts 51–84% of its unmasked attention on keys the mask hides, and row 0
// gives the future the entire distribution (its top single key is k34, with 47%).
//
// The scores above the diagonal are -Inf in the trace, so they cannot be read back
// from Scores; they are recomputed from the same stored Q and K with the same scale,
// in the same float32 dot order.
func (a *App) stageMaskCounterfactual(c *ui.Context) {
	tr := a.Trace
	l := a.layer()
	n := len(tr.Tokens)
	dh := tr.Config.DHead()
	if n < 2 || a.Head >= len(l.Heads) {
		return
	}
	h := l.Heads[a.Head]
	mean, rows := meanFutureShare(h, n, dh)
	if rows == 0 {
		return
	}
	row := clampIndex(a.SelToken, n)
	cf := computeMaskCounterfactual(h, row, n, dh)
	if !cf.OK {
		return
	}
	a.cardNamed(c, "掩码反事实", func() {
		sectionTitle(c, "掩码挡掉了什么", fmt.Sprintf("把同一行去掉掩码算一遍 · 层 %d · 头 %d", a.Layer, a.Head))
		mutedText(c, "因果掩码不是提醒，是这道题唯一的难度来源。上面那张矩阵的右上角是空的，"+
			"看起来像块留白——这里用模型自己存下来的 Q 和 K，把选中那一行不看掩码地重算一遍，"+
			"看看它本来会看哪里。")
		kvW(c, "这一行原来是", fmt.Sprintf("位置 %d", row), 92)
		kvW(c, "被挡掉的份额", fmt.Sprintf("%.0f%%", 100*cf.FutureShare), 92)
		if cf.FutureTop >= 0 {
			kvW(c, "最想看的位置", fmt.Sprintf("k%d（%.0f%%，在它后面 %d 个位置）",
				cf.FutureTop, 100*cf.FutureTopShare, cf.FutureTop-row), 92)
		}
		kvW(c, "熵（去掩码后）", fmt.Sprintf("%.2f nats（上限 ln %d = %.2f）",
			cf.MeanEntropy, n, math.Log(float64(n))), 92)
		kvW(c, "熵（模型实算）", fmt.Sprintf("%.2f nats（只看得到 %d 个 key）",
			cf.VisibleEntropy, row+1), 92)
		// The per-head average is what makes this a property of the model rather than
		// an artefact of the one row the reader happens to be on.
		workingsLinesCopyable(c, []string{
			fmt.Sprintf("这一行去掉掩码后，%.0f%% 的注意力会落在自己之后的位置上", 100*cf.FutureShare),
			fmt.Sprintf("头 %d 整句平均是 %.0f%%（%d 行）——每个头都在 51%%–84%% 之间", a.Head, 100*mean, rows),
			fmt.Sprintf("熵：模型实算 %.2f → 去掉掩码 %.2f nats", cf.VisibleEntropy, cf.MeanEntropy),
		})
		// Row 0 is computed on its own rather than assumed. The first draft used the
		// selected row's numbers in the sentence about row 0 — "最大的一格是 k29 的
		// 18%" was cf's top future key while the sentence named row 0, which is the kind
		// of mixed-scope number this project keeps finding.
		first := computeMaskCounterfactual(h, 0, n, dh)
		rowZero := ""
		if first.OK {
			rowZero = fmt.Sprintf("第 0 行最夸张：它只有自己一个可见 key，去掉掩码后 %.0f%% 的注意力"+
				"跑到后面，最大的一格是 k%d 的 %.0f%%。",
				100*first.FutureShare, first.FutureTop, 100*first.FutureTopShare)
		}
		misconception(c,
			"「掩码只是不让模型偷看，去掉也不会有大变化」",
			fmt.Sprintf("去掉以后变化极大：这一行 %.0f%% 的注意力立刻跑到后面去了。%s"+
				"也就是说掩码不是「顺手加的安全带」——没有它，模型第一步就会去读后面还没生成的内容，"+
				"训练目标也就失去意义了。", 100*cf.FutureShare, rowZero))
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
	workingsLinesCopyable(c, []string{
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

	a.cardNamed(c, "注意力矩阵", func() {
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

	a.stageHeadProfiles(c)

	a.stageHeadIndependence(c)

	a.cardNamed(c, "加权求和", func() {
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

	a.cardNamed(c, "残差流", func() {
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

	// The two analytical cards come after the stage's namesake figure, not before it.
	// They used to lead the stage, which measured as: the residual-stream card — the
	// one the stage is named for — started at y=1071 in a 720-DIP window, entirely
	// below the fold, because two explanation cards were sitting on top of it. A
	// reader opening "残差 / LN" must see the residual stream first; the questions of
	// what the layers say and how much they rewrite are follow-ups.
	a.stageLogitLens(c)

	a.stageLayerRewrite(c)

	a.cardNamed(c, "LayerNorm", func() {
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
	})

	a.cardNamed(c, "残差为何不爆", func() {
		// This card is one callout with no heading, so its chip is the whole label a
		// reader gets; it names the question the callout answers rather than the
		// misconception it rebuts, since the row is a table of contents.
		//
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

// stageLogitLens answers a question the residual stage raises but never answers:
// the stream accumulates a vector, so what does the model *say* with it, at each
// depth?
//
// This is the logit lens: apply the final LayerNorm and the tied unembedding to a
// row that is not the final row. It is a read-out trick, not a computation the
// network performs, and the measured result on this model is that it is a *bad*
// way to explain the network: of 180 intermediate rungs across the prompt, 153
// disagree with the final answer and 105 of those are more than 90% sure of
// themselves. So the card's job is not "watch the prediction improve" — it is to
// show that a confident intermediate answer means nothing, which is the same lesson
// the output stage's "97% is not correctness" callout makes from the other side.
func (a *App) stageLogitLens(c *ui.Context) {
	tr := a.Trace
	pos := clampIndex(a.SelToken, len(tr.Tokens))
	// At T=1, not at the reader's temperature. The temperature slider is scoped to
	// the output stage — dimmed everywhere else, with a tooltip saying which stages
	// read it — and TestEachControlChangesOnlyItsOwnStages enforces that. Reading a
	// diagnostic through the reader's temperature made the residual stage redraw when
	// the slider moved, which is the contract breaking rather than a nicety: a control
	// that "does nothing here" must not do something here.
	lens := computeLogitLens(a.W, tr, pos, 1.0)
	if !lens.OK {
		return
	}
	finalRung := lens.Rungs[len(lens.Rungs)-1]
	a.cardNamed(c, "每层说什么", func() {
		sectionTitle(c, "每一层会说什么", fmt.Sprintf("在位置 %d 上逐层读出预测", pos))
		// No markdown in this app: ui.Text draws the characters it is given.
		mutedText(c, "把网络最后一层的 LayerNorm 与反嵌入提前套用到中间某一层，就是「logit lens」——"+
			"它问的是「如果模型停在这里，它会说什么」。这不是模型真的算过的东西："+
			"网络只对最后一行做一次投影，中间这些答案没被训练过、也不参与输出。")
		// The bar is p(final answer) at each depth, not the top-1 probability and
		// not the entropy. Those two were the first thing drawn here and they made a
		// chart with no information in it: this model's intermediate distributions are
		// near one-hot (entropy 0.00 against a 5.55-nat ceiling), so every bar was a
		// hairline. p(final) is the quantity that actually moves, and the one that
		// answers the question — how much of the final answer is already here.
		rows := make([]barRow, 0, len(lens.Rungs))
		var maxP float32
		for _, rung := range lens.Rungs {
			if p := rung.Probs[lens.Final]; p > maxP {
				maxP = p
			}
		}
		for _, rung := range lens.Rungs {
			p := rung.Probs[lens.Final]
			label := rung.Name
			// One distinction, and the palette's own: colAccent2 means "this is what
			// the model actually produced", everything else is a read-out. The first
			// version also coloured rungs that happened to agree with the final answer
			// in colAccent, which was wrong twice over — the bar's length already says
			// how much of the final answer is there, so a rung can agree while showing
			// a short bar, and the colour had no legend to explain it.
			colour := colGreen
			if rung.IsFinal {
				colour = colAccent2
			}
			rows = append(rows, barRow{
				Label:      label,
				Value:      p,
				ValueText:  fmt.Sprintf("%.1f%%", 100*p),
				Color:      colour,
				LabelWidth: 96,
				ValueWidth: 56,
				Tip: fmt.Sprintf("%s：它对最终答案 %q 给了 %.2f%%，自己的答案是 %q（%.1f%%）%s",
					label, byteText(lens.Final), 100*p,
					byteText(rung.Top), 100*rung.Probs[rung.Top], agreeNote(rung)),
			})
		}
		bars(c, rows, maxP, colGreen)
		kv(c, "横条", fmt.Sprintf("每一层给最终答案 %q 的概率", byteText(lens.Final)))

		// Scoped to this position and said so: the callout below quotes the
		// whole-prompt figures, and the two must not read as disagreeing
		// measurements of one quantity.
		agreement := fmt.Sprintf("只看这个位置：%d / %d 个中间层说对了", countAgree(lens), lens.RungsCompared)
		if lens.ConfidentWrong > 0 {
			agreement = fmt.Sprintf("只看这个位置：%d / %d 个中间层「自信地错」——答案不是最终答案，"+
				"却对自己的答案有 90%% 以上的把握", lens.ConfidentWrong, lens.RungsCompared)
		}
		// The first working line used to read "最终答案 "␣"：输入向量 x 给了 0.00%",
		// which is a sentence about a probability that was never computed for that
		// row in a way a reader can use — the input vector's lens answer is a
		// different byte entirely, and 0.00% is the rounded residue of that. Say
		// what the first rung actually claims instead.
		// The last intermediate rung, which is the one a reader comparing "before the
		// final projection" with "the final output" will look at.
		lastIntermediate := lens.Rungs[len(lens.Rungs)-2]
		first := lens.Rungs[0]
		workingsLinesCopyable(c, []string{
			fmt.Sprintf("%s 的答案是 %q（它给最终答案 %.2f%%）",
				first.Name, byteText(first.Top), 100*first.Probs[lens.Final]),
			fmt.Sprintf("最后一个中间层「%s」给最终答案 %.2f%%，最终输出 %.2f%%",
				lastIntermediate.Name, 100*lastIntermediate.Probs[lens.Final],
				100*finalRung.Probs[lens.Final]),
			agreement,
		})
		// The whole-prompt figure, which is what makes this more than one position's
		// anecdote. Computed, not quoted: this sentence used to hardcode the default
		// prompt's numbers (92 of 180 rungs, all 36 positions), so a reader who shortened
		// the prompt was told about a ladder that was not the one drawn above it.
		summary := computePromptLensSummary(a.W, tr, 1.0)
		scope := fmt.Sprintf("整句 %d 个位置一共有 %d 个中间层", summary.Positions, summary.Rungs)
		if summary.OK && summary.Rungs > 0 {
			if summary.Disagree == 0 {
				misconception(c,
					"「logit lens 能说明模型是一层一层把答案想清楚的」",
					fmt.Sprintf("这句提示词的每一层都给了同一个答案：%s，没有一个与最终答案不同。"+
						"这不说明 lens 可靠——换一句提示词再看，中间层就会开始给出不同的答案。"+
						"结论要跟着数据走，而不是事先认定。", scope))
			} else {
				misconception(c,
					"「logit lens 能说明模型是一层一层把答案想清楚的」",
					fmt.Sprintf("它不能。把整句提示词的每一层都读一遍：%s，其中 %d 个的答案与最终答案不同，"+
						"有 %d 个对自己的错误答案有 90%% 以上的把握，%d 个位置里有 %d 个至少有一层如此。"+
						"所以中间层「很确定」不说明它和最终答案一致——这和输出阶段那条「97%% 不等于正确」"+
						"是同一件事的两个方向。lens 可以用来看某一层「偏向什么」，不能用来看它「知道什么」。",
						scope, summary.Disagree, summary.ConfidentWrong,
						summary.Positions, summary.PositionsWithConfidentWrong))
			}
		}
	})
}

// countAgree counts the intermediate rungs that already give the final answer.
func countAgree(lens logitLens) int {
	n := 0
	for _, r := range lens.Rungs {
		if !r.IsFinal && r.AgreesWithFinal {
			n++
		}
	}
	return n
}

// agreeNote says whether a rung agrees with the final answer, for the bar tooltip.
func agreeNote(r lensRung) string {
	switch {
	case r.IsFinal:
		return " · 这是模型真正的输出"
	case r.AgreesWithFinal:
		return " · 与最终答案一致"
	}
	return " · 与最终答案不同"
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
		// Bracketed on both signs, through the same helper the embedding sum uses: the
		// residual chain's two lines then share one shape instead of one bracketing its
		// negative term and the other printing its positive term bare.
		term := signed(s.Term)
		lines = append(lines, fmt.Sprintf("%+.4f %s %s = %+.4f", running, s.Label, term, s.Sum))
		running = s.Sum
	}
	next := "送往下一层"
	if isLast {
		next = "送入最终 LayerNorm"
	}
	lines = append(lines, fmt.Sprintf("每一次加法都不覆盖原值，所以层 %d 的输出完整%s", layer, next))
	workingsLinesCopyable(c, lines)
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
	workingsLinesCopyable(c, []string{
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
	a.cardNamed(c, "FFN 内部", func() {
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

	// After the stage's own figure. Written above it, this card's header started at
	// y=667 in a 720-DIP window: the FFN heading fit but every row it introduces was
	// below the fold, so the stage opened on a chart about the FFN rather than the FFN.
	a.stageFFNAnswerDirection(c)

	a.stageFFNConcentration(c)
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
	workingsLinesCopyable(c, []string{
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

	a.cardNamed(c, "最后位置", func() {
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

	a.stageNextByteSkill(c)

	a.stageWeightTying(c)

	a.cardNamed(c, "Top-12", func() {
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

	a.cardNamed(c, "采样", func() {
		sectionTitle(c, "采样", "从分布里抽一个 token")
		ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
			if ui.PrimaryButton(c, "采样并续写").Clicked() {
				a.sampleNext()
			}
			// The undo button is in the top bar, not here: this card is on one stage
			// and below the fold at 1180×720, so a button here left Ctrl+Z with no
			// clickable path on six of the seven stages. What stays on this card is
			// the keyboard hint, next to the sampling it usually takes back.
			if a.undo != nil {
				ui.Text(c, "Ctrl+Z 撤销「"+a.undo.what+"」").FontSize(fontMicro).TextColor(colFaint).Shrink(0)
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
			workingsLinesCopyable(c, []string{
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
		// Ctrl+Z is listed only when there is a sample to take back, for the same
		// reason the card's undo button only appears then: advertising a key that
		// currently does nothing teaches the reader the app ignores them.
		undo := ""
		if a.undo != nil {
			// Named from the recorded action, like the card's button: Ctrl+Z no longer
			// only takes back a sample.
			undo = " · Ctrl+Z 撤销「" + a.undo.what + "」"
		}
		// The precedence note is here because the line below it is unconditional and
		// a focused control is not. Measured: with 重置 focused, Enter activates the
		// button rather than sampling; with a step button focused, Space activates it
		// rather than playing. That is the right precedence — a button under the
		// keyboard is what the reader most recently chose to operate — but a reader who
		// tabbed to a control and then found Enter doing something else needs to be told
		// why, and the line that advertises Enter is where they will look.
		ui.Text(c, fmt.Sprintf("快捷键：← → 切换阶段 · %s · 空格 播放/暂停 · Enter 采样%s · %s 跳转 · PgUp/PgDn 滚动",
			shift, undo, stageDigitRange())).FontSize(fontMicro).TextColor(colFaint)
		ui.Text(c, "（用 Tab 选中某个按钮时，空格与 Enter 先作用于那个按钮——例如「重置」会被直接按下。）").
			FontSize(fontMicro).TextColor(colFaint).LineHeight(1.35)
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
		// "top-5" read as "the five keys this query attends to most", which is the opposite of what the
		// card shows: topAbsBars picks the largest MAGNITUDE, so a very negative score — the kind softmax
		// all but erases — makes the cut while a modest positive one does not. Name the rule the card
		// actually uses.
		sectionTitle(c, "绝对值最大的 5 个分数", fmt.Sprintf("query = token %d", i))
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

// nextByteSkillIntro is the opening paragraph of the per-position card, shared by
// the card and by the one-token explanation so a reader always meets the idea
// before the state that has no data for it.
const nextByteSkillIntro = "模型不只在提示词末尾预测下一个字。因果掩码让位置 t 只看得到 t 及其之前的内容，" +
	"所以每个位置都能给出一个诚实的下一字节猜测——而提示词里就写着正确答案。" +
	"每一格是一个位置：绿色是猜对了，暖色是猜错了，颜色更深的错得更离谱。"

// stageNextByteSkill shows that this model predicts a next byte at every
// position, not only at the end of the prompt — and how often it is wrong.
//
// The output stage used to end on one row: the last position's distribution.
// That is the row the sampler reads, so it is the right thing to put on top, but
// it is one of T predictions the model makes and the only one with no answer to
// check against — the prompt ends there. Every other row has one: the byte the
// reader typed next. The trace already holds them (Trace.FinalNorm is T × d_model
// and Forward projects only the last row), so this card costs one unembedding per
// position and turns "this is a next-token predictor" from a claim into a score.
//
// Measured on the bundled weights and the default prompt: top-1 correct 24/35,
// mean loss 1.121 nats against ln(256) = 5.545 for guessing uniformly. The loss is
// the more informative figure — 69% accuracy sounds impressive until the per-row
// detail shows the misses are mostly the corpus's own words.
func (a *App) stageNextByteSkill(c *ui.Context) {
	tr := a.Trace
	skill := computeNextByteSkill(a.W, tr, a.Temp)
	if skill.Scored <= 0 {
		// One token, nothing to check against. Say why the card is empty rather
		// than leaving the stage silent: this is the reader's first prompt if they
		// typed a single character, and a missing card in a lesson that is all
		// about "every position predicts" reads like a bug in the app.
		a.cardNamed(c, "每位置预测", func() {
			sectionTitle(c, "每个位置都在预测", "这个提示词只有一个位置")
			mutedText(c, nextByteSkillIntro)
			mutedText(c, "只是这个提示词只有一个位置，而一次预测要拿提示词里真实的下一个字节来核对——"+
				"位置 0 之后没有字节了。多打一个字，这张卡片就会变成逐位置的图。")
			kv(c, "可以核对的位置", "0（需要至少两个 token）")
			a.causalInvarianceLine(c)
		})
		return
	}
	a.cardNamed(c, "每位置预测", func() {
		sectionTitle(c, "每个位置都在预测", fmt.Sprintf("T − 1 = %d 个可核对的位置", skill.Scored))
		mutedText(c, nextByteSkillIntro)
		a.predictionStrip(c, skill)
		a.nextByteSkillSummary(c, tr, skill)
		a.skillBreakdownLine(c, tr, skill)
		// After the accuracy figures, not before them: the check is about what the strip
		// can *rely* on, and a reader meets it as the answer to "and are these predictions
		// stable?" rather than as an unexplained count of positions.
		a.causalInvarianceLine(c)
		// One legend line, not four: the counts it would carry are already in the
		// summary above, and on this stage every row of chrome pushes the sampling
		// card's own button out of the window. What the strip's colours and its
		// interaction need is one sentence, not a colour key that repeats numbers.
		ui.Row(c).Gap(10).Wrap().AlignItems(ui.Center).Children(func() {
			ui.Row(c).Gap(5).AlignItems(ui.Center).Children(func() {
				ui.Box(c).Size(9, 9).Radius(2).Background(colGreen.Alpha(0.75))
				ui.Text(c, "猜对").FontSize(fontMicro).TextColor(colMuted)
			})
			ui.Row(c).Gap(5).AlignItems(ui.Center).Children(func() {
				ui.Box(c).Size(9, 9).Radius(2).Background(colWarm.Alpha(0.75))
				ui.Text(c, "猜错").FontSize(fontMicro).TextColor(colMuted)
			})
			ui.Row(c).Gap(5).AlignItems(ui.Center).Children(func() {
				ui.Box(c).Size(9, 9).Radius(2).Background(colWarm)
				ui.Text(c, "更深 = 猜得离谱（−ln p > 4）").FontSize(fontMicro).TextColor(colMuted)
			})
			ui.Row(c).Gap(5).AlignItems(ui.Center).Children(func() {
				ui.Box(c).Size(9, 9).Radius(2).Border(1.5, colAccent2)
				ui.Text(c, "点击固定，Esc 解除").FontSize(fontMicro).TextColor(colFaint)
			})
		})
	})
}

// nextByteSkillSummary is the sentence under the per-position block: the two
// aggregate figures with their reference points.
func (a *App) nextByteSkillSummary(c *ui.Context, tr *model.Trace, skill nextByteSkill) {
	if skill.Scored <= 0 {
		return
	}
	pct := 100 * float64(skill.Hits) / float64(skill.Scored)
	// ln(vocab) is the loss of a model that puts 1/256 on every byte — the
	// "no better than guessing" line. Quoting 1.12 nats without it leaves the
	// reader unable to tell a good number from a bad one, the same problem the
	// entropy line above has and solves the same way.
	saved := 100 * (1 - skill.MeanLoss/skill.UniformReference)
	// One line, not a three-line workings block: this card sits above the sampling
	// card on the tallest stage, and each extra line here pushes that card's own
	// button further past the window's bottom edge. The arithmetic is still all
	// present — the total, the mean and the reference it beats — it just reads as
	// a sentence rather than a derivation, because the derivation is visible in
	// the individual positions the inspector shows.
	summary := fmt.Sprintf("猜对 %d / %d（%.0f%%）· 平均损失 %.3f nats · 均匀猜测 ln(%d) = %.3f → 降低了 %.0f%%",
		skill.Hits, skill.Scored, pct, skill.MeanLoss,
		int(math.Round(math.Exp(skill.UniformReference))), skill.UniformReference, saved)
	if skill.Underflowed > 0 {
		// Saying "平均损失 +Inf nats" reads as a fault in the app. The cause is arithmetic:
		// the true byte's probability rounded to zero, so its loss has no finite value.
		summary += fmt.Sprintf("（%d 个位置的真实字节概率下溢到 0，未计入均值）", skill.Underflowed)
	}
	workingsLinesCopyable(c, []string{summary})
	// The loss is computed at the reader's temperature, and the curve is not a
	// slope: it is U-shaped with its minimum at T=1, the value the weights were
	// trained at. Measured on the bundled weights and the default prompt, mean
	// loss runs 6.392 (T=0.1), 1.510 (T=0.5), 1.121 (T=1.0), 1.246 (T=2.0). That
	// shape is the useful lesson and the card states it whether or not the reader
	// has moved the slider, because the number on screen is about to move and
	// nothing else on the stage says why.
	losses := fmt.Sprintf("%.3f", skill.MeanLoss)
	if skill.Temperature != 1 {
		// Quote the T=1 figure beside the current one: the reader needs to know
		// which way their slider moved this number and by how much.
		atOne := computeNextByteSkill(a.W, tr, 1.0).MeanLoss
		losses = fmt.Sprintf("%.3f（T=1 时是 %.3f）", skill.MeanLoss, atOne)
	}
	// Which temperature minimises the loss is measured, not asserted. This sentence used to
	// say in fixed prose that the minimum is at T=1 "the value the weights were trained at" —
	// true for the default prompt when it was written, and false for most prompts: measured,
	// "attention is all you need" bottoms out at 1.5 and "hi" at 2.0, and even the default
	// prompt's minimum is at 1.25 rather than exactly 1.0.
	curve := computeTemperatureCurve(a.W, tr)
	if curve.OK {
		shape := fmt.Sprintf("当前损失 %s。这条曲线是实测的：在本句提示词上，平均损失最低在 T = %.2f（%.3f nats），T = 1 时是 %.3f。",
			losses, curve.Best, curve.BestLoss, curve.AtOne)
		if math.Abs(curve.Best-1) < 1e-9 {
			shape += "权重就是训练在 T = 1 那个标度上，所以最低点落在那里。"
		} else if curve.Best > 1 {
			// Only the measurement is stated. An earlier draft added "说明真值字节不是模型的首选",
			// which is a plausible cause but not something this card measures — the card can see
			// the curve, not why it bends.
			shape += "最低点不在 T = 1，而在更暖的一侧：把分布压得更尖（往冷调）会让损失变大。"
			if math.Abs(curve.Best-2.0) < 1e-9 {
				shape += "滑块到顶是 2.0，这条提示词的曲线到顶仍未回头。"
			}
		} else {
			shape += "最低点在更冷的一侧：模型对首选字节的把握比损失数字看上去的更大。"
		}
		shape += "往两端调损失都变大，但那是温度的标度，不是模型变差（argmax 一个都不会变）。"
		ui.Text(c, shape).FontSize(fontMicro).TextColor(colFaint).LineHeight(1.4)
	} else {
		ui.Text(c, fmt.Sprintf("当前损失 %s。往两端调损失都变大，但那是温度的标度，不是模型变差（argmax 一个都不会变）。", losses)).
			FontSize(fontMicro).TextColor(colFaint).LineHeight(1.4)
	}
}

// inspectorPrediction is the output stage's second inspector card: one position's
// prediction in full — what the model guessed, what the prompt actually had, and
// the probability it gave the truth, so the loss below can be re-derived by hand.
//
// It lives in the inspector rather than in the stage card because that is where
// this app puts "one element of the figure, in full": the attention cell, the
// token's bytes and the LayerNorm dimension all do the same. The strip itself
// stays in the stage column, where it is wide enough to be a figure — 35 cells in
// a 330-DIP panel would be 8 DIP each.
func (a *App) inspectorPrediction(c *ui.Context) {
	tr := a.Trace
	// Two tokens are the minimum for a prediction to have an answer to check.
	if len(tr.Tokens) < 2 {
		return
	}
	skill := computeNextByteSkill(a.W, tr, a.Temp)
	i := a.predictionCursor(skill)
	if i < 0 || i >= len(skill.Rows) {
		return
	}
	row := skill.Rows[i]
	source := "损失最大的位置"
	switch {
	case a.PredSel >= 0:
		source = "已固定 · Esc 或再点一次解除"
	case a.PredHover >= 0:
		source = "指针所在的位置"
	}
	card(c, func() {
		sectionTitle(c, fmt.Sprintf("t=%d 的预测", row.Pos), source)
		// The inspector is 330 DIP wide and its card is padded, so the label column
		// has to stay narrow enough to leave the values room on one line. The labels
		// are worded for that width rather than shortened at render time: "实际是"
		// and "给答案的概率" say the whole thing in 6 characters, where "实际下一个
		// 字节" wrapped its own value onto the next line.
		const labelW = 76
		kvW(c, "模型猜", fmt.Sprintf("%q（p = %.4f）", byteText(row.Guess), row.Probs[row.Guess]), labelW)
		kvW(c, "实际是", fmt.Sprintf("%q", byteText(row.Actual)), labelW)
		kvW(c, "给答案的概率", fmt.Sprintf("%.4f", row.Probs[row.Actual]), labelW)
		kvW(c, "损失", fmt.Sprintf("−ln(%.4f) = %.3f nats", row.Probs[row.Actual], row.Loss), labelW)
		// The premise of the whole card, checked rather than asserted: the causal
		// mask means row t was computed without ever seeing token t+1.
		//
		// Both indices are zero-based and named as such. The app numbers tokens
		// and positions from 0 everywhere — the strip, the heatmap axes, "token
		// #35 / 35" — and the first draft of this sentence mixed the two bases,
		// so a panel headed t=8 said it compared "第 9 个字节" while comparing
		// token 9, which is the tenth.
		mutedText(c, fmt.Sprintf("位置 %d 的计算里没有任何来自位置 %d 及之后的信息"+
			"（因果掩码在第 %d 个位置截断），所以拿 token %d（%q）与它比对，"+
			"是在核对模型真的做过的一次预测。",
			row.Pos, row.Pos+1, row.Pos, row.Pos+1, byteText(row.Actual)))
	})
}

// stageHeadProfiles compares what the four heads of this layer do over the whole
// prompt, which is the question the lesson asserts an answer to and never showed.
//
// The inspector already lists one row's entropy per head, which is four numbers about
// a single query: two heads can tie there while looking at completely different
// places. These are the two behaviours actually visible in this model's matrices — how
// much a row puts on the token immediately before it, and how many rows concentrate
// more than half their weight on one key.
//
// The measured result is worth a card because it is uneven rather than uniform: on the
// bundled weights and the default prompt, layer 0's previous-token share runs
// 0.66 / 0.22 / 0.37 / 0.61 across its four heads, while layer 1's four sit at
// 0.12–0.14 — indistinguishable. So "each head learns a different pattern" is true of
// the first layer and false of the second, and a reader switching the head control on
// layer 1 would otherwise be looking for a difference that is not there.
func (a *App) stageHeadProfiles(c *ui.Context) {
	l := a.layer()
	profiles := computeHeadProfiles(*l, len(a.Trace.Tokens))
	if len(profiles) == 0 {
		return
	}
	a.cardNamed(c, "四个头", func() {
		sectionTitle(c, "四个头各自在做什么", fmt.Sprintf("层 %d · 整句 %d 行的平均", a.Layer, profiles[0].Rows))
		// No markdown in this app: ui.Text draws the characters it is given.
		mutedText(c, "检查器里那张「这一行的统计」只描述一行：两个头可以在同一行上有相同的熵，"+
			"却在看完全不同的地方。这里换两个能区分行为的量——行内有多少权重落在紧邻的前一个 token，"+
			"以及有多少行把一半以上的权重压在同一个 key 上。")
		// Bars share one axis across the heads so the comparison is a shape.
		rows := make([]barRow, 0, len(profiles))
		var maxPrev float64
		for _, p := range profiles {
			if p.PrevShare > maxPrev {
				maxPrev = p.PrevShare
			}
		}
		for _, p := range profiles {
			label := fmt.Sprintf("H%d", p.Head)
			colour := colGreen
			if p.Head == a.Head {
				label = fmt.Sprintf("H%d ◂", p.Head)
				colour = colAccent2
			}
			rows = append(rows, barRow{
				Label:      label,
				Value:      float32(p.PrevShare),
				ValueText:  fmt.Sprintf("%.0f%%", 100*p.PrevShare),
				Color:      colour,
				LabelWidth: 40,
				ValueWidth: 40,
				Tip: fmt.Sprintf("头 %d：平均 %.0f%% 的权重给紧邻的前一个 token · %d / %d 行把半数以上押在同一个 key · 平均熵 %.2f / %.2f nats",
					p.Head, 100*p.PrevShare, p.PeakRows, p.Rows, p.MeanEntropy, p.EntropyCeiling),
			})
		}
		ui.Text(c, "每行给「紧邻的前一个 token」的平均权重").FontSize(fontMicro).TextColor(colFaint)
		bars(c, rows, float32(maxPrev), colGreen)
		// The spread is the thing worth naming: four heads doing one thing, or four
		// heads doing four things, are different pictures and the bars show which.
		lo, hi := profiles[0].PrevShare, profiles[0].PrevShare
		for _, p := range profiles[1:] {
			if p.PrevShare < lo {
				lo = p.PrevShare
			}
			if p.PrevShare > hi {
				hi = p.PrevShare
			}
		}
		spread := hi - lo
		// The advice names a layer to switch to, so it has to know which layer the reader is
		// already on. Saying "换到第 0 层" while showing layer 0 asks them to go where they
		// already are — measured on prompt "ab", whose layer 0 spread is 75%-92%, comfortably
		// under the 0.2 threshold that selects this branch. On the first layer there is no
		// further layer to point at, so the card says what the flat spread means instead of
		// sending the reader somewhere.
		var verdict string
		if a.Layer == 0 {
			verdict = fmt.Sprintf("四个头在这一层上很接近（%.0f%%–%.0f%%），看不出分工。"+
				"第 0 层就是第一层，再往下没有可比的层——所以这里不是「还没看够」，"+
				"而是这一层的四个头确实在做同一件事的不同强度版本", 100*lo, 100*hi)
		} else {
			verdict = fmt.Sprintf("四个头在这一层上很接近（%.0f%%–%.0f%%），看不出分工——"+
				"换到第 0 层看看，那里的头差别明显得多", 100*lo, 100*hi)
		}
		if spread >= 0.2 {
			verdict = fmt.Sprintf("四个头差别明显：%.0f%%–%.0f%%，也就是有的头主要在盯前一个 token、"+
				"有的在整句里找别的位置——这就是「每个头学到不同的模式」在这份权重里的具体样子",
				100*lo, 100*hi)
		}
		workingsLinesCopyable(c, []string{
			fmt.Sprintf("前一位权重：H0 %.0f%% · H1 %.0f%% · H2 %.0f%% · H3 %.0f%%",
				100*profiles[0].PrevShare, 100*profiles[1].PrevShare,
				100*profiles[2].PrevShare, 100*profiles[3].PrevShare),
			fmt.Sprintf("押注超过半数的行：H0 %d · H1 %d · H2 %d · H3 %d（共 %d 行）",
				profiles[0].PeakRows, profiles[1].PeakRows,
				profiles[2].PeakRows, profiles[3].PeakRows, profiles[0].Rows),
			verdict,
		})
	})
}

// skillBreakdownLine puts the per-position accuracy beside the breakdown that explains
// it, because the overall figure flatters this model.
//
// Measured across three prompts: positions whose predecessor is a space are right 4
// times in 18, while everything else is right 73 in 96. On the default prompt the mean
// loss is 2.87 nats after a space against 0.83 elsewhere. The cause is structural — a
// space is a word boundary, so the next byte starts a new word, and 61 training
// sentences do not teach which word comes next. It does know how to continue a word it
// is already inside.
func (a *App) skillBreakdownLine(c *ui.Context, tr *model.Trace, skill nextByteSkill) {
	b := computeSkillBreakdown(tr, skill)
	if !b.OK {
		return
	}
	pct := func(hits, rows int) string {
		if rows == 0 {
			return "—"
		}
		return fmt.Sprintf("%d/%d（%.0f%%）", hits, rows, 100*float64(hits)/float64(rows))
	}
	// Text colours, not data colours. colGreen means "a value in a chart" everywhere
	// else in this app, and a prose label wearing it reads as a plotted value;
	// TestDataColoursStayOutOfText enforces that, and it caught this.
	ui.Row(c).Gap(16).Wrap().AlignItems(ui.Center).Children(func() {
		ui.Text(c, "前一个是空格（词边界）"+pct(b.AfterSpaceHits, b.AfterSpaceRows)).
			FontSize(fontMicro).TextColor(colMuted)
		ui.Text(c, "其余位置 "+pct(b.OtherHits, b.OtherRows)).
			FontSize(fontMicro).TextColor(colMuted)
		ui.Text(c, fmt.Sprintf("平均损失 %.2f / %.2f nats", b.AfterSpaceLoss, b.OtherLoss)).
			FontSize(fontMicro).TextColor(colFaint)
	})
	// Only worth the sentence when the gap is actually there; a prompt planted with
	// spaces in unusual places could invert it, and the claim has to follow.
	if b.AfterSpaceLoss > 1.5*b.OtherLoss {
		mutedText(c, "错得最多的地方是词边界：空格之后是一个新词的开头，而这个模型只读过 61 句关于 "+
			"transformer 的英文，它知道一个词内部字母怎么接下去，不知道下一个词是什么。")
	}
}

// stageLayerRewrite compares how much each block changes the residual stream, which
// is the layer-level counterpart to the four-head comparison on the Softmax stage.
//
// The app has always let a reader switch layers, and every figure redraws — but
// nothing said whether the two layers are doing comparable work. They are not.
// Measured on the bundled weights: layer 0 leaves the stream almost unrelated to what
// entered it (cosine 0.22–0.24 across three prompts) and writes 2.8–3.2 times its
// input length, while layer 1 keeps the direction (0.92–0.93) and adds 0.41–0.45. Most
// of the transformation happens in the first block, and the second refines it.
//
// That is the same shape the logit lens shows from the other side: the prediction is
// already there after layer 0's FFN, and layer 1 sharpens it.
func (a *App) stageLayerRewrite(c *ui.Context) {
	layers := computeLayerRewrite(a.Trace)
	if len(layers) < 2 {
		return
	}
	a.cardNamed(c, "每层改多少", func() {
		sectionTitle(c, "每一层改写了多少", "进入 / 离开这一层的残差流有多像")
		mutedText(c, "上面那张阶梯图看的是每一层「说什么」，这里看每一层「改多少」。"+
			"余弦接近 1 表示这一层基本只是把原向量传下去；接近 0 表示离开时的方向与进来时几乎无关。"+
			"比值是这一层写进流里的长度除以它读到的长度——比值大，说明这一层在大量新增内容而不是微调。")
		rows := make([]barRow, 0, len(layers))
		var maxRatio float64
		for _, l := range layers {
			if l.FFNRatio > maxRatio {
				maxRatio = l.FFNRatio
			}
		}
		for _, l := range layers {
			label := fmt.Sprintf("层 %d", l.Layer)
			colour := colGreen
			if l.Layer == a.Layer {
				label = fmt.Sprintf("层 %d ◂", l.Layer)
				colour = colAccent2
			}
			rows = append(rows, barRow{
				Label:      label,
				Value:      float32(l.FFNRatio),
				ValueText:  fmt.Sprintf("‖ΔFFN‖/‖x‖ = %.2f", l.FFNRatio),
				Color:      colour,
				LabelWidth: 46,
				ValueWidth: 112,
				Tip: fmt.Sprintf("层 %d：进出余弦 %.3f · ΔFFN 长度是输入的 %.2f 倍 · "+
					"Δattn 是 %.2f 倍 · ‖x‖ %.1f → %.1f",
					l.Layer, l.CosineInOut, l.FFNRatio, l.AttnRatio, l.InNorm, l.OutNorm),
			})
		}
		bars(c, rows, float32(maxRatio), colGreen)
		first, last := layers[0], layers[len(layers)-1]
		workingsLinesCopyable(c, []string{
			fmt.Sprintf("层 %d：进出余弦 %.3f，ΔFFN 是输入的 %.2f 倍（‖x‖ %.1f → %.1f）",
				first.Layer, first.CosineInOut, first.FFNRatio, first.InNorm, first.OutNorm),
			fmt.Sprintf("层 %d：进出余弦 %.3f，ΔFFN 是输入的 %.2f 倍（‖x‖ %.1f → %.1f）",
				last.Layer, last.CosineInOut, last.FFNRatio, last.InNorm, last.OutNorm),
			fmt.Sprintf("前一层改写了方向，后一层基本保持了方向——%d 层模型里大部分变换发生在第一层",
				len(layers)),
		})
		misconception(c,
			"「每一层做同样多的活，层数加倍的模型就是把同一个变换做两遍」",
			fmt.Sprintf("两层做的量差得很远。层 %d 离开时的方向与进入时几乎无关（余弦 %.2f），"+
				"写进去的内容是它读到的 %.1f 倍；层 %d 的余弦是 %.2f，只加了 %.2f 倍。"+
				"这也解释了上面那张阶梯：预测在层 0 的 FFN 之后就已经成形，层 1 是在把它磨尖。"+
				"所以「加一层」不等于「把同一件事再做一遍」——每一层的活是训练出来的，各不相同。",
				first.Layer, first.CosineInOut, first.FFNRatio,
				last.Layer, last.CosineInOut, last.FFNRatio))
	})
}

// stageCards names, per stage, the cards a reader can jump to from the step bar.
//
// The row exists because the stages outgrew the viewport: measured at 1180×720, step 5
// is 4.4 screens tall and step 4 is 3.3, and the only way to find out what came next was
// to scroll for it. The chips make a stage's structure visible at a glance — four cards
// on the residual stage, one on the FFN stage — and each is one click from its card.
//
// The entries must match the chip labels passed to cardNamed, in the order the cards are
// drawn. TestCardRowsMatchTheRenderedCards checks both, so a card cannot be added
// without appearing in the row or renamed without this list going stale.
//
// The lesson card is not listed: it is the stage's own text, at the end of every stage,
// and a chip for it would be a chip for "the rest of the page".
var stageCards = [][]string{
	{"token 序列", "字符 → 字节", "数据流与形状"},
	{"嵌入向量", "嵌入有含义吗", "位置向量", "位置编码的形状"},
	// The mask counterfactual is drawn after the score matrix (measured: matrix 400,
	// counterfactual 930), so the row lists them in that order.
	{"Q / K / V", "分数矩阵", "掩码反事实"},
	{"注意力矩阵", "四个头", "头之间重合吗", "加权求和"},
	// Order must match the draw order: the row is a table of contents, and a chip whose
	// position disagrees with its card's position is worse than no row. Measured from the
	// scroll each chip produces: 残差流 93, 每层说什么 665, 每层改多少 1133, LayerNorm 1460,
	// 残差为何不爆 1767. The registry had the last three in a different order than the page.
	{"残差流", "每层说什么", "每层改多少", "LayerNorm", "残差为何不爆"},
	// The stage's own FFN figure comes first (measured at 93 against 664 for the
	// concentration card, which was placed after it last round for exactly this reason).
	{"FFN 内部", "FFN 写出了答案", "哪个单元在干活"},
	{"最后位置", "每位置预测", "绑定权重", "Top-12", "采样"},
}

// stageFFNConcentration answers whether the same hidden units matter everywhere, which
// the FFN stage raised and never addressed: it draws one row's 384 units, so a reader
// sees which units fire at the selected position and nothing about the other 35.
//
// The two layers answer differently, and the difference is large. Measured on the
// bundled weights with the default prompt:
//
//	layer 0   130 of 384 units carry half the total activation; 0 units are on at every
//	          position — the work is spread across many, and which ones depends on where
//	layer 1    28 of 384 units carry half of it; 17 units are positive at every position
//	          — a small set firing everywhere
//
// Stable across prompts (layer 0 needs 118–130, layer 1 needs 28–35). Read beside the
// residual stage's "每一层改写了多少", the two say the same thing from two directions:
// the first block transforms broadly, the second applies a narrow, repeated correction.
func (a *App) stageFFNConcentration(c *ui.Context) {
	// Two tokens minimum, not one. With a single position "on at every position" is
	// vacuous — it just means "on at that one" — and with no other position to compare
	// against, "which units matter where" has no answer. Measured on the one-token prompt
	// "a": the card reported 228 of 384 units always on and a 2.9x concentration gap
	// between the layers, figures that describe the prompt's length rather than the
	// model. Drawing nothing is the honest response.
	if len(a.Trace.Tokens) < 2 {
		return
	}
	rows := computeAllFFNConcentration(a.Trace)
	if len(rows) < 2 {
		return
	}
	a.cardNamed(c, "哪个单元在干活", func() {
		sectionTitle(c, "单元是怎么分工的", "整句所有位置合起来看")
		mutedText(c, "上面那条色带只画了选中位置的 384 个单元。把整句每个位置都算进来，"+
			"要看的是两个数：多少个单元才凑够一半的激活量，以及有没有单元在每个位置都开着。")
		barsOut := make([]barRow, 0, len(rows))
		var maxHalf float64
		for _, r := range rows {
			if float64(r.HalfMass) > maxHalf {
				maxHalf = float64(r.HalfMass)
			}
		}
		for _, r := range rows {
			label := fmt.Sprintf("层 %d", r.Layer)
			colour := colGreen
			if r.Layer == a.Layer {
				label = fmt.Sprintf("层 %d ◂", r.Layer)
				colour = colAccent2
			}
			barsOut = append(barsOut, barRow{
				Label:      label,
				Value:      float32(r.HalfMass),
				ValueText:  fmt.Sprintf("%d / %d 个", r.HalfMass, r.Units),
				Color:      colour,
				LabelWidth: 46,
				ValueWidth: 96,
				Tip: fmt.Sprintf("层 %d：%d 个单元凑够一半激活 · 每位置都开 %d 个 · 从不打开 %d 个 · 正激活占 %.0f%%",
					r.Layer, r.HalfMass, r.AlwaysOn, r.NeverOn, 100*r.PositiveShare),
			})
		}
		ui.Text(c, "凑够一半激活量需要多少个单元（越少越集中）").
			FontSize(fontMicro).TextColor(colFaint)
		bars(c, barsOut, float32(maxHalf), colGreen)
		first, last := rows[0], rows[len(rows)-1]
		workingsLinesCopyable(c, []string{
			fmt.Sprintf("层 %d：%d 个单元凑够一半激活，每位置都开的有 %d 个，从不打开 %d 个",
				first.Layer, first.HalfMass, first.AlwaysOn, first.NeverOn),
			fmt.Sprintf("层 %d：%d 个单元凑够一半激活，每位置都开的有 %d 个，从不打开 %d 个",
				last.Layer, last.HalfMass, last.AlwaysOn, last.NeverOn),
			fmt.Sprintf("两层都是 384 个单元，但集中程度差了 %.1f 倍",
				float64(first.HalfMass)/math.Max(1, float64(last.HalfMass))),
		})
		if first.HalfMass > last.HalfMass*2 {
			misconception(c,
				"「每一层的 FFN 都是同一台特征提取机器，换个层只是换套权重」",
				fmt.Sprintf("两层的用法很不一样。层 %d 要把 %d 个单元摊开才凑够一半的激活量，"+
					"而且没有任何单元在整句每个位置都开着——它按位置调动不同的单元。"+
					"层 %d 只用 %d 个单元就够了，其中 %d 个从头开到尾；还有 %d 个从头到尾没开过。"+
					"这是「前一层广泛改写、后一层做窄而重复的修正」在 FFN 里的样子，"+
					"与第 5 步那张「每一层改写了多少」说的是同一件事。",
					first.Layer, first.HalfMass, last.Layer, last.HalfMass,
					last.AlwaysOn, last.NeverOn))
		}
	})
}

// stageWeightTying shows what the tied output matrix means for a prediction, which the
// output stage asserts ("权重绑定") without ever demonstrating.
//
// With the tying, byte b's logit is exactly <E[b], x>, so it factors into three named
// quantities: the byte's embedding length, the final vector's length, and the cosine
// between them. The card shows that product for the model's top byte — measured, 9.973 ×
// 9.351 × 0.2763 = 25.770, which is the logit itself — and then shows which of the three
// actually decides the ranking.
//
// The answer is the cosine. Across the 256 bytes the embedding lengths span only 8.29 to
// 11.71, and their correlation with the logit is −0.000, while the cosine runs −0.324 to
// +0.276 and correlates +0.997. So "the model prefers the bytes whose input embedding
// points the same way as the final vector" is not a metaphor here; it is the arithmetic,
// and only 563 of the 32640 byte pairs order differently by logit than by cosine.
func (a *App) stageWeightTying(c *ui.Context) {
	p := computeWeightTying(a.W, a.Trace, 1.0)
	if !p.OK {
		return
	}
	a.cardNamed(c, "绑定权重", func() {
		sectionTitle(c, "权重绑定让 logit 变成一个余弦", "同一个矩阵读入也读出")
		mutedText(c, "输出矩阵就是嵌入矩阵本身，所以字节 b 的 logit 恰好等于它和最终向量的内积，而内积又能拆成三个量相乘。"+
			"这张卡片拆开模型选出那个字节的分数，再看三个量里究竟是哪一个在决定名次。")
		kvW(c, "模型选的字节", fmt.Sprintf("%q（第 1 / %d 名，p = %.1f%%）",
			rune(p.Byte), p.Total, 100*p.Probability), 108)
		kvW(c, "logit", fmt.Sprintf("%.3f", p.Logit), 108)
		kvW(c, "= ‖E[b]‖ × ‖x‖ × cos", fmt.Sprintf("%.3f × %.3f × %.4f", p.EmbedNorm, p.FinalNorm, p.Cosine), 108)
		kvW(c, "三者相乘", fmt.Sprintf("%.3f（与上面的 logit 一致）", p.EmbedNorm*p.FinalNorm*p.Cosine), 108)
		// Which factor decides the ordering. Two bars on one axis would mislead here —
		// they are not the same kind of quantity — so state each and let the numbers speak.
		workingsLinesCopyable(c, []string{
			fmt.Sprintf("内积 = ‖E[b]‖ × ‖x‖ × cos = %.3f × %.3f × %.4f = %.3f",
				p.EmbedNorm, p.FinalNorm, p.Cosine, p.Logit),
			fmt.Sprintf("‖E[b]‖ 在 %d 个字节上只从 %.2f 变到 %.2f，与 logit 的相关是 %+.3f",
				p.Total, p.EmbedNormMin, p.EmbedNormMax, p.LogitNormCorrelation),
			fmt.Sprintf("cos 从 %+.3f 变到 %+.3f，与 logit 的相关是 %+.3f——名次基本由它决定",
				p.CosineMin, p.CosineMax, p.LogitCosineCorrelation),
			fmt.Sprintf("按 logit 排和按 cos 排不一样的字节对：%d / %d（%.1f%%）",
				p.PairsOrderedDifferently, p.Pairs, 100*float64(p.PairsOrderedDifferently)/float64(p.Pairs)),
		})
		misconception(c,
			"「权重绑定只是为了省参数，对模型的预测没有影响」",
			fmt.Sprintf("省参数是它的动机，但它同时决定了预测的形状：logit 就是内积，内积就是长度乘余弦。"+
				"这个模型里各字节的嵌入长度差不多（%.2f–%.2f），所以名次几乎完全由余弦决定——"+
				"与 logit 的相关是 %+.3f，只有 %.1f%% 的字节对两种排法不同。"+
				"也就是说「模型喜欢哪个字节」在这里等于「哪个字节的输入嵌入与输出方向更同向」。"+
				"绑定不是免费的：它把输入空间和输出空间焊在一起，这是一个结构选择，不只是省了 2.4 万个参数。",
				p.EmbedNormMin, p.EmbedNormMax, p.LogitCosineCorrelation,
				100*float64(p.PairsOrderedDifferently)/float64(p.Pairs)))
	})
}

// stageHeadIndependence answers why the heads are concatenated rather than summed, which
// the stage shows as a drawing and never justifies.
//
// The 加权求和 card draws each head's contribution and then "全部头相加", and the reason
// four slices are worth carrying is that they say different things. That is measurable:
// the projected outputs of two heads can be compared as vectors, per position. On the
// bundled weights at layer 0 the four heads are close to orthogonal — pairwise mean cosine
// 0.027 to 0.144 — and their lengths are comparable, 19% to 32% of the total. Summing them
// before their own projections would therefore be destructive rather than merely
// redundant.
//
// Layer 1 is a different picture (0.098 to 0.537), so the card measures the layer the
// reader has selected rather than quoting one number for the model; the two layers'
// attention heads are not equally specialised, which is the same asymmetry the residual
// stage reports.
func (a *App) stageHeadIndependence(c *ui.Context) {
	rows := make([]headIndependence, 0, len(a.Trace.Layers))
	for li := range a.Trace.Layers {
		m := computeHeadIndependence(a.Trace.Layers[li], len(a.Trace.Tokens))
		if !m.OK {
			continue
		}
		m.Layer = li
		rows = append(rows, m)
	}
	if len(rows) == 0 {
		return
	}
	a.cardNamed(c, "头之间重合吗", func() {
		sectionTitle(c, "四个头说的是同一件事吗", "两两比较各头的投影输出")
		mutedText(c, "下面那张图画的是「每个头各写了多少」，这张问的是另一件事：四个头写的东西像不像。"+
			"像，那四个头就是浪费；不像，把四段拼接起来才是对的。量法是两点：两两之间投影输出的平均余弦（越接近 0 越互不相干），"+
			"以及每个头占了多少长度。")
		out := make([]barRow, 0, len(rows))
		var maxPair float64
		for _, r := range rows {
			if r.MaxPairCosine > maxPair {
				maxPair = r.MaxPairCosine
			}
		}
		if maxPair <= 0 {
			maxPair = 1
		}
		for _, r := range rows {
			label := fmt.Sprintf("层 %d", r.Layer)
			colour := colGreen
			if r.Layer == a.Layer {
				label = fmt.Sprintf("层 %d ◂", r.Layer)
				colour = colAccent2
			}
			out = append(out, barRow{
				Label:      label,
				Value:      float32(r.MaxPairCosine),
				ValueText:  fmt.Sprintf("最大 cos %+.3f", r.MaxPairCosine),
				Color:      colour,
				LabelWidth: 46,
				ValueWidth: 104,
				Tip: fmt.Sprintf("层 %d：两两平均余弦 %+.3f 到 %+.3f；各头占长度 %.0f%%–%.0f%%",
					r.Layer, r.MinPairCosine, r.MaxPairCosine, 100*r.MinShare, 100*r.MaxShare),
			})
		}
		ui.Text(c, "两两之间最大的平均余弦（越接近 0，四个头越各说各的）").
			FontSize(fontMicro).TextColor(colFaint)
		bars(c, out, float32(maxPair), colGreen)
		for _, r := range rows {
			workingsLinesCopyable(c, []string{
				fmt.Sprintf("层 %d：四个头的投影输出两两平均余弦在 %+.3f 到 %+.3f 之间",
					r.Layer, r.MinPairCosine, r.MaxPairCosine),
				fmt.Sprintf("层 %d：各头占 ‖Δ‖ 总量的 %.0f%%–%.0f%%，量级相当",
					r.Layer, 100*r.MinShare, 100*r.MaxShare),
			})
		}
		first := rows[0]
		if len(rows) > 1 {
			first = rows[a.Layer]
		}
		if first.MaxPairCosine < 0.25 {
			misconception(c,
				"「多头注意力就是把同样的注意力做几遍再取平均，多几个头只是更稳」",
				fmt.Sprintf("四个头不是彼此的副本。层 %d 上，任意两个头的投影输出之间的平均余弦只有 %+.3f 到 %+.3f，"+
					"几乎互不相干，而且每个头的量级相当（%.0f%%–%.0f%%）。"+
					"把它们各自乘上自己的 Wo 切片再拼起来，等于让模型在同一个位置同时读四种不同的关系；"+
					"若是先相加再投影，这些差别会互相抵消。所以「多几个头」不是求平均求稳，而是并行的四条通道。",
					first.Layer, first.MinPairCosine, first.MaxPairCosine,
					100*first.MinShare, 100*first.MaxShare))
		}
	})
}

// stageFFNAnswerDirection answers what the FFN's output has to do with the answer, which
// the FFN stage never said: it draws the hidden units and the output sum and stops there.
//
// Rank every byte by the cosine between its input embedding and the vector in question,
// then ask where the byte that actually comes next lands. On the bundled weights:
//
//	layer 0   input mean rank 103  →  FFN output mean rank 4.5, top-10 at 30 of 35 positions
//	layer 1   input mean rank 3.7  →  FFN output mean rank 10.7
//
// So layer 0's FFN does not refine a hint that was already in its input — the hint is not
// there before it. Layer 1's input is already near the answer, and its FFN does not improve
// on that rank, which is the same asymmetry the residual stage reports as "the first block
// transforms, the second corrects".
func (a *App) stageFFNAnswerDirection(c *ui.Context) {
	if len(a.Trace.Tokens) < 2 {
		return
	}
	rows := make([]ffnAnswerDirection, 0, len(a.Trace.Layers))
	for li := range a.Trace.Layers {
		m := computeFFNAnswerDirection(a.W, a.Trace, li)
		if m.OK {
			rows = append(rows, m)
		}
	}
	if len(rows) == 0 {
		return
	}
	a.cardNamed(c, "FFN 写出了答案", func() {
		sectionTitle(c, "FFN 的输出指向下一个字节吗", "把 256 个字节按余弦排名")
		mutedText(c, "上一张卡片看的是单元怎么分工。这里问一个更直接的问题：把 256 个字节按「它的输入嵌入和这个向量的余弦」排名，"+
			"真实的下一个字节排第几？什么都不做的话，256 个里随便一个的平均名次是 128。")
		first := rows[0]
		out := make([]barRow, 0, len(rows)*2)
		var maxRank float64 = 8
		for _, m := range rows {
			if m.InputMeanRank > maxRank {
				maxRank = m.InputMeanRank
			}
			if m.OutputMeanRank > maxRank {
				maxRank = m.OutputMeanRank
			}
		}
		for _, m := range rows {
			hl := m.Layer == a.Layer
			inLabel := fmt.Sprintf("层 %d 输入", m.Layer)
			outLabel := fmt.Sprintf("层 %d FFN 后", m.Layer)
			inColour, outColour := colFaint, colGreen
			if hl {
				inColour, outColour = colMuted, colAccent2
				inLabel += " ◂"
				outLabel += " ◂"
			}
			out = append(out, barRow{
				Label: inLabel, Value: float32(m.InputMeanRank),
				ValueText: fmt.Sprintf("%.0f 名", m.InputMeanRank),
				Color:     inColour, LabelWidth: 96, ValueWidth: 56,
				Tip: fmt.Sprintf("层 %d 的输入向量：真实下一个字节的平均名次 %.1f / %d",
					m.Layer, m.InputMeanRank, m.Vocab),
			})
			out = append(out, barRow{
				Label: outLabel, Value: float32(m.OutputMeanRank),
				ValueText: fmt.Sprintf("%.0f 名", m.OutputMeanRank),
				Color:     outColour, LabelWidth: 96, ValueWidth: 56,
				Tip: fmt.Sprintf("层 %d 的 FFN 输出：平均名次 %.1f，前 10 名占 %d / %d 个位置",
					m.Layer, m.OutputMeanRank, m.OutputTop10, m.Positions),
			})
		}
		ui.Text(c, fmt.Sprintf("真实下一个字节的平均名次（越靠上越好，%d 个字节里随机是 ~%d 名）",
			first.Vocab, first.Vocab/2)).FontSize(fontMicro).TextColor(colFaint)
		bars(c, out, float32(maxRank), colGreen)
		for _, m := range rows {
			workingsLinesCopyable(c, []string{
				fmt.Sprintf("层 %d：输入里真实字节平均第 %.1f 名（余弦 %+.3f），FFN 之后第 %.1f 名（余弦 %+.3f）",
					m.Layer, m.InputMeanRank, m.InputMeanCos, m.OutputMeanRank, m.OutputMeanCos),
				fmt.Sprintf("层 %d：FFN 之后真实字节进入前 10 名 %d / %d 个位置",
					m.Layer, m.OutputTop10, m.Positions),
			})
		}
		// Layer 0 is the interesting one: its input is near chance and its output is not.
		if first.InputMeanRank > 3*first.OutputMeanRank {
			misconception(c,
				"「FFN 只是给已经算好的结果加点非线性调味，方向是注意力定的」",
				fmt.Sprintf("方向不是注意力定的。层 %d 的输入里，真实的下一个字节在 %d 个字节中平均排第 %.0f 名——和瞎猜差不多；"+
					"经过这一层的 FFN 之后，它排到第 %.0f 名，%d 个位置里有 %d 个进了前 10。"+
					"也就是说这一步是从无到有地把答案方向写进残差流，而不是修饰一个已有的信号。"+
					"这也正是第 5 步 logit lens 那张阶梯看到的现象——预测在层 0 的 FFN 之后才成形——的原因。",
					first.Layer, first.Vocab, first.InputMeanRank, first.OutputMeanRank,
					first.Positions, first.OutputTop10))
		}
	})
}

// stageEmbeddingStructure answers whether the embedding table encodes anything, which is
// the question the coloured strips raise and never answer.
//
// Every strip in this app exists to show a value, and a 96-wide coloured row per token
// invites the reading that bytes with similar meanings sit in similar places. On these
// weights that reading is false at every scale measured: pairs that follow each other in
// the prompt average 0.021 cosine against a random-pair mean of 0.003 with a spread of 0.10,
// vowels are 0.03 closer to each other than to consonants, and only one of the twenty most
// similar byte pairs is printable ASCII on both sides — the top pair is "I" and a control
// byte.
//
// The card exists because the plausible inference is the wrong one. A 96-dimension lookup
// trained on 61 sentences is not a geometry, and a reader who took the picture as evidence
// of semantic structure would learn something false about the model.
func (a *App) stageEmbeddingStructure(c *ui.Context) {
	e := computeEmbeddingStructure(a.W, a.Prompt)
	// The card's whole point is comparing text-adjacent pairs against unrelated ones, so it
	// needs adjacent pairs to compare. A one-token prompt has none, and the first version
	// rendered the comparison anyway with a mean of "+0.0000" — a number with an empty
	// denominator, which is what TestNoMalformedTextAcrossTheSweep exists to catch.
	if !e.OK || e.AdjacentPairs < 3 {
		return
	}
	a.cardNamed(c, "嵌入有含义吗", func() {
		sectionTitle(c, "这 96 个维度里有「意思」吗", fmt.Sprintf("%d 个字节两两比较", e.Vocab))
		mutedText(c, "右边那条色带每一格是一个维度。很自然会想：意思相近的字节，向量是不是也靠得近？"+
			"这个问题可以量——把 256 个字节的嵌入两两算余弦，看「文本里挨着出现的字节」是不是比随便两个更像。")
		// The comparison the reader needs: adjacent pairs against the random reference,
		// on one axis so the near-equality is a shape rather than two numbers.
		out := []barRow{
			{Label: "文本里相邻", Value: float32(math.Abs(e.AdjacentCos)),
				ValueText: fmt.Sprintf("%+.4f", e.AdjacentCos),
				Color:     colAccent2, LabelWidth: 100, ValueWidth: 64,
				Tip: fmt.Sprintf("提示词里紧挨着的 %d 对字节，平均余弦", e.AdjacentPairs)},
			{Label: "随便两个", Value: float32(math.Abs(e.RandomCos)),
				ValueText: fmt.Sprintf("%+.4f", e.RandomCos),
				Color:     colGreen, LabelWidth: 100, ValueWidth: 64,
				Tip: fmt.Sprintf("全部 %d 对的平均余弦", e.RandomPairs)},
		}
		maxAbs := math.Abs(e.AdjacentCos)
		if v := math.Abs(e.RandomCos); v > maxAbs {
			maxAbs = v
		}
		if maxAbs < 0.05 {
			maxAbs = 0.05 // a flat bar chart at this scale is the finding; give it room
		}
		ui.Text(c, "平均余弦（文本里相邻 vs 随便两个）").
			FontSize(fontMicro).TextColor(colFaint)
		bars(c, out, float32(maxAbs), colGreen)
		// Name the excluded pairs rather than quietly dropping them: a repeated byte has
		// cosine 1.0 with itself, and leaving those in would make the comparison look like
		// evidence of structure.
		adjacent := fmt.Sprintf("文本里相邻的 %d 对：平均余弦 %s", e.AdjacentPairs, cosText(e.AdjacentCos))
		if e.SkippedSelfPairs > 0 {
			adjacent = fmt.Sprintf("文本里相邻的 %d 对（另有 %d 对是同一个字节重复，余弦恒为 1，不计入）：平均余弦 %s",
				e.AdjacentPairs, e.SkippedSelfPairs, cosText(e.AdjacentCos))
		}
		workingsLinesCopyable(c, []string{
			adjacent,
			fmt.Sprintf("随便两对 %d 组：平均余弦 %+.4f，标准差 %.4f", e.RandomPairs, e.RandomCos, e.SDRandom),
			fmt.Sprintf("元音之间的平均余弦 %s，元音与辅音之间 %s", cosText(e.VowelCos), cosText(e.ConsonantCos)),
			fmt.Sprintf("最像的 20 对里，只有 %d 对两边都是可打印 ASCII", e.SimilarPrintableOf20),
		})
		// Name the top pairs: one glance at control bytes settles whether the similarity
		// means anything, in a way a summary statistic does not.
		if len(e.TopPairs) > 0 {
			var parts []string
			for i := 0; i < 4 && i < len(e.TopPairs); i++ {
				p := e.TopPairs[i]
				// byteText is the app's one way of writing a byte: space as ␣, control
				// bytes as \xNN, printable ASCII as itself. Using %q here printed Go's own
				// escaping ('\u009a'), which is a detail of this program's language rather
				// than of the model.
				parts = append(parts, fmt.Sprintf("%s-%s %.2f", byteText(p.A), byteText(p.B), p.Cos))
			}
			kvW(c, "最像的四对", strings.Join(parts, " · "), 96)
		}
		// Whether the difference is noise is a measurement, not a sentence: a prompt whose
		// bytes repeat in a pattern (abababab, say) produces adjacent pairs that really do sit
		// above the random spread. Measured on "ab"x50: adjacent +0.1737 against a random mean
		// of +0.0033 with spread 0.1017 — 1.7 standard deviations apart, while the fixed
		// sentence claimed the difference was inside the noise. The figures beside it were
		// always computed; the reading of them was not.
		verdict := "两者差别落在噪声里。"
		if e.SDRandom > 0 {
			if z := math.Abs(e.AdjacentCos-e.RandomCos) / e.SDRandom; z > 0.5 {
				verdict = fmt.Sprintf("两者相差 %.1f 个标准差，超出了随机波动——按这张表的说法，这段文本里相邻出现的字节对确实比随便两个更接近。"+
					"不过这里量的是「文本里挨着出现」，不是「意思相近」：把同一段字节按顺序重复，相邻对自然重复同一个方向。", z)
			}
		}
		misconception(c,
			"「嵌入向量越像，两个 token 的意思就越接近」",
			fmt.Sprintf("这张表里没有这回事。文本里挨着出现的字节对，平均余弦 %+.4f，而随便挑两对是 %+.4f（标准差 %.2f）——%s"+
				"元音之间只比元音与辅音近 %.2f。最像的 20 对里只有 %d 对两边都是可读字符，排第一的是「I」和一个控制字节。"+
				"原因不神秘：这个模型的位置表与嵌入表是在 61 句话上训练的，96 个维度够它记住「哪些字节常见」，不够长出语义几何。"+
				"真实的大模型里这种结构确实存在——但那是几十亿 token 训出来的，不是「嵌入」这个结构本身自带的。",
				e.AdjacentCos, e.RandomCos, e.SDRandom, verdict,
				math.Abs(e.VowelCos-e.ConsonantCos), e.SimilarPrintableOf20))
	})
}

// cosText formats a cosine for prose, dropping the sign when the value is too small to
// carry one.
//
// "+0.0000" and "-0.0000" are what %+.4f produces for numbers indistinguishable from zero,
// and both read as a formatting fault rather than as a measurement. The app already refuses
// them in TestNoMalformedTextAcrossTheSweep; this is the formatter that makes the refusal
// easy to honour. Nearest-zero cosines show as "0.0000" with no sign, which is the true
// statement — a cosine that small does not tell you a direction.
func cosText(v float64) string {
	if math.Abs(v) < 0.00005 {
		return "0.0000"
	}
	return fmt.Sprintf("%+.4f", v)
}

// causalInvarianceLine shows the causal mask's defining consequence, measured on the
// window the reader is actually looking at: shortening the input by its last byte leaves
// every remaining position's prediction where it was.
//
// The app teaches the mask in prose on four stages ("因果掩码让位置 t 只看得到 t 及其之前
// 的内容") and, until this, never showed the thing that claim is for. It is the property the
// whole left-to-right design rests on, and one an implementation can quietly fail: a row
// attending one key too far still sums to 1 and still draws a triangle, so nothing else in
// the app would look wrong. Measured on the bundled weights and the default prompt, all 35
// shared positions keep the same argmax; the largest probability movement is exactly 0 on the
// default build and 9.5e-7 on the SIMD one (see causalTolerance for why the latter is not 0).
//
// The figure is the reason a decoder is cheap to run: one forward pass predicts at every
// position at once, so extending the sequence needs one new row, not a re-run of the prefix.
//
// The denominator is the shorter run's positions, not the longer one's: the added byte
// creates a position with nothing before it to be compared against, and counting that as a
// disagreement would turn an invariant into a failure. When a row does disagree the line
// names its position instead of only counting — the reader's next question is "which one".
func (a *App) causalInvarianceLine(c *ui.Context) {
	cmp := a.causalInvariance()
	if !cmp.OK {
		return
	}
	// What the line states is the *bar*, not the measured deviation, and that is deliberate
	// twice over. A reader needs to know what "没变" is being measured against — "the same"
	// without a tolerance is not a claim — and the bar is a constant, so the sentence is the
	// same on every build. Printing cmp.WorstDelta instead would make the frame depend on
	// which matmul kernel was compiled in (exactly 0 on the default build, ~1e-6 on the SIMD
	// one), and docs/output.png is compared byte for byte under both — the CI caught exactly
	// that. The measured deviation is still checked, in the tests, where a number that differs
	// per build belongs.
	line := fmt.Sprintf("因果掩码核对：窗口去掉最后一个字节重跑一遍，剩下 %d 个位置的预测没变"+
		"（判据是每个概率的偏差 ≤ %.0e；掩码写错时实测差 %.0e 量级）",
		cmp.Compared, causalTolerance, causalLeakMagnitude)
	if cmp.Identical < cmp.Compared {
		line = fmt.Sprintf("因果掩码核对：窗口去掉最后一个字节重跑一遍，剩下的 %d 个位置里有 %d 个的预测变了",
			cmp.Compared, cmp.Compared-cmp.Identical)
		if cmp.MismatchAt >= 0 {
			line += fmt.Sprintf("；第一个在位置 %d", cmp.MismatchAt)
		}
	}
	ui.Text(c, line).FontSize(fontMicro).TextColor(colFaint).LineHeight(1.4)
}
