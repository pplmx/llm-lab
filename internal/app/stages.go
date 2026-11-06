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
}

var lessons = []lesson{
	{
		name:    "分词",
		title:   "把文字切成 token",
		body:    "模型不认识字母，只认识整数。这里用最朴素的字节级分词：把提示词的 UTF-8 字节逐个映射成 0–255 的 id。英文一个字符一个 token，一个汉字要占三个 token——右边显示的数字就是 id。真实的 LLM 用 BPE 把高频字节对合并成子词，词表通常三万到十五万条，道理相同：先离散化，再查表。",
		formula: "id_t = byte(prompt)[t]",
	},
	{
		name:    "嵌入",
		title:   "查表得到向量",
		body:    "嵌入表是一张 vocab × d_model 的矩阵，每一行就是一个 token 的「意思」。查表得到向量后，再加上位置向量：注意力本身对顺序不敏感，位置信息必须显式注入，否则 “dog bites man” 和 “man bites dog” 在模型眼里完全一样。",
		formula: "x_t = E[token_t] + P[t]",
	},
	{
		name:    "注意力分数",
		title:   "Query 与 Key 的点积",
		body:    "每个位置的向量被三个矩阵投影成 Query（我在找什么）、Key（我能提供什么）、Value（我携带的内容）。第 t 个位置用它的 Query 去点乘前面每个位置的 Key，得到「相关度」。除以 √d_head 是为了让方差稳定，否则维度一高分数就会爆炸，softmax 直接饱和。注意右上三角是空的：因果掩码让模型看不到未来。",
		formula: "score[t][s] = (q_t · k_s) / √d_head,  s ≤ t",
	},
	{
		name:    "Softmax",
		title:   "变成权重再混合",
		body:    "一行分数经过 softmax 变成概率分布，每一行的和恰好是 1。用这些权重去加权求和 Value，就得到这个位置「从上下文里取回的信息」。多个头并行做同一件事，各自学不同的关注模式：有的盯上一个 token，有的盯句首，有的盯标点。",
		formula: "α = softmax(score),  out_t = Σ_s α[t][s] · v_s",
	},
	{
		name:    "残差 / LN",
		title:   "残差与层归一化",
		body:    "子层的输出不会覆盖原向量，而是加回去——这就是残差连接，它让梯度有一条高速公路，也让「原始信息」一路保留到最后。加法之前先做 LayerNorm：减均值、除标准差，再乘可学习的 γ、加 β。看直方图会发现归一化后分布被拉回零附近，这就是数值稳定的来源。一个块里有两次这样的加法：一次接注意力，一次接 FFN。",
		formula: "x ← x + Attn(LN₁(x));  x ← x + FFN(LN₂(x))",
	},
	{
		name:    "FFN",
		title:   "逐位置的前馈网络",
		body:    "注意力负责在位置之间搬运信息，FFN 负责在每个位置上单独做非线性变换：先升维到 4 倍宽，过 GELU，再降回原维度。参数量的三分之二都在这里。GELU 在负值区不是硬零，而是平滑地趋近零，所以激活会呈现稀疏但连续的分布。",
		formula: "FFN(x) = W₂ · GELU(W₁ · LN(x))",
	},
	{
		name:    "输出 / 采样",
		title:   "从向量到下一个字",
		body:    "最后一层的输出经过最终 LayerNorm，再与嵌入矩阵相乘（权重绑定），得到词表上每个 token 的 logit。logit 不是概率，要除以温度再做 softmax。温度趋近 0 等于贪心，温度大于 1 会让分布变平、输出更随机。",
		formula: "p = softmax(logits / T)",
	},
}

func (a *App) stage(c *ui.Context) {
	tr := a.Trace
	if len(tr.Tokens) == 0 {
		mutedText(c, "输入一些文字开始。")
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

func (a *App) stageTokens(c *ui.Context) {
	tr := a.Trace
	// The chip grid that used to open this stage now lives in the strip above
	// every stage, so this card states the counts and leaves the reading of
	// the sequence itself to that strip.
	card(c, func() {
		sectionTitle(c, "提示词变成 token 序列", fmt.Sprintf("T = %d", len(tr.Tokens)))
		ui.Row(c).Gap(16).Wrap().Children(func() {
			ui.Text(c, fmt.Sprintf("字节数 %d", len(tr.Tokens))).FontSize(11).TextColor(colFaint)
			ui.Text(c, fmt.Sprintf("字符数 %d", utf8.RuneCountInString(tr.Text))).FontSize(11).TextColor(colFaint)
			ui.Text(c, fmt.Sprintf("上下文上限 %d", tr.Config.Ctx)).FontSize(11).TextColor(colFaint)
			if extra := len(tr.Tokens) - utf8.RuneCountInString(tr.Text); extra > 0 {
				ui.Text(c, fmt.Sprintf("多字节字符多占 %d 个 token", extra)).FontSize(11).TextColor(colAccent2)
			}
		})
		mutedText(c, "窗口顶部的 token 条就是这条序列：每个格子是一个 token，上面的字是它的字面形态，"+
			"下面的数字是它的 id。点任意一格，右侧检查器会显示这个 token 在当前阶段的数值。")
		ui.Row(c).Gap(6).Wrap().AlignItems(ui.Center).Children(func() {
			ui.Text(c, "试试").FontSize(10).TextColor(colFaint).Shrink(0)
			for _, p := range promptPresets {
				p := p
				chip := ui.Row(c).Padding(4, 8).Radius(5).Gap(4).AlignItems(ui.Center).
					Background(colPanel2).Border(1, colBorder).Children(func() {
					ui.Text(c, p.name).FontSize(10.5).TextColor(colMuted).NoWrap()
				})
				if a.Prompt == p.prompt {
					chip = chip.Background(colAccent.Alpha(0.14)).Border(1, colAccent.Alpha(0.6))
				}
				if chip.Clicked() {
					a.applyPrompt(p.prompt)
				}
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
					ui.Text(c, label).FontSize(14).TextColor(colText)
					ui.Row(c).Gap(2).Children(func() {
						for _, b := range g.bytes {
							ui.Text(c, fmt.Sprintf("%d", b)).FontSize(9).TextColor(colFaint).
								Padding(2, 3).Radius(3).Background(colPanel2)
						}
					})
				})
			}
		})
	})

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
			{6, "输出", fmt.Sprintf("[%d]", tr.Config.Vocab)},
		}
		ui.Row(c).Wrap().Gap(6).AlignItems(ui.Center).Children(func() {
			for i, r := range rows {
				i, r := i, r
				chip := ui.Column(c).Gap(1).Padding(5, 9).Radius(6).
					Background(colPanel2).Border(1, colBorder).Children(func() {
					ui.Text(c, r.name).FontSize(10).TextColor(colMuted).NoWrap()
					ui.Text(c, r.shape).FontSize(10.5).TextColor(colAccent2).NoWrap()
				})
				if r.step == a.Step {
					chip = chip.Background(colAccent.Alpha(0.12)).Border(1, colAccent.Alpha(0.6))
				}
				if chip.Clicked() {
					a.selectStep(r.step)
				}
				if i < len(rows)-1 {
					ui.Text(c, "→").FontSize(11).TextColor(colFaint)
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
		ui.Text(c, "token embedding").FontSize(10).TextColor(colFaint)
		vectorStrip(c, tr.Embed[i], 20, &a.HoverDimIn)
		ui.Text(c, "position embedding").FontSize(10).TextColor(colFaint)
		vectorStrip(c, tr.Pos[i], 20, &a.HoverDimIn)
		ui.Text(c, "x = embedding + position").FontSize(10).TextColor(colAccent2)
		vectorStrip(c, tr.Input[i], 24, &a.HoverDimIn)
	})

	card(c, func() {
		sectionTitle(c, "位置向量随位置变化", "取前四个位置的 P[t]")
		for t := 0; t < 4 && t < len(tr.Pos); t++ {
			ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
				ui.Text(c, fmt.Sprintf("t=%d", t)).FontSize(10).TextColor(colFaint).Width(26)
				ui.Box(c).Grow(1).Children(func() {
					vectorStrip(c, tr.Pos[t], 14, &a.HoverDimIn)
				})
			})
		}
	})
}

func (a *App) stageScores(c *ui.Context) {
	tr := a.Trace
	l := a.layer()
	i := clampIndex(a.SelToken, len(tr.Tokens))
	h := l.Heads[a.Head]
	card(c, func() {
		sectionTitle(c, fmt.Sprintf("token %d 的 Q / K / V", i), fmt.Sprintf("头 %d · d_head = %d", a.Head, tr.Config.DHead()))
		ui.Text(c, "Query").FontSize(10).TextColor(colFaint)
		vectorStrip(c, h.Q[i], 18, &a.HoverDimQ)
		ui.Text(c, "Key（同位置）").FontSize(10).TextColor(colFaint)
		vectorStrip(c, h.K[i], 18, &a.HoverDimK)
		ui.Text(c, "Value（同位置）").FontSize(10).TextColor(colFaint)
		vectorStrip(c, h.V[i], 18, &a.HoverDimV)
	})

	card(c, func() {
		sectionTitle(c, "分数矩阵", "行 = query，列 = key，右上三角被掩码")
		ui.Row(c).Gap(14).AlignItems(ui.Start).Children(func() {
			ui.Column(c).Gap(4).Children(func() {
				a.heatmap(c, h.Scores, "score")
			})
			ui.Column(c).Gap(6).Grow(1).Children(func() {
				mutedText(c, "点积越大，说明这个位置的 Query 与那个位置的 Key 越匹配。除法因子让分数的量级不随维度增长。")
				ui.Row(c).Gap(12).Children(func() {
					ui.Row(c).Gap(5).AlignItems(ui.Center).Children(func() {
						ui.Box(c).Size(9, 9).Radius(2).Background(colWarm)
						ui.Text(c, "正分：鼓励注意").FontSize(10).TextColor(colMuted)
					})
					ui.Row(c).Gap(5).AlignItems(ui.Center).Children(func() {
						ui.Box(c).Size(9, 9).Radius(2).Background(colNeg)
						ui.Text(c, "负分：抑制注意").FontSize(10).TextColor(colMuted)
					})
				})
				kv(c, "缩放", fmt.Sprintf("1/√%d = %.3f", tr.Config.DHead(), 1/math.Sqrt(float64(tr.Config.DHead()))))
				kv(c, "可见", fmt.Sprintf("%d 个 key", i+1))
			})
		})
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
				a.heatmap(c, h.Attn, "attn")
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
	})

	card(c, func() {
		sectionTitle(c, "加权求和", "本头的输出，以及它在残差里的份额")
		ui.Text(c, fmt.Sprintf("Σ α·v（头内输出，d_head = %d）", tr.Config.DHead())).FontSize(10).TextColor(colFaint)
		vectorStrip(c, h.Out[i], 18, &a.HoverDimH)
		// Head h only multiplies the rows of Wo that belong to its own slice,
		// so this is the head's actual contribution, not the sum of all heads.
		ui.Text(c, "· Wo[h] 切片（本头贡献 Δ_h，d_model 宽）").FontSize(10).TextColor(colFaint)
		vectorStrip(c, l.HeadProj[a.Head][i], 18, &a.HoverDimH)
		ui.Text(c, "全部头相加 = 本层注意力输出").FontSize(10).TextColor(colAccent2)
		vectorStrip(c, l.AttnProj[i], 18, &a.HoverDimH)
	})
}

func (a *App) stageResidual(c *ui.Context) {
	l := a.layer()
	i := clampIndex(a.SelToken, len(a.Trace.Tokens))

	card(c, func() {
		sectionTitle(c, "残差流", "两次加法，信息都绕过了子层")
		ui.Text(c, "进入注意力子层 x").FontSize(10).TextColor(colFaint)
		vectorStrip(c, l.In[i], 16, &a.HoverDimIn)
		ui.Text(c, "注意力子层输出 Δ_attn").FontSize(10).TextColor(colFaint)
		vectorStrip(c, l.AttnProj[i], 16, &a.HoverDimIn)
		ui.Text(c, "x + Δ_attn").FontSize(10).TextColor(colAccent2)
		vectorStrip(c, l.AfterAttn[i], 20, &a.HoverDimIn)
		ui.Text(c, "FFN 子层输出 Δ_ffn").FontSize(10).TextColor(colFaint)
		vectorStrip(c, l.FFNOut[i], 16, &a.HoverDimIn)
		ui.Text(c, "x + Δ_attn + Δ_ffn（送往下一层）").FontSize(10).TextColor(colAccent2)
		vectorStrip(c, l.AfterFFN[i], 20, &a.HoverDimIn)
	})

	card(c, func() {
		before := l.AfterAttn[i]
		after := l.Norm2[i]
		mb, sb := meanStd(before)
		ma, sa := meanStd(after)
		sectionTitle(c, "LayerNorm", fmt.Sprintf("μ %.3f → %.3f · σ %.3f → %.3f", mb, ma, sb, sa))
		ui.Text(c, "归一化前").FontSize(10).TextColor(colFaint)
		histogram(c, before, 32, colWarm)
		ui.Text(c, "归一化后").FontSize(10).TextColor(colAccent2)
		histogram(c, after, 32, colAccent2)
		mutedText(c, "LayerNorm 先减均值、除标准差，再乘可学习的 γ、加 β；直方图展示最终的 γ/β 输出，所以方差不必恰好为 1。")
	})
}

func (a *App) stageFFN(c *ui.Context) {
	l := a.layer()
	i := clampIndex(a.SelToken, len(a.Trace.Tokens))
	hidden := l.Hidden[i]
	active := 0
	for _, v := range hidden {
		if v > 0.01 {
			active++
		}
	}
	card(c, func() {
		sectionTitle(c, "升维 → GELU → 降维", fmt.Sprintf("d_ff = %d", len(hidden)))
		ui.Text(c, "W₁·LN(x) 后过 GELU").FontSize(10).TextColor(colFaint)
		vectorStrip(c, hidden, 22, &a.HoverDimH)
		ui.Row(c).Gap(16).Wrap().Children(func() {
			ui.Text(c, fmt.Sprintf("输出 > 0.01 的维度 %d / %d (%.0f%%)", active, len(hidden), 100*float64(active)/float64(len(hidden)))).FontSize(11).TextColor(colMuted)
			_, hi := minMaxFloat(hidden)
			ui.Text(c, fmt.Sprintf("峰值 %+.2f", hi)).FontSize(11).TextColor(colFaint)
		})
		histogram(c, hidden, 36, colGreen)
		mutedText(c, "GELU 不是 ReLU：负值不会被截成 0，只是被平滑压向 0，所以分布右侧拖出一条长尾，这就是「稀疏但连续」。")
		ui.Text(c, "× W₂ 回到 d_model").FontSize(10).TextColor(colFaint)
		vectorStrip(c, l.FFNOut[i], 18, &a.HoverDimH)
	})
}

func (a *App) stageOutput(c *ui.Context) {
	tr := a.Trace
	last := len(tr.Tokens) - 1
	probs := tr.Probabilities(a.Temp)
	top := model.TopK(probs, 12)
	var maxP float32
	for _, id := range top {
		if probs[id] > maxP {
			maxP = probs[id]
		}
	}

	card(c, func() {
		sectionTitle(c, "最后位置的向量", "最终 LayerNorm 之后")
		vectorStrip(c, tr.FinalNorm[last], 22, &a.HoverDimIn)
		kv(c, "下一 token", fmt.Sprintf("温度 %.2f · 熵 %.2f nats", a.Temp, model.Entropy(probs)))
	})

	card(c, func() {
		sectionTitle(c, "Top-12 概率", "softmax(logits / T)")
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
				Tip:        fmt.Sprintf("id %d · %q · #%d", id, byteText(id), rank+1),
				Index:      rank,
			})
		}
		bars(c, rows, maxP, colAccent2)
	})

	card(c, func() {
		sectionTitle(c, "采样", "从分布里抽一个 token")
		ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
			if ui.PrimaryButton(c, "采样并续写").Clicked() {
				a.sampleNext()
			}
			ui.Text(c, "续写结果会追加到提示词，并重新跑一遍前向。").FontSize(10).TextColor(colFaint)
		})
		if a.LastSample != "" {
			kv(c, "上一次", a.LastSample)
		}
	})
}

func (a *App) lesson(c *ui.Context) {
	l := lessons[a.Step]
	card(c, func() {
		sectionTitle(c, "这一段在讲什么", l.name)
		mutedText(c, l.body)
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
				FontSize(10).TextColor(colFaint).LineHeight(1.4)
		}
		ui.Text(c, "快捷键：← → 切换阶段 · Shift+← → 切换 token · 空格 播放/暂停 · 1–7 跳转").
			FontSize(10).TextColor(colFaint)
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
					ui.Text(c, fmt.Sprintf("%d", tr.Tokens[i].Byte)).FontSize(13).TextColor(colText).
						Padding(3, 7).Radius(4).Background(colPanel2)
					ui.Text(c, fmt.Sprintf("#%d", i)).FontSize(9).TextColor(colFaint)
				})
			}
		})
		kv(c, "rune", fmt.Sprintf("%q", tk.Rune))
		kv(c, "位置", fmt.Sprintf("%d / %d", a.SelToken, len(tr.Tokens)-1))
	})
	card(c, func() {
		sectionTitle(c, "为什么中文更长", "")
		mutedText(c, "ASCII 一个字节就是一个 token；一个汉字在 UTF-8 里占 3 个字节，所以是 3 个 token。")
		kv(c, "提示词", fmt.Sprintf("%d 字节 / %d 字符", len(tr.Text), utf8.RuneCountInString(tr.Text)))
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
		kv(c, "缩放", fmt.Sprintf("1/√%d = %.3f", tr.Config.DHead(), 1/math.Sqrt(float64(tr.Config.DHead()))))
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
		mutedText(c, "熵越接近 0 越专注；越接近 log(n) 越分散。")
	})
}

func (a *App) inspectorResidual(c *ui.Context) {
	l := a.layer()
	i := clampIndex(a.SelToken, len(a.Trace.Tokens))
	card(c, func() {
		sectionTitle(c, "向量", "残差流")
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
		mutedText(c, "归一化后再乘 γ、加 β，所以最终方差不必恰好为 1。")
	})
}

func (a *App) inspectorFFN(c *ui.Context) {
	l := a.layer()
	i := clampIndex(a.SelToken, len(a.Trace.Tokens))
	hidden := l.Hidden[i]
	active := 0
	for _, v := range hidden {
		if v > 0.01 {
			active++
		}
	}
	mean, std := meanStd(hidden)
	_, hi := minMaxFloat(hidden)
	card(c, func() {
		sectionTitle(c, "隐藏层统计", fmt.Sprintf("d_ff = %d", len(hidden)))
		kv(c, "激活 > 0.01", fmt.Sprintf("%d / %d (%.0f%%)", active, len(hidden), 100*float64(active)/float64(len(hidden))))
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
		mutedText(c, "温度趋近 0 时分布变尖，等于贪心；大于 1 时变平，输出更随机。")
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
