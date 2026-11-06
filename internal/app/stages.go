package app

import (
	"fmt"
	"math"
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
		title:   "第 1 步 · 把文字切成 token",
		body:    "模型不认识字母，只认识整数。这里用最朴素的字节级分词：把提示词的 UTF-8 字节逐个映射成 0–255 的 id。英文一个字符一个 token，一个汉字要占三个 token——右边显示的数字就是 id。真实的 LLM 用 BPE 把高频字节对合并成子词，词表通常三万到十五万条，道理相同：先离散化，再查表。",
		formula: "id_t = byte(prompt)[t]",
	},
	{
		name:    "嵌入",
		title:   "第 2 步 · 查表得到向量",
		body:    "嵌入表是一张 vocab × d_model 的矩阵，每一行就是一个 token 的「意思」。查表得到向量后，再加上位置向量：注意力本身对顺序不敏感，位置信息必须显式注入，否则 “dog bites man” 和 “man bites dog” 在模型眼里完全一样。",
		formula: "x_t = E[token_t] + P[t]",
	},
	{
		name:    "注意力分数",
		title:   "第 3 步 · Query 与 Key 的点积",
		body:    "每个位置的向量被三个矩阵投影成 Query（我在找什么）、Key（我能提供什么）、Value（我携带的内容）。第 t 个位置用它的 Query 去点乘前面每个位置的 Key，得到「相关度」。除以 √d_head 是为了让方差稳定，否则维度一高分数就会爆炸，softmax 直接饱和。注意右上三角是空的：因果掩码让模型看不到未来。",
		formula: "score[t][s] = (q_t · k_s) / √d_head,  s ≤ t",
	},
	{
		name:    "Softmax",
		title:   "第 4 步 · 变成权重再混合",
		body:    "一行分数经过 softmax 变成概率分布，每一行的和恰好是 1。用这些权重去加权求和 Value，就得到这个位置「从上下文里取回的信息」。多个头并行做同一件事，各自学不同的关注模式：有的盯上一个 token，有的盯句首，有的盯标点。",
		formula: "α = softmax(score),  out_t = Σ_s α[t][s] · v_s",
	},
	{
		name:    "残差 / LN",
		title:   "第 5 步 · 残差与层归一化",
		body:    "子层的输出不会覆盖原向量，而是加回去——这就是残差连接，它让梯度有一条高速公路，也让「原始信息」一路保留到最后。加法之前先做 LayerNorm：减均值、除标准差，再乘可学习的 γ、加 β。看直方图会发现归一化后分布被拉回零附近，这就是数值稳定的来源。一个块里有两次这样的加法：一次接注意力，一次接 FFN。",
		formula: "x ← x + Attn(LN₁(x));  x ← x + FFN(LN₂(x))",
	},
	{
		name:    "FFN",
		title:   "第 6 步 · 逐位置的前馈网络",
		body:    "注意力负责在位置之间搬运信息，FFN 负责在每个位置上单独做非线性变换：先升维到 4 倍宽，过 GELU，再降回原维度。参数量的三分之二都在这里。GELU 在负值区不是硬零，而是平滑地趋近零，所以激活会呈现稀疏但连续的分布。",
		formula: "FFN(x) = W₂ · GELU(W₁ · LN(x))",
	},
	{
		name:    "输出 / 采样",
		title:   "第 7 步 · 从向量到下一个字",
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
	card(c, func() {
		sectionTitle(c, "Token 序列", fmt.Sprintf("T = %d", len(tr.Tokens)))
		a.tokenChips(c, tr.Tokens)
		ui.Row(c).Gap(16).Children(func() {
			ui.Text(c, fmt.Sprintf("字节数 %d", len(tr.Tokens))).FontSize(11).TextColor(colFaint)
			ui.Text(c, fmt.Sprintf("字符数 %d", utf8.RuneCountInString(tr.Text))).FontSize(11).TextColor(colFaint)
			ui.Text(c, fmt.Sprintf("上下文上限 %d", tr.Config.Ctx)).FontSize(11).TextColor(colFaint)
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
		sectionTitle(c, "全流程一览", "点击任意阶段直接跳转")
		ui.Row(c).Wrap().Gap(6).AlignItems(ui.Center).Children(func() {
			for i, l := range lessons {
				i, l := i, l
				chip := ui.Row(c).Padding(6, 9).Radius(6).Gap(5).AlignItems(ui.Center).
					Background(colPanel2).Border(1, colBorder).Children(func() {
					ui.Text(c, fmt.Sprintf("%d", i+1)).FontSize(9).TextColor(colFaint)
					ui.Text(c, l.name).FontSize(11).TextColor(colMuted)
				})
				if i == a.Step {
					chip = chip.Background(colAccent.Alpha(0.16)).Border(1, colAccent)
				}
				if chip.Clicked() {
					a.selectStep(i)
				}
				if i < len(lessons)-1 {
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
		ui.Row(c).Gap(8).AlignItems(ui.Center).Children(func() {
			ui.Text(c, l.title).FontSize(14).Bold()
			ui.Spacer(c).Grow(1)
			ui.Text(c, l.formula).FontSize(11).TextColor(colAccent2).
				Padding(4, 8).Radius(5).Background(colPanel2)
		})
		mutedText(c, l.body)
	})
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
