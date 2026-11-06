# Changelog

格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### Fixed

- `-weights` 加载前校验模型形状、最终 LayerNorm 长度与 byte vocab，畸形权重不再在 `Forward` 崩溃
- `Tokenize` 对无效 UTF-8 保持原始字节，`Token.Start` 改为文档承诺的 rune 偏移
- 截断提示词后按绝对位置保留选中 token；切换阶段重置主滚动、停止自动播放并清理 hover
- 检查器改为可滚动，1180×720 下 hover 卡片不再被窗口裁掉
- 修正 Top-12 尾部概率显示、LayerNorm 文案与侧栏编号

### Changed

- 前向把 `AttnProj` 改为逐头投影求和并移除 `matMulRows` 的零值分支：每层少算一次 d×d 投影，64 token 提示词每次前向少 130 次分配
- `tools/train.py` 默认步数改回 600，并拒绝非正的 steps/batch/repeats 与 NaN 输出

## [0.1.0] - 2026-11-06

首个可用版本：一个纯 Go 实现的 Transformer 教学工具，模型与界面在同一个二进制里。

### Added

- 字节级 tokenizer 与 25.3 万参数的微型 GPT（2 层 × 4 头 × 96 维，ctx 64），权重 `go:embed`
- 七个可单步前进/后退的阶段：分词、嵌入、Q/K/V 与分数、softmax 与加权、残差/LayerNorm、FFN、logits 与采样
- 自绘可视化：注意力/分数热力图（因果掩码、冷暖双色）、token 向量条带、概率条形图、激活直方图
- 交互：改提示词实时重跑前向、层/头切换、温度滑块、token 选择、悬停查看格子数值、自动播放
- 右侧检查器显示选中 token 的注意力分配、向量统计与当前阶段公式
- `just` 多端构建：默认一次产出 Linux（amd64 + arm64）、Windows（amd64）、macOS（universal）
- `just run` / `dev` / `test` / `doctor` / `artifacts` / `clean`，以及 `-w/-h/-step` 启动参数
- CI（三平台测试 + 三平台构建）与打 tag 自动发布安装包的 release 工作流

### Changed

- 训练脚本导出 600 步权重的同时导出固定提示词 logits，供 Go 侧对拍
- 采样从「按码点取字符」改为「按字节追加」，避免写出非法 UTF-8
- 上下文截断前进到 UTF-8 字符边界，不会从半个汉字开始

### Fixed

- 第 4 阶段把四个头的注意力输出之和标成「本头 × Wo」：新增 `HeadProj`，界面改成本头贡献与全部头相加两条
- 第 5 阶段只画了注意力残差：补齐 `Δ_ffn` 与最终 `x + Δ_attn + Δ_ffn`
- 最小窗口 1180×720 下顶部工具栏被垂直居中裁切（中栏缺 `FillHeight`）
- 64 token 长提示词时热力图右侧若干列未绘制（格子按请求尺寸而非实际布局计算）
- 分数热力图用 `|score|` 着色，正负分无法区分
- Top-K 条形图按标签字符串还原时间顺序，超过 9 个 token 后顺序错乱
