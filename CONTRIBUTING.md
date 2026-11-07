# 参与开发

## 环境

- Go 1.27.3+（`go.mod` 里写死了最低版本，本地更低时 Go 会自动拉工具链）
- Linux 额外需要 GTK 3：`sudo apt install libgtk-3-0`
- 构建用 [just](https://github.com/casey/just)：`brew install just` / `winget install Casey.Just` / `cargo install just`
- 只有改模型权重或对拍夹具时才需要 Python 3 + PyTorch（CPU 版即可）

## 常用命令

```bash
just run          # 本机运行
just dev          # 热重载
just test         # go vet + go test
just check        # gofmt 检查 + vet + test（CI 的主要门槛）
just build        # 三端构建
just fmt          # gofmt -w
just bench        # 基准测试（前向 + 整帧渲染，多轮）
just test-simd    # 用实验性 SIMD 跑测试（GOEXPERIMENT=simd）
just bench-simd   # 用实验性 SIMD 跑前向基准
just build-simd   # 用实验性 SIMD 构建（可加平台参数）
just run-simd     # 用实验性 SIMD 本机运行
```

## 代码地图

| 路径 | 职责 |
| --- | --- |
| `internal/model/` | 纯 Go 前向：张量运算、tokenizer、`Forward` 与 `Trace`、权重加载 |
| `internal/app/` | MyGo 界面：`app.go` 布局与交互、`stages.go` 七个阶段视图、`widgets.go` 自绘组件 |
| `tools/train.py` | PyTorch 训练并导出 `weights.json`（含参考 logits） |
| `tools/parity.py` | 导出逐层中间张量，供 Go 侧对拍 |

## 改动的验证标准

**CI 比 `just check` 多跑三步**，提交前想完全对齐就跑 `just check-ci`，它逐条对应 `.github/workflows/ci.yml`：

```bash
just check-ci                      # check + 下面三条

go mod tidy && git diff --exit-code -- go.mod go.sum
go test -race ./internal/model/     # 只有这个包有 goroutine
GOEXPERIMENT=simd go test ./...     # 实验性 SIMD 内核
```

两个门槛分开是有意的：`just check` 约 35 秒，`just check-ci` 冷缓存约 75 秒（SIMD 要把整个测试集再跑一遍）。日常提交跑前者；动到模块依赖、并发或 SIMD 内核时跑后者。

`-race` 只跑 `internal/model`：`internal/app` 是单线程 UI 代码，而在 race 检测器下一次 `ui.Tester.Frame()` 要 24ms（正常 0.8ms），整个包会跑到 11 分钟，超过 Go 默认的 10 分钟测试超时。

**界面改动的截图对比**：`just snapshots` 把整套截图（七阶段 × 三种窗口尺寸，加上 CJK、长提示词、空提示词、截断、未训练等变体，目前 42 张）渲染到一个目录，`just snapshots-diff <基线> [新目录]` 逐字节比对，列出变化的文件名与「缺失/新增」。不改变视觉的改动应当 0 变化；故意改视觉的应当只差在动手的那几个阶段。比对前会先检查基线目录存在且非空——把空目录当基线会误报「全部变化」。

```bash
just snapshots /tmp/before        # 改之前
just snapshots /tmp/after         # 改之后
just snapshots-diff /tmp/before /tmp/after
```

**模型或界面数字**：`go test ./internal/model` 必须全绿。动了前向逻辑要重新生成对拍夹具，否则 `TestParityEveryTensor` 会失败：

```bash
python3 tools/parity.py
go test ./internal/model -run TestParityEveryTensor -v
```

**界面布局**：无窗口渲染检查，不需要开 GUI：

```bash
RENDER=1 go test ./internal/app -run 'TestRenderSteps|TestRenderSizes'
RENDER=1 RENDER_DIR=/tmp/after go test ./internal/app -run 'TestRenderSteps|TestRenderSizes'
```

截图默认写到 `/tmp/llmlab-step*.png`、`/tmp/size-*.png`，覆盖 1180/1500/1920 三种宽度与 64 token 长提示词。改动 UI 请在 PR 里附 1180×720 和 1500×940 两张。

对比改动前后时给两次渲染各指定一个 `RENDER_DIR`：默认位置每跑一次就覆盖，跑完一轮再想比就没有基线了。`cmp -s` 是够用的比对方式——这个渲染是确定的，同一状态两次出图逐字节相同。

**打包**：`just build` 要能出三端产物；只改代码不涉及打包配置时，至少跑一次 `just build windows/amd64` 确认交叉编译没坏。

## 界面上的数字与被它支撑的说法

这个工具的教学价值全在于「界面说的每句话都成立」。测试能保证代码不崩、布局不溢出，但保不住下面这几件事——它们都是实际发生过的缺陷：

1. **同一组图必须用同一个尺度比较。** LayerNorm 前后两张直方图原本各按自己的极值缩放，于是收窄一半的分布看起来和原来一样宽。要比较的两个图，横轴必须是同一个区间。
2. **颜色深浅要说清上限。** 热力图的色深是 `|v| ÷ 上限`，而上限随提示词变化（本模型 5.4–8.3）。不写出来，读者会以为亮暗是绝对刻度。
3. **图形要标出参照点。** 矩阵的行列位置、向量条带的起止维度、直方图的 0 位——没有这些，图形是没法读的色块。
4. **数字要支撑它旁边那句话。** FFN 卡片曾用「输出 > 0.01 的维度 38%」来支持「稀疏」，而实测这 384 个值没有一个接近零，那个数字描述的只是阈值选择。换成「最大的 10% 承担 49% 的质量」才真的支撑得住。
5. **孤立数字要有量纲参照。** `熵 0.16 nats` 本身没有意义，加上「上限 log(256) = 5.55，当前占 3%」才读得懂。
6. **切片要说清总体。** 「Top-12 概率」的 12 项覆盖了 99.99999% 的概率质量，读者很容易以为词表只有 12 个候选。
7. **文档里的数字必须从代码派生。** README 曾写「36 token 逐个标、64 token 每 6 个标一次」，实测是每 3 个和每 4 个——两个数字都是估的。
8. **成对的东西要由一个来源决定。** 这类缺陷的表现是「两个值必须一致，但没有任何机制保证」，改一边不会报警。实际发生过的：窗口最小尺寸 1180×720 与截图套件的最小尺寸脱钩（删掉任一边，另一边不会失败）；温度控件的提示写死「第 7 步」而层/头两个控件的提示是从区间常量生成的；README 说截图覆盖「七个阶段」而矩阵只渲染四个——文档超前于实现。
9. **状态的生命周期要定义完整。** 读者做完一个动作后，另一个状态该不该跟着变，必须逐个说明并测到。实际发生过的：固定了热力图格子后在 token 条里换一个 token，检查器标题写新 token、下面的格子卡还描述旧 token；向量条带的高亮框在指针离开后继续画（值该留、框不该留，代码只做了前一半）。
10. **「发生时对不对」和「结束后对不对」是两件事。** 上面的高亮框和固定格子都有测试覆盖「点击后正确」，没有一条覆盖「离开后正确」。补测试时把动作拆成「做 → 停 → 再观察」。

对应的测试：`app_test.go` 里 `TestLayerNormHistogramsShareOneAxis`、`TestScoreLegendStatesTheColorScale`、`TestFFNCardQuantifiesSparsity`、`TestOutputEntropyStatesItsCeiling`、`TestTop12StatesTheVocabularySize`；`readme_test.go` 的整组断言；第 8–10 类对应 `TestWindowMinimumMatchesTheSmallestSnapshot`、`TestEveryStageIsRenderedAtEverySize`、`TestTemperatureNoteFollowsItsRange`、`TestSelectingAnotherTokenDropsThePin`、`TestStripCaretDoesNotOutliveThePointer`。

**写测试时注意两件事**，都是踩过的：

- 断言要从**渲染出来的文本**里读数字（`tester.Texts()`），不能只重算模型——只比模型时，把界面数字改错测试照样通过。
- 破坏性验证要用**非默认样本**。硬编码「0.95」「3%」这类值第一次都恰好等于默认选中项的真实值，测试因此通过；换个 token 才暴露。

## 提交

- 一个提交做一件事，信息用祈使句，说明「改了什么、为什么」，例如 `fix: 热力图按实际布局宽度绘制格子`
- 不要提交 `build/`、`.mygo/`、`weights.json` 之外的大文件
- 新增依赖请在 PR 里说明用途；Go 依赖只加必要项，MyGo 无 cgo 是刻意保持的约束

## 发布

打 tag 即触发 `.github/workflows/release.yml`：三台 runner 各自打包，汇总后上传到 GitHub Release。

```bash
git tag v0.1.1 && git push origin v0.1.1
```
