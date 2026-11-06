# LLM Lab · Transformer 解剖台

用 **MyGo** 的原生 UI（纯 Go、GPU 绘制、没有 WebView、没有 HTML/JS）写的一个 Transformer 架构学习工具：输入一句话，然后逐层剥开它——分词、嵌入、Q/K/V、注意力矩阵、残差与 LayerNorm、FFN、logits 与采样，每一步都能看到真实的数字。

![注意力矩阵](docs/attention.png)

## 为什么用 MyGo

- **一个 Go 二进制就够**：模型、前向计算、界面全在一个进程里，11 MB，没有 Python、没有 Node、没有浏览器内核。
- **界面就是数据**：`view` 是状态的函数，改一个滑块整帧重画，热力图/向量条带用 `ui.Box(...).Draw` 直接画在 GPU 上。
- **秒开**：和 Electron 系的教学工具比，启动时间与内存占用低一个数量级。

## 快速开始

需要 Go 1.27+（go.mod 要求 1.27.3，低版本会自动拉工具链），Linux 上还需要 GTK 3（`sudo apt install libgtk-3-0`）。

```bash
# 直接运行（默认 1500×940）
go run .

# 小屏：可以指定窗口尺寸，最小 1180×720
go run . -w 1180 -h 720

# 开发模式（改代码自动重启）
go tool mygo dev
```

## 多端构建

构建统一走 **just**（[安装](https://github.com/casey/just#installation)：`brew install just` / `winget install Casey.Just` / `scoop install just` / `cargo install just`）：

```bash
just            # 一次性构建 Windows + Linux + macOS
just build      # 同上；也可只构建指定平台
just linux      # linux/amd64 + linux/arm64
just windows    # windows/amd64
just mac        # darwin/universal（Intel + Apple Silicon）
just artifacts  # 列出已有产物与体积
just clean      # 清空 build/
```

默认产物（输出到 `build/<平台>/`）：

| 目标 | 产物 |
| --- | --- |
| `linux/amd64`、`linux/arm64` | 可执行文件、`llm-lab_<ver>_<arch>.deb`、带 `install.sh` 的 `.tar.gz`、`PKGBUILD` |
| `windows/amd64` | `LLM Lab.exe`；在 Windows 上构建时自动补 `LLM Lab-setup-<ver>-amd64.exe`（NSIS） |
| `darwin/universal` | `LLM Lab.app`（同时含 arm64 与 amd64）；在 macOS 上构建时额外产出 `.dmg`，可配置签名/公证 |

MyGo 不依赖 cgo，所以**在任意系统上都能交叉编译三端**，不需要 macOS 机器也能出 `.app`。只有三件事必须在 macOS 本机完成：`.dmg` 打包、Developer ID 签名、公证。Windows 安装包需要 `makensis`，Linux/macOS 上没装就跳过，不影响 exe 本体。

单平台构建不会清掉其他平台的产物，`just linux` 之后 `build/windows-amd64/` 仍在。Windows 上运行 just 需要 Git Bash（justfile 用 bash 作为 shell）。

不想装 just 时，等价命令是：

```bash
go tool mygo build -platform linux/amd64,linux/arm64,windows/amd64,darwin/universal
```

## 开发

```bash
just fmt      # gofmt -w
just check    # gofmt 检查 + go vet + go test（CI 同款）
```

改模型或界面后的验证标准（对拍、无窗口截图）见 [CONTRIBUTING.md](CONTRIBUTING.md)，版本变更见 [CHANGELOG.md](CHANGELOG.md)。CI 在 Linux/macOS/Windows 三个平台跑测试并各打一次包；推 `v*` tag 会自动建 release 并上传三端安装包。

启动后：

| 操作 | 效果 |
| --- | --- |
| 顶部输入框 | 改提示词，实时重跑前向 |
| 层 / 头 分段控件 | 切换观察第几层、第几个注意力头 |
| 温度滑块 | 实时重算 softmax 分布 |
| 点击 token 胶囊 | 右侧检查器显示该 token 的注意力分配与向量统计 |
| 悬停热力图单元格 | 显示 (query, key, 分数, 权重)，点一下选中该行 |
| 采样并续写 | 按当前温度抽一个 token 追加到提示词 |

## 七个阶段与代码的对应关系

| 界面阶段 | 公式 | 代码 |
| --- | --- | --- |
| 1 分词 | `id = byte(prompt)` | `internal/model/tokenizer.go` |
| 2 嵌入 | `x = E[id] + P[t]` | `gpt.go` 的 `Forward` 开头 |
| 3 Q/K/V 与分数 | `score = q·k / √d_head`，`s ≤ t` | `Forward` 的注意力循环 |
| 4 Softmax 与加权 | `α = softmax(score)`，`out = Σ α·v` | 同上 |
| 5 残差 / LN | `x ← x + Attn(LN₁(x))`；`x ← x + FFN(LN₂(x))` | `tensor.go: layerNorm` |
| 6 FFN | `W₂ · GELU(W₁ · LN(x))` | `tensor.go: gelu` |
| 7 输出与采样 | `p = softmax(logits / T)` | `weights.go: Probabilities` |

每次 `Forward` 会把每个中间张量录进 `model.Trace`（`gpt.go`），界面只读这份缓存——所以拖动滑块、切换层/头都不需要重算整个网络。

## 模型

内置的是一个**字符/字节级微型 GPT**，参数量 25.3 万：

```
vocab 256 · d_model 96 · 2 层 · 4 头 · d_ff 384 · ctx 64
```

权重由 `tools/train.py` 用 PyTorch（CPU，约 45 秒）在 61 句关于 Transformer 的英文句子训练 600 步得到，导出为 `internal/model/weights.json` 并 `go:embed` 进二进制。

```bash
python3 tools/train.py                 # 复现权重
python3 tools/train.py --steps 2000    # 更收敛、分布更尖
go run . -weights /path/to/other.json  # 换一份权重（需为 vocab 256 的 byte-level 同架构）
```

模型输出的是「像语料但不是语料」的英文，这一点是刻意的：这是一个观察数值流动的工具，不是一个能聊天的模型。

## 正确性

`go test ./...` 覆盖：

- softmax 每行和为 1、因果掩码让上三角严格为 0、LayerNorm 输出零均值单位方差
- 同一输入两次前向结果逐位一致
- tokenizer 中英文往返一致；截断不会切在多字节字符中间
- **逐张量与 PyTorch 对齐**：`tools/parity.py` 把同一份 `weights.json` 在 PyTorch 里跑一遍，导出 46 token 提示词的全部中间结果（embedding、每层 Norm1、每个头的 Q/K/V、score、attn、头输出、AttnProj、Norm2、GELU 前后、FFN 输出、最终 Norm、logits），Go 侧 `TestParityEveryTensor` 逐项比对，所有张量最大误差 1.7e-5，logits 1.7e-5（GELU 与 softmax 的 exp 都用查表近似，误差分别 ≤ 4e-7、2.4e-6，由 `TestGeluMatchesErf`、`TestExpNegMatchesExp` 锁定；标量与 SIMD 构建结果一致）。这比只比最终 logits 严格得多：头切片顺序、拼接布局、GELU 变体、LayerNorm 的 eps 出错都会被抓到
- 单头贡献之和严格等于该层的注意力输出（`TestHeadProjSumsToAttnProj`），界面上「本头贡献 Δ_h」不是拿总和冒充的
- 用 `ui.Tester` 模拟真实点击，验证阶段切换

界面本身也能在没有窗口的情况下渲染成 PNG（`ui.Render`）：`RENDER=1 go test ./internal/app -run 'TestRenderSteps|TestRenderSizes'` 会把七个阶段、1180/1500/1920 三种窗口宽度、以及 64 token 长提示词的截图写到 `/tmp`，用于回归检查布局与溢出。

性能（AMD EPYC 7742，Go 1.27，单核 / 4 核）：前向 16 token 1.4 / 1.2 ms、64 token 6.2 / 2.4 ms；用实验性 SIMD 构建（`GOEXPERIMENT=simd just build`，Go 1.27 的 `simd` 包，amd64 走 AVX、arm64 走 NEON）后 16 token 0.53 / 0.46 ms、64 token 2.5 / 1.4 ms。矩阵乘按行、注意力按头并行，最多 8 个 worker；整帧 `ui.Render` 1500×940 ≈ 9.4 ms（CPU 光栅化，热力图只画因果掩码内的格子）。基准：`just bench`、`just bench-simd`。

## 已知限制

- 权重很小，输出质量只够演示；中文输入可以分词和可视化，但模型没在中文上训练，续写没有意义
- 只实现了前向传播，没有反向传播和训练过程可视化
- 上下文 64 token，超出后截断到最近 64 个（左侧栏状态行会提示）；截断会前进到 UTF-8 字符边界，不会从半个汉字开始
- 注意力矩阵依赖鼠标悬停，没有键盘导航
- Linux 上若无 GPU，MyGo 会退回 CPU 绘制（日志提示 `native UI draws without the GPU`），界面功能不受影响

## 目录

```
main.go                  窗口与权重加载
justfile                 多端构建、测试、运行命令
internal/model/          纯 Go 模型：tokenizer / tensor / gpt / weights
internal/model/testdata/ PyTorch 逐层对拍夹具（parity.json.gz）
internal/app/            MyGo 界面：布局、组件、七个阶段的视图、检查器
tools/train.py           PyTorch 训练并导出 weights.json
tools/parity.py          PyTorch 逐层中间结果导出，用于对拍
docs/                    截图
.github/                 CI、release、dependabot 配置
.vscode/                 共享的编辑器配置与调试配置
```

仓库约定：LF 换行 + `.editorconfig` 缩进（Go 用 tab，justfile 4 空格），
`weights.json` 与对拍夹具在 diff 与语言统计中折叠，构建产物不入库。
