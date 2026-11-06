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
just check        # gofmt 检查 + vet + test（CI 同款）
just build        # 三端构建
just fmt          # gofmt -w
just bench        # 基准测试（前向 + 整帧渲染，多轮）
just test-simd    # 用实验性 SIMD 跑测试（GOEXPERIMENT=simd）
just bench-simd   # 用实验性 SIMD 跑前向基准
```

## 代码地图

| 路径 | 职责 |
| --- | --- |
| `internal/model/` | 纯 Go 前向：张量运算、tokenizer、`Forward` 与 `Trace`、权重加载 |
| `internal/app/` | MyGo 界面：`app.go` 布局与交互、`stages.go` 七个阶段视图、`widgets.go` 自绘组件 |
| `tools/train.py` | PyTorch 训练并导出 `weights.json`（含参考 logits） |
| `tools/parity.py` | 导出逐层中间张量，供 Go 侧对拍 |

## 改动的验证标准

**模型或界面数字**：`go test ./internal/model` 必须全绿。动了前向逻辑要重新生成对拍夹具，否则 `TestParityEveryTensor` 会失败：

```bash
python3 tools/parity.py
go test ./internal/model -run TestParityEveryTensor -v
```

**界面布局**：无窗口渲染检查，不需要开 GUI：

```bash
RENDER=1 go test ./internal/app -run 'TestRenderSteps|TestRenderSizes'
```

截图写到 `/tmp/llmlab-step*.png`、`/tmp/size-*.png`，覆盖 1180/1500/1920 三种宽度与 64 token 长提示词。改动 UI 请在 PR 里附 1180×720 和 1500×940 两张。

**打包**：`just build` 要能出三端产物；只改代码不涉及打包配置时，至少跑一次 `just build windows/amd64` 确认交叉编译没坏。

## 提交

- 一个提交做一件事，信息用祈使句，说明「改了什么、为什么」，例如 `fix: 热力图按实际布局宽度绘制格子`
- 不要提交 `build/`、`.mygo/`、`weights.json` 之外的大文件
- 新增依赖请在 PR 里说明用途；Go 依赖只加必要项，MyGo 无 cgo 是刻意保持的约束

## 发布

打 tag 即触发 `.github/workflows/release.yml`：三台 runner 各自打包，汇总后上传到 GitHub Release。

```bash
git tag v0.1.1 && git push origin v0.1.1
```
