# LLM Lab —— 构建、测试与运行
#
#   just              三端全量构建（Linux + Windows + macOS）
#   just build        同上；也可指定平台：just build windows/amd64
#   just linux        只构建 Linux（amd64 + arm64）
#   just windows      只构建 Windows（amd64）
#   just mac          只构建 macOS（universal，Intel + Apple Silicon）
#   just run          本机运行；just dev 热重载
#   just test         静态检查 + 全部测试
#   just clean        清掉 build/
#
# MyGo 不依赖 cgo，任意系统都能交叉编译三端；只有 macOS 的 .dmg
# 和签名需要在 macOS 上完成。

set shell := ["bash", "-uc"]

# go.mod 要求 Go 1.27.3。本地版本更低时交由 Go 自动下载工具链；
# 想固定版本就 GOTOOLCHAIN=go1.27.3 just build。
export GOTOOLCHAIN := env("GOTOOLCHAIN", "auto")

linux_platforms   := "linux/amd64,linux/arm64"
windows_platforms := "windows/amd64"
mac_platforms     := "darwin/universal"
all_platforms     := linux_platforms + "," + windows_platforms + "," + mac_platforms

# 默认目标：三端全量构建
default: build

# 构建指定平台，逗号分隔（默认三端全量）
build target=all_platforms: (_build target)

# Linux：amd64 + arm64，产出可执行文件、.deb、tar.gz、PKGBUILD
linux: (_build linux_platforms)

# Windows：amd64，产出 exe；在 Windows 上会自动补 NSIS 安装包
windows: (_build windows_platforms)

# macOS：universal，一个 .app 同时支持 Intel 与 Apple Silicon
mac: (_build mac_platforms)

alias macos := mac
alias win := windows

# 本机直接运行
run:
    go run .

# 热重载开发
dev:
    go tool mygo dev

# 静态检查 + 全部测试
test:
    go vet ./...
    go test ./...

# CI 同款：gofmt 检查 + vet + 测试
check:
    @files="$(gofmt -l .)"; if [ -n "$files" ]; then echo "未格式化："; echo "$files"; exit 1; fi
    go vet ./...
    go test ./...

# 格式化全部 Go 代码
fmt:
    gofmt -w .

# 基准测试：模型前向 + 整帧渲染（多轮，便于 benchstat 对比）
bench:
    go test ./internal/model ./internal/app -run '^$' -bench . -benchmem -count 5

# 用实验性 SIMD 跑测试（Go 1.27+，amd64 走 AVX，arm64 走 NEON；见 README 性能一节）
test-simd:
    GOEXPERIMENT=simd go test ./...

# 用实验性 SIMD 跑前向基准
bench-simd:
    GOEXPERIMENT=simd go test ./internal/model -run '^$' -bench . -benchmem -count 5

# 环境自检（报 webkit 缺失可忽略，原生 UI 不需要）
doctor:
    -go tool mygo doctor

# 清理构建产物
clean:
    rm -rf build

# 列出已有产物及体积
artifacts:
    @du -sh build/*/* 2>/dev/null | sort -k2 || echo "build/ 为空，先运行 just build"

_build target:
    @echo "→ 构建平台：{{target}}"
    go tool mygo build -platform "{{target}}"
