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

# 本地门槛：gofmt 检查 + vet + 测试。CI 另外跑 go mod tidy 检查、
# internal/model 的 -race、以及 GOEXPERIMENT=simd —— 想一次对齐跑 just check-ci
check:
    @files="$(gofmt -l .)"; if [ -n "$files" ]; then echo "未格式化："; echo "$files"; exit 1; fi
    go vet ./...
    go test ./...

# CI 的全部检查，逐条对应 .github/workflows/ci.yml。比 check 慢大约一分钟
# （SIMD 一遍要重跑整个测试集），所以两者分开：日常提交用 check，动到模块依赖、
# 并发或 SIMD 内核时用这个，避免「本地全绿、CI 变红」。
check-ci: check
    go mod tidy
    git diff --exit-code -- go.mod go.sum
    go test -race -timeout 300s ./internal/model/
    GOEXPERIMENT=simd go test -timeout 300s ./...

# 格式化全部 Go 代码
fmt:
    gofmt -w .

# 基准测试：模型前向 + 整帧渲染（多轮，便于 benchstat 对比）
bench:
    go test ./internal/model ./internal/app -run '^$' -bench . -benchmem -count 5

# 复现 README「性能」一节引用的数字（单核 / 4 核，标量与 SIMD）。
# 上面那个 bench 用机器默认 GOMAXPROCS（这台 128），量与 README 引用的不是同一组条件。
bench-cores:
    just _bench-cores ""
    just _bench-cores "GOEXPERIMENT=simd"

# 参数是环境变量前缀，空串表示标量构建。
_bench-cores envprefix:
    @echo "→ 前向基准（{{ if envprefix == "" { "标量" } else { "SIMD" } }}），单核与 4 核"
    {{ envprefix }} GOMAXPROCS=1 go test ./internal/model -run '^$' -bench BenchmarkForward -count 5
    {{ envprefix }} GOMAXPROCS=4 go test ./internal/model -run '^$' -bench BenchmarkForward -count 5
    @echo "→ 整帧渲染 1500×940（4 核，与 README 那个数字同一条件）"
    GOMAXPROCS=4 go test ./internal/app -run '^$' -bench BenchmarkView -count 5

# 用实验性 SIMD 跑测试（Go 1.27+，amd64 走 AVX，arm64 走 NEON；见 README 性能一节）
test-simd:
    GOEXPERIMENT=simd go test ./...

# 用实验性 SIMD 跑前向基准
bench-simd:
    GOEXPERIMENT=simd go test ./internal/model -run '^$' -bench . -benchmem -count 5

# 用实验性 SIMD 构建（默认三端全量；也可指定平台：just build-simd windows/amd64）
build-simd target=all_platforms:
    @echo "→ 构建平台（SIMD）：{{target}}"
    GOEXPERIMENT=simd go tool mygo build -platform "{{target}}"

# 用实验性 SIMD 本机运行
run-simd:
    GOEXPERIMENT=simd go run .

# 环境自检（报 webkit 缺失可忽略，原生 UI 不需要）
doctor:
    -go tool mygo doctor

# 清理构建产物
clean:
    rm -rf build

# 渲染全部截图到指定目录（默认 /tmp/llm-lab-shots）
snapshots dir="/tmp/llm-lab-shots":
    RENDER=1 RENDER_DIR="{{dir}}" go test ./internal/app -run 'TestRenderSteps|TestRenderSizes' -count=1
    @echo "截图已写入 {{dir}}（$(ls {{dir}}/*.png | wc -l) 张）"

# 重新生成 README 引用的 6 张截图（写进 docs/）
#
# 它们是仓库资产，不是每次运行现生成的，所以界面变了没人更新——第 229 轮发现六张
# 全在展示一个已经不存在的界面：侧栏从 0 编号、没有数据流卡片、没有常见误解块。
# 改完界面后跑一次这个。
docs:
    RENDER_DOCS=1 go test ./internal/app -run TestRenderDocs -count=1
    @echo "docs/ 已更新（$(ls docs/*.png | wc -l) 张）"

# 与另一份截图逐字节对比：just snapshots-diff <基线目录> [新目录]
#
# 改界面时先说清楚预期：不改变视觉的改动应当 0 差异，故意改视觉的应当只差在
# 动手的那几个阶段。两者都靠这个配方核对，比手写 cmp 循环可靠——手写时漏掉
# 一张、或者把空目录当基线（会误报「全部变化」）都不会有人发现。
snapshots-diff base new="/tmp/llm-lab-shots":
    #!/usr/bin/env bash
    if [ ! -d "{{base}}" ]; then echo "基线目录不存在：{{base}}"; exit 1; fi
    if [ -z "$(ls -A {{base}} 2>/dev/null)" ]; then echo "基线目录是空的：{{base}}"; exit 1; fi
    changed=0 missing=0 added=0
    for f in "{{base}}"/*.png; do
        n=$(basename "$f")
        if [ ! -f "{{new}}/$n" ]; then
            echo "缺失：$n"; missing=$((missing+1))
        elif ! cmp -s "$f" "{{new}}/$n"; then
            echo "变化：$n"; changed=$((changed+1))
        fi
    done
    for f in "{{new}}"/*.png; do
        n=$(basename "$f")
        if [ ! -f "{{base}}/$n" ]; then echo "新增：$n"; added=$((added+1)); fi
    done
    total=$(ls "{{base}}"/*.png 2>/dev/null | wc -l)
    echo "基线 $total 张：$changed 变化、$missing 缺失、$added 新增"
    if [ "$missing" -gt 0 ]; then exit 1; fi

# 列出已有产物及体积
artifacts:
    @files="$(du -sh build/*/* 2>/dev/null | sort -k2)"; \
        if [ -n "$files" ]; then echo "$files"; else echo "build/ 为空，先运行 just build"; fi

_build target:
    @echo "→ 构建平台：{{target}}"
    go tool mygo build -platform "{{target}}"
