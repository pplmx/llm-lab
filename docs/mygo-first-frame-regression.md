# Upstream report: MyGo v0.10.0 makes the first frame 3.8× slower

写给 MyGo 上游（`github.com/egoist/mygo`）的问题报告底稿，也写给本仓库的人：为什么 `just check-ci` 的 SIMD 那一步从 300s 放到了 600s。

**结论**：**同一个 view 在一个新窗口里的第一帧**从 v0.8.3 的 13.2ms 变成 50.3ms（v0.10.0）；**同一个窗口的第二帧起没有变化**（0.91ms → 0.96ms）。两者交错测三遍，差异稳定。

**已提到上游**：[egoist/mygo#2085](https://github.com/egoist/mygo/issues/2085)。写在那里的一条更细的结论，本仓库也记着：**回归发生在 v0.8.3 → v0.9.1，不是 v0.10.0**——v0.9.1 就让每个 `ui.Tester` 建自己的文字系统（#1730），v0.10.0 只是在这个基础上再高一点。

## 复现

`llm-lab` 的 `internal/app` 里，`BenchmarkFirstFrame` 是 `ui.Render(a.View, 1500, 940, 1)`（每次都是新窗口），`BenchmarkSteadyFrame` 是同一个 `ui.Tester` 反复 `Frame()`：

```go
func BenchmarkFirstFrame(b *testing.B) {
	a := newTestApp(b)
	a.Step = 0
	for i := 0; i < b.N; i++ {
		ui.Render(a.View, 1500, 940, 1)
	}
}

func BenchmarkSteadyFrame(b *testing.B) {
	a := newTestApp(b)
	a.Step = 0
	tr := ui.NewTester(a.View, 1500, 940)
	tr.Frame() // 预热
	for i := 0; i < b.N; i++ {
		tr.Frame()
	}
}
```

`GOMAXPROCS=4`，`-benchtime 10x`，两个 commit 交错跑三遍（机器上另有重负载作业，所以必须交错）：

| 轮次 | v0.8.3 首帧 | v0.10.0 首帧 | v0.8.3 稳态 | v0.10.0 稳态 |
| --- | --- | --- | --- | --- |
| 1 | 13.50 ms | 50.26 ms | 0.90 ms | 0.97 ms |
| 2 | 13.26 ms | 50.45 ms | 0.92 ms | 0.94 ms |
| 3 | 13.22 ms | 50.29 ms | 0.93 ms | 0.98 ms |

**首帧 3.8×，稳态持平。**

同一份 view、每批 10 个新 Tester，三个版本各测一遍（本应用 216–263 个文字元素、机器空闲时交错）：

| 版本 | 第 1–10 个 | 11–20 | 21–30 | 31–40 | 41–50 |
| --- | --- | --- | --- | --- | --- |
| v0.8.3 | 20.1 ms¹ | 12.6 ms | 11.3 ms | 11.1 ms | **11.6 ms** |
| v0.9.1 | 72.3 ms | 52.1 ms | 46.7 ms | 45.7 ms | **52.2 ms** |
| v0.10.0 | 65.9 ms | 48.9 ms | 50.8 ms | 46.4 ms | **60.3 ms** |

¹ 第一批含一次性启动，三行都是；后面的批次才是稳态值。

**这张表说明了回归的归属**：v0.9.1 已经在 52ms，v0.10.0 只是略高。而且它是**平的**——第 21–50 个稳定在 46–52ms，不像 #1944 那样逐个变慢。本应用不调 `ui.RegisterFont`，所以 #1944 的 fontconfig 累积机制在这里不适用；这里付的是「每个新窗口建一套文字系统」本身的代价。

`ui.NewTester` + 一次 `Frame()` 是同一个现象：8.1ms → 47.0ms。

## CPU profile 指向哪里

`go tool pprof -top` 打首帧基准（v0.10.0，852 个样本）：

```
0.76s 32.34%  runtime.cgocall
0.31s 13.19%  <unknown>
0.20s  8.51%  [libc.so.6]
0.16s  6.81%  internal/raster.(*Image).RGBA
0.10s  4.26%  runtime.memmove
```

`cgocall` 全部来自 `purego.syscall_SyscallN`，其中 97% 经 `pangoEngine.shape`（0.47s，20%）与 `pangoEngine.draw`（0.24s，10%）。也就是说时间花在**文字整形与光栅化**上，而且只在第一帧——第二帧起字形缓存命中，完全消失。

**这里有一条先前的推断被数据推翻了，记下来。** 最初我把原因归给 v0.10.0 新增的 `ui/rasterscale.go`（`glyphSpace` / `rasterScale` / `scaleWatches`，按元素记录文字按什么 scale 光栅化），因为它是那次升级里与文字最相关的新代码。但 v0.9.1 的对照表明回归在 v0.9.1 就已经发生——那时 `rasterscale.go` 还不存在。所以真正的机制是 #1730 引入的**每个 Tester 一套文字系统**，`rasterscale.go` 不是原因。

profile 只能说明时间花在文字整形与光栅化上（`pangoEngine.shape` / `draw`），**没有定位到具体那几行**；这份报告给的是现象、版本归属与 profile，不是补丁。

## 对本仓库的影响

- 测试里每个 `ui.NewTester` / `ui.Render` 都是一个新窗口，所以整个 `internal/app` 跟着变慢
- `GOEXPERIMENT=simd go test -timeout 300s ./...` 实测从 216s 涨到约 380s，**超出 CI 原来的 300s**；把本轮新增的测试全部 `-skip` 掉仍然超时，所以红来自升级本身
- 处理：只把那一个数字按实测放到 600s（`justfile`、`ci.yml`），其余标准不动

## 想要上游确认的

1. 是否提供「一个测试包共用一个文字系统」的方式（#1944 自己提的「A second point」）。多数测试套件并不并行跑 Tester，却为这份隔离付了全部代价，这是最有用的一条
2. 首帧多出来的 ~37ms 里，多少是新建一套 Pango 引擎的固定开销、多少随文字元素数量增长（本应用一帧有 216–263 个文字元素，按阶段实测）
3. 如果这是设计代价，能否写进文档，让项目知道要按它来定测试超时
