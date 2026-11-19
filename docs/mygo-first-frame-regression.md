# Upstream report: MyGo v0.10.0 makes the first frame 3.8× slower

写给 MyGo 上游（`github.com/egoist/mygo`）的问题报告底稿，也写给本仓库的人：为什么 `just check-ci` 的 SIMD 那一步从 300s 放到了 600s。

**结论**：升到 v0.10.0 后，**同一个 view 在一个新窗口里的第一帧**从 13.2ms 变成 50.3ms；**同一个窗口的第二帧起没有变化**（0.91ms → 0.96ms）。两者交错测三遍，差异稳定。

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

v0.10.0 新增了 `ui/rasterscale.go`（`glyphSpace` / `rasterScale` / `scaleWatches`），按元素记录「文字按什么 scale 光栅化」，以支持缩放时清晰渲染。`glyphSpace` 每个文字元素每帧都要走一次，并在首次绘制时建立 watch。这是新代码里与「首帧」最吻合的一块，但**我没有逐行定位到具体那几行**，所以这份报告给的是现象与 profile，不是补丁。

## 对本仓库的影响

- 测试里每个 `ui.NewTester` / `ui.Render` 都是一个新窗口，所以整个 `internal/app` 跟着变慢
- `GOEXPERIMENT=simd go test -timeout 300s ./...` 实测从 216s 涨到约 380s，**超出 CI 原来的 300s**；把本轮新增的测试全部 `-skip` 掉仍然超时，所以红来自升级本身
- 处理：只把那一个数字按实测放到 600s（`justfile`、`ci.yml`），其余标准不动

## 想要上游确认的

1. 首帧多出来的 ~37ms 是否是 `rasterscale.go` 的设计代价（每个元素的 watch 与首次 scale 计算），还是可以按窗口懒建/复用
2. 稳态不慢，说明缓存有效；问题是**首帧成本随文字元素数量增长**——本应用一帧有 216–263 个文字元素（按阶段实测）
3. 如果这是预期的，能否给个「关掉逐元素 scale watch」的开关，让不需要缩放的桌面应用退回旧路径
