## 改了什么

<!-- 一句话说明动机，再列出改动点 -->

-

## 怎么验证的

- [ ] `just test` 通过
- [ ] `just build` 三端产物正常
- [ ] UI 有改动：附上 1180×720 与 1500×940 的截图（`RENDER=1 go test ./internal/app -run 'TestRenderSteps|TestRenderSizes'`，图在 `/tmp`）
- [ ] 模型有改动：`go test ./internal/model -run TestParityEveryTensor` 通过

## 备注

<!-- 已知限制、后续工作、需要 reviewer 特别注意的地方 -->
