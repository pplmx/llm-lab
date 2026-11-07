package app

// The smallest window the layouts are verified at. main.go uses these as the window's
// minimum size and to clamp the -w/-h flags; layouts_test.go and the snapshot suite render
// at exactly 1180×720, so anything smaller is untested territory rather than a smaller
// version of the app.
const (
	MinWindowWidth  = 1180
	MinWindowHeight = 720
)

// ClampWindowSize raises a requested window size to the minimum that has a verified layout.
//
// The flags exist so a screenshot can be taken at a specific size; a value below the minimum
// would produce a window the layouts were never checked against, with the prompt field
// truncating and the previous-step button covered — which is why the window itself also
// carries a minimum. Clamping the flag as well keeps the two in step: a request that reaches
// the window is already legal.
func ClampWindowSize(w, h int) (int, int) {
	if w < MinWindowWidth {
		w = MinWindowWidth
	}
	if h < MinWindowHeight {
		h = MinWindowHeight
	}
	return w, h
}
