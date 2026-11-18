package app

// The smallest desktop window the three-column layout is verified at. main.go uses these as
// the window's minimum size and to clamp the -w/-h flags; layouts_test.go and the snapshot
// suite render at exactly 1180×720, so anything smaller is a different layout, not a smaller
// version of this one.
const (
	MinWindowWidth  = 1180
	MinWindowHeight = 720
)

// CompactWidth is the width below which the three-column layout is replaced by the compact
// one: one column, with the stage list as a sheet and the inspector below the lesson.
//
// Why a second layout at all: the app now builds for iPhones and iPads, and a phone's screen
// is 390–430 DIP wide — a third of the desktop minimum. Before this existed, a 393-DIP render
// put the 224-DIP sidebar and the 330-DIP inspector side by side with a 161-DIP stage column
// between them, and the inspector's own cards were cut off at x=521, past the window. The
// app could compile for Android and iOS and was unusable on both.
//
// The breakpoint is the desktop minimum itself, so the two layouts cannot disagree about which
// one a width belongs to: every width that keeps the three columns is one the desktop layout
// was rendered at, and every width below it is compact. A tablet in portrait is 744–1024 DIP
// and takes the compact layout; its landscape width takes the desktop one.
const CompactWidth = MinWindowWidth

// ClampWindowSize raises a requested window size to the minimum that has a verified layout.
//
// The flags exist so a screenshot can be taken at a specific size; a value below the minimum
// would produce a window the layout was never checked against, with the prompt field
// truncating and the previous-step button covered — which is why the window itself also
// carries a minimum. Clamping the flag as well keeps the two in step: a request that reaches
// the window is already legal.
//
// Phones never reach this: the window takes the screen's size there, and the compact layout
// is what draws. The clamp is for the -w/-h flags, which a desktop renderer passes.
func ClampWindowSize(w, h int) (int, int) {
	if w < MinWindowWidth {
		w = MinWindowWidth
	}
	if h < MinWindowHeight {
		h = MinWindowHeight
	}
	return w, h
}
