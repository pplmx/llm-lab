package app

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestWindowMinimumMatchesTheSmallestSnapshot ties the window's minimum size to the
// narrowest layout the snapshot suite covers.
//
// main.go pins the window to at least 1180×720, and layouts_test.go renders 1180×720
// as its smallest case. Nothing connects the two: removing the minimum lets a reader
// shrink the window to where the stage bar's buttons sit under the inspector panel
// (measured at 900×600 — "下一步" covers "← 上一步"), and widening the snapshot's
// smallest size would leave the window free to go below what is tested. Either change
// alone is silent.
//
// The pairing is what this asserts: the smallest rendered size and the window floor
// are the same numbers.
func TestWindowMinimumMatchesTheSmallestSnapshot(t *testing.T) {
	root := filepath.Join("..", "..")

	main, err := os.ReadFile(filepath.Join(root, "main.go"))
	if err != nil {
		t.Fatal(err)
	}
	const minW, minH = 1180, 720
	if !strings.Contains(string(main), "MinWidth:") {
		t.Fatal("main.go no longer sets a minimum window width; the layout is only " +
			"verified down to 1180×720, so the window must not go below it")
	}
	if !strings.Contains(string(main), "MinWidth:  1180") &&
		!strings.Contains(string(main), "MinWidth: 1180") {
		t.Fatalf("the window's minimum width is no longer %d", minW)
	}
	if !strings.Contains(string(main), "MinHeight: 720") {
		t.Fatalf("the window's minimum height is no longer %d", minH)
	}

	// And the snapshot suite must still render at exactly that size, so the
	// guarantee is demonstrated rather than assumed.
	render, err := os.ReadFile("render_sizes_test.go")
	if err != nil {
		t.Fatal(err)
	}
	want := "{1180, 720}"
	if !strings.Contains(string(render), want) {
		t.Fatalf("render_sizes_test.go no longer renders %s, so nothing shows the "+
			"layout works at the window's minimum size", want)
	}
}
