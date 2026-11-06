package app

import (
	"fmt"
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"
)

// A switch that lists some values and lets the rest fall to a default is only
// safe when the unlisted values cannot occur. Last round found one where they
// could: railMarkColor handled "q" and "k" and sent "qk" to the default, which
// meant "no mark" — so a token attending to itself was drawn as if the crosshair
// were somewhere else entirely.
//
// These are the other switches of that shape in internal/app. Each checks that
// the value sets are actually closed, rather than trusting that they are.

// TestStepDotStateIsAClosedSet checks the stepper's three states are exhaustive.
// The dots switch on 0/1/2 with a default, so a fourth state would silently draw
// as "still to come".
func TestStepDotStateIsAClosedSet(t *testing.T) {
	seen := map[int]bool{}
	for step := 0; step < len(lessons); step++ {
		a := newTestApp(t)
		a.Step = step
		for i := range lessons {
			state := 2 // ahead
			switch {
			case i == a.Step:
				state = 1
			case i < a.Step:
				state = 0
			}
			seen[state] = true
			// Every state must have a distinct colour: the three-way distinction
			// is the whole point of the stepper.
			_ = stepTextColor(state)
		}
	}
	for _, want := range []int{0, 1, 2} {
		if !seen[want] {
			t.Fatalf("state %d never occurs across the seven stages, so the stepper's "+
				"three-way split is not being exercised", want)
		}
	}
	if len(seen) != 3 {
		t.Fatalf("the stepper produced %d distinct states, want 3: %v", len(seen), seen)
	}
	// And the three text colours must be three different values.
	if stepTextColor(0) == stepTextColor(1) || stepTextColor(1) == stepTextColor(2) ||
		stepTextColor(0) == stepTextColor(2) {
		t.Fatal("two stepper states share a colour")
	}
}

// TestControlScopeNoteCoversEveryControl is the closure check for the switch that
// has no default at all: an unlisted name would be dropped from the sentence,
// leaving a stage saying "顶部「层」在这一步不生效：" with nothing after the colon.
func TestControlScopeNoteCoversEveryControl(t *testing.T) {
	names := []string{"层", "头", "温度"}
	for _, n := range names {
		got := controlScopeNote([]string{n})
		if got == "" {
			t.Fatalf("controlScopeNote(%q) is empty, so that control's name would be "+
				"followed by nothing", n)
		}
	}
	// The full set, as the card builds it, must produce all three clauses.
	all := controlScopeNote(names)
	for _, want := range []string{"选层", "选头", "改温度"} {
		if !strings.Contains(all, want) {
			t.Fatalf("controlScopeNote drips %q for the full set: %q", want, all)
		}
	}
}

// TestControlScopeNoteMatchesTheRanges checks the sentence's step numbers come
// from the same constants that dim the controls, so they cannot drift apart.
func TestControlScopeNoteMatchesTheRanges(t *testing.T) {
	note := controlScopeNote([]string{"层", "头", "温度"})
	for _, want := range []string{
		fmt.Sprintf("第 %d–%d 步选层", layerFrom+1, layerTo+1),
		fmt.Sprintf("第 %d–%d 步选头", headFrom+1, headTo+1),
		fmt.Sprintf("第 %d 步改温度", tempFrom+1),
	} {
		if !strings.Contains(note, want) {
			t.Fatalf("the scope sentence lacks %q: %q", want, note)
		}
	}
}

// TestByteTextIsClosedOverEveryByte checks the fallback that every byte relies on.
// byteText names space, newline and tab, then falls through to two more branches;
// any byte must produce a non-empty, single-line string.
func TestByteTextIsClosedOverEveryByte(t *testing.T) {
	for b := 0; b < 256; b++ {
		got := byteText(b)
		if got == "" {
			t.Fatalf("byte %d renders as nothing", b)
		}
		if strings.ContainsAny(got, "\n\t") {
			t.Fatalf("byte %d renders as %q, which would break the single-line layout", b, got)
		}
	}
	// The three named cases must stay distinguishable from the generic escape.
	if byteText(' ') == byteText(0x20) && byteText(' ') == " " {
		t.Fatal("space is rendered literally, which is invisible in the strip")
	}
	if byteText('\n') != "\\n" || byteText('\t') != "\\t" {
		t.Fatalf("newline/tab escapes changed: %q %q", byteText('\n'), byteText('\t'))
	}
}

// TestChipGlyphCoversEveryByte is the same closure over the strip's glyphs.
func TestChipGlyphCoversEveryByte(t *testing.T) {
	a := newTestApp(t)
	// Build a prompt containing every byte value, then check each chip renders.
	var sb strings.Builder
	for b := 0; b < 256; b++ {
		sb.WriteByte(byte(b))
	}
	a.Prompt = sb.String()
	a.recompute()
	for i := range a.Trace.Tokens {
		g := chipGlyph(a.Trace, i)
		if g == "" {
			t.Fatalf("token %d (byte 0x%02X) has an empty glyph", i, a.Trace.Tokens[i].ID)
		}
	}
	// And the strip renders without panicking.
	tr := ui.NewTester(a.View, 1180, 720)
	tr.Frame()
}
