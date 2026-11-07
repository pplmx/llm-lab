package app

import (
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

// TestSpacingValuesComeFromASmallSet keeps paddings and gaps from turning into one-off
// numbers.
//
// The layouts use 0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 16 and 24 today. Each value has a
// reason — a chip's inner padding is not a card's padding — but the set itself is what keeps
// elements that should align aligning. A new layout that invents 7 or 11 makes two things
// that sat on the same line stop, and nothing on screen says which is wrong.
//
// The test parses every Gap/Padding/Margin call in the three layout files and requires each
// number to be in the list. Adding a step is allowed; it just has to be a deliberate edit
// here rather than a typo in one call site.
func TestSpacingValuesComeFromASmallSet(t *testing.T) {
	allowed := map[int]bool{
		0: true, 1: true, 2: true, 3: true, 4: true, 5: true, 6: true, 8: true,
		9: true, 10: true, 12: true, 14: true, 16: true, 24: true,
	}

	spacing := regexp.MustCompile(`\.(?:Gap|Padding|Margin)\(([0-9, ]+)\)`)
	total := 0
	for _, name := range []string{"app.go", "stages.go", "widgets.go"} {
		src, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range spacing.FindAllStringSubmatch(string(src), -1) {
			for _, field := range strings.Split(m[1], ",") {
				field = strings.TrimSpace(field)
				if field == "" {
					continue
				}
				v, err := strconv.Atoi(field)
				if err != nil {
					t.Fatalf("%s: cannot parse spacing %q", name, field)
				}
				total++
				if !allowed[v] {
					t.Fatalf("%s uses spacing %d, which is not in the set; either it is a "+
						"typo or the design gained a step that belongs in this list", name, v)
				}
			}
		}
	}
	if total < 50 {
		t.Fatalf("only found %d spacing values; the regexp probably stopped matching",
			total)
	}
	t.Logf("checked %d spacing values against %d allowed steps", total, len(allowed))
}
