package model

import "testing"

// TestTokenizeKeepsInvalidBytes verifies that a byte-level tokenizer does not
// expand an invalid byte into the three bytes of U+FFFD.
func TestTokenizeKeepsInvalidBytes(t *testing.T) {
	s := string([]byte{0xFF, 'a', 0x80})
	tokens := Tokenize(s)
	want := []int{0xFF, 'a', 0x80}
	if len(tokens) != len(want) {
		t.Fatalf("%d tokens, want %d", len(tokens), len(want))
	}
	for i, id := range want {
		if tokens[i].ID != id || int(tokens[i].Byte) != id {
			t.Fatalf("token %d = id %d byte %d, want %d", i, tokens[i].ID, tokens[i].Byte, id)
		}
	}
}

// TestTokenizeStartIsRuneOffset checks the field's documented contract: all
// byte tokens of one rune share that rune's index, not the byte offset.
func TestTokenizeStartIsRuneOffset(t *testing.T) {
	const s = "éa中"
	tokens := Tokenize(s)
	want := []int{0, 0, 1, 2, 2, 2}
	if len(tokens) != len(want) {
		t.Fatalf("%d tokens, want %d", len(tokens), len(want))
	}
	for i, start := range want {
		if tokens[i].Start != start {
			t.Fatalf("token %d start = %d, want %d", i, tokens[i].Start, start)
		}
	}
}

// TestTruncateNonPositiveContext documents the empty-result edge case instead
// of letting a negative context slice out of bounds.
func TestTruncateNonPositiveContext(t *testing.T) {
	for _, ctx := range []int{0, -1} {
		if got := Truncate([]int{1, 2, 3}, ctx); len(got) != 0 {
			t.Fatalf("Truncate(ctx=%d) = %v, want empty", ctx, got)
		}
	}
}
