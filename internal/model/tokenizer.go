package model

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// Token is one byte of the UTF-8 encoded prompt. Byte-level tokenization keeps
// the vocabulary tiny and makes multi-byte scripts visible in the UI: a CJK
// character is three tokens.
type Token struct {
	ID    int
	Byte  byte
	Text  string // the rendered form: a letter, or an escape such as \n
	Start int    // rune offset in the prompt
	Rune  rune   // the rune this byte belongs to, for grouping
}

// Tokenize turns a prompt into byte tokens.
func Tokenize(s string) []Token {
	tokens := make([]Token, 0, len(s))
	for i, r := range s {
		var buf [4]byte
		n := utf8.EncodeRune(buf[:], r)
		for k := 0; k < n; k++ {
			b := buf[k]
			tokens = append(tokens, Token{
				ID:    int(b),
				Byte:  b,
				Text:  renderByte(b),
				Start: i,
				Rune:  r,
			})
		}
	}
	return tokens
}

func renderByte(b byte) string {
	switch b {
	case ' ':
		return "␣"
	case '\n':
		return "\\n"
	case '\t':
		return "\\t"
	}
	if b >= 32 && b < 127 {
		return string(rune(b))
	}
	return fmt.Sprintf("\\x%02x", b)
}

// DecodeBytes turns ids back into a string, dropping invalid UTF-8 tails so a
// preview never shows broken runes.
func DecodeBytes(ids []int) string {
	buf := make([]byte, len(ids))
	for i, id := range ids {
		buf[i] = byte(id)
	}
	s := string(buf)
	if utf8.Valid(buf) {
		return s
	}
	var b strings.Builder
	for len(buf) > 0 {
		r, n := utf8.DecodeRune(buf)
		if r == utf8.RuneError && n == 1 {
			b.WriteString("�")
			buf = buf[1:]
			continue
		}
		b.WriteRune(r)
		buf = buf[n:]
	}
	return b.String()
}

// Truncate cuts ids to the model's context window. When it has to drop bytes
// it advances to the next UTF-8 lead byte, so a truncated prompt never starts
// in the middle of a multi-byte character.
func Truncate(ids []int, ctx int) []int {
	if len(ids) <= ctx {
		return ids
	}
	start := len(ids) - ctx
	for start < len(ids) && byte(ids[start])&0xC0 == 0x80 {
		start++
	}
	return ids[start:]
}
