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

// Tokenize turns a prompt into byte tokens. Invalid UTF-8 bytes are kept
// as-is: this is a byte-level tokenizer, so a malformed prompt must not
// expand into replacement-character bytes. Valid multi-byte runes share one
// rune index in Start and one Rune value across all their byte tokens.
func Tokenize(s string) []Token {
	tokens := make([]Token, 0, len(s))
	runeIndex := 0
	for i := 0; i < len(s); {
		r, n := utf8.DecodeRuneInString(s[i:])
		if r == utf8.RuneError && n == 1 {
			b := s[i]
			tokens = append(tokens, Token{
				ID:    int(b),
				Byte:  b,
				Text:  renderByte(b),
				Start: runeIndex,
				Rune:  0,
			})
			i++
			runeIndex++
			continue
		}
		for k := 0; k < n; k++ {
			b := s[i+k]
			tokens = append(tokens, Token{
				ID:    int(b),
				Byte:  b,
				Text:  renderByte(b),
				Start: runeIndex,
				Rune:  r,
			})
		}
		i += n
		runeIndex++
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

// DecodeBytes turns ids back into a string, replacing each invalid UTF-8 byte
// with U+FFFD so a preview never shows a broken rune.
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
// in the middle of a multi-byte character. A non-positive context window
// yields no tokens.
func Truncate(ids []int, ctx int) []int {
	if ctx <= 0 {
		return nil
	}
	if len(ids) <= ctx {
		return ids
	}
	start := len(ids) - ctx
	for start < len(ids) && byte(ids[start])&0xC0 == 0x80 {
		start++
	}
	return ids[start:]
}
