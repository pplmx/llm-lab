package model

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestValidateRejectsInvalidConfig pins the shape checks that protect
// Forward from malformed -weights files. Before these checks, n_head=0
// divided by zero and a missing final LayerNorm ran off the end of a slice.
func TestValidateRejectsInvalidConfig(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Weights)
	}{
		{"n_head zero", func(w *Weights) { w.Config.NHead = 0 }},
		{"n_head not a divisor of d_model", func(w *Weights) { w.Config.NHead = 5 }},
		{"vocab is not byte-sized", func(w *Weights) { w.Config.Vocab = 128 }},
		{"d_model zero", func(w *Weights) { w.Config.DModel = 0 }},
		{"d_model negative", func(w *Weights) { w.Config.DModel = -96 }},
		{"n_layer zero", func(w *Weights) { w.Config.NLayer = 0 }},
		{"d_ff zero", func(w *Weights) { w.Config.DFF = 0 }},
		{"ctx zero", func(w *Weights) { w.Config.Ctx = 0 }},
		{"lnf_g missing", func(w *Weights) { w.LNFG = nil }},
		{"lnf_b short", func(w *Weights) { w.LNFB = w.LNFB[:len(w.LNFB)-1] }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w := Random(DefaultConfig, 1)
			tc.mutate(w)
			if err := w.validate(); err == nil {
				t.Fatal("validate accepted malformed weights")
			}
		})
	}
}

// TestParseRejectsEmptyConfig covers a JSON file with a config object that
// omits every shape field; it must fail before Forward can divide by zero.
func TestParseRejectsEmptyConfig(t *testing.T) {
	if _, err := Parse([]byte(`{"config":{"vocab":256}}`), "test"); err == nil {
		t.Fatal("Parse accepted a weights file with no model shape")
	}
}

// TestParseErrorsNameTheActualProblem checks how a bad weights file reads, not
// just that it is rejected. main.go calls log.Fatalf on a -weights load error, so
// this string is the only thing the reader ever sees of the failure.
func TestParseErrorsNameTheActualProblem(t *testing.T) {
	cases := []struct {
		name string
		body string
		// want appears in the message
		want []string
		// wantNot must not appear: these were the old, misleading phrasings
		wantNot []string
	}{
		{
			name:    "empty file",
			body:    "",
			want:    []string{"empty", "weights.json"},
			wantNot: []string{"unexpected end"},
		},
		{
			name:    "truncated json",
			body:    `{"config":`,
			want:    []string{"not valid JSON", "byte 10"},
			wantNot: []string{"empty"},
		},
		{
			name: "no recognisable keys at all",
			body: `{"my_config":{"vocab":256},"my_tok_embed":[]}`,
			want: []string{"no configuration", "config", "tok_embed"},
			// "vocab is 0" reads as a wrong value when the whole schema is wrong.
			wantNot: []string{"vocab is 0"},
		},
		{
			name: "complete config, no tensors",
			body: `{"config":{"vocab":256,"d_model":96,"n_layer":2,"n_head":4,"d_ff":384,"ctx":64}}`,
			// The largest missing tensor is named first, with its expected size.
			want:    []string{"tok_embed", "24576", "vocab x d_model"},
			wantNot: []string{"lnf_g"},
		},
		{
			name:    "wrong vocab",
			body:    `{"config":{"vocab":100,"d_model":96,"n_layer":2,"n_head":4,"d_ff":384,"ctx":64}}`,
			want:    []string{"vocab is 100", "want 256"},
			wantNot: []string{"empty"},
		},
	}
	dir := t.TempDir()
	for _, c := range cases {
		p := filepath.Join(dir, "weights.json")
		if err := os.WriteFile(p, []byte(c.body), 0o644); err != nil {
			t.Fatal(err)
		}
		_, err := Load(p)
		if err == nil {
			t.Fatalf("%s: accepted a file it should reject", c.name)
		}
		msg := err.Error()
		for _, w := range c.want {
			if !strings.Contains(msg, w) {
				t.Errorf("%s: message lacks %q: %s", c.name, w, msg)
			}
		}
		for _, w := range c.wantNot {
			if strings.Contains(msg, w) {
				t.Errorf("%s: message still says %q: %s", c.name, w, msg)
			}
		}
		// No assertion that every message names the file: validate() has no
		// path to name, and main.go already prefixes "weights <path>: " when it
		// reports a -weights failure. Only the parse errors carry it, because
		// Parse is what a library caller would see.
	}
}

// TestLoadErrorsSayWhatWasBeingLoaded covers the one error path Load produces on its own.
//
// Load returns os.ReadFile's error untouched for a missing or unreadable file, and that error
// names the path but not what the program was doing with it: "open /x.json: no such file or
// directory" would be printed for a weights file, an image and a CSV alike. main.go used to
// supply the missing context by prefixing the path itself, which then doubled the prefix on
// every error Parse returns — "weights /x.json: weights: the file parsed but has no
// configuration" — so the prefix could not simply stay there.
//
// Every error this function returns now names the operation, which is what lets a caller print
// it verbatim. The test pins both halves: the read failure says "read weights", and the parse
// failures keep their own single prefix.
func TestLoadErrorsSayWhatWasBeingLoaded(t *testing.T) {
	dir := t.TempDir()
	missing := filepath.Join(dir, "absent.json")
	_, err := Load(missing)
	if err == nil {
		t.Fatal("Load accepted a path that does not exist")
	}
	msg := err.Error()
	if !strings.Contains(msg, "read weights") {
		t.Fatalf("the error does not say what was being read: %q", msg)
	}
	if !strings.Contains(msg, missing) {
		t.Fatalf("the error does not name the path: %q", msg)
	}
	// And exactly one level of context: the doubled prefix is what this fix removed, so a
	// message that counts "weights" twice has regressed.
	if n := strings.Count(msg, "weights"); n != 1 {
		t.Fatalf("the error names weights %d times, so the caller's prefix has come back: %q", n, msg)
	}

	// The parse side is unchanged: one prefix, produced by Parse rather than by the caller.
	empty := filepath.Join(dir, "empty.json")
	if writeErr := os.WriteFile(empty, nil, 0o644); writeErr != nil {
		t.Fatal(writeErr)
	}
	_, err = Load(empty)
	if err == nil {
		t.Fatal("Load accepted an empty file")
	}
	if got := strings.Count(err.Error(), "weights"); got != 1 {
		t.Fatalf("the empty-file error names weights %d times: %q", got, err.Error())
	}
}
