package model

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestNoUnusedFunctionsInModel finds unexported functions in this package that
// nothing refers to.
//
// The same check runs over internal/app. Adding it here matters because the model
// package is where the arithmetic lives: a helper left behind after a rewrite looks
// like the definition of how a tensor is computed, and a reader has no way to tell
// it is no longer called.
//
// Textual rather than type-based on purpose: over-approximating references can only
// produce false negatives, never a report on a function that is actually used.
func TestNoUnusedFunctionsInModel(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	fset := token.NewFileSet()

	declared := map[string]string{}
	stripped := make([]string, 0, len(files))

	for _, name := range files {
		src, err := os.ReadFile(name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		text := string(src)
		f, err := parser.ParseFile(fset, name, src, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}
		for _, d := range f.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Name == nil || fd.Name.IsExported() {
				continue
			}
			if strings.HasPrefix(fd.Name.Name, "Test") ||
				strings.HasPrefix(fd.Name.Name, "Benchmark") ||
				strings.HasPrefix(fd.Name.Name, "Example") {
				continue // entry points, called by the testing package
			}
			declared[fd.Name.Name] = name
			off := fset.Position(fd.Name.Pos()).Offset
			if off >= 0 && off+len(fd.Name.Name) <= len(text) {
				text = text[:off] + strings.Repeat(" ", len(fd.Name.Name)) +
					text[off+len(fd.Name.Name):]
			}
		}
		stripped = append(stripped, stripGoComments(text))
	}

	joined := strings.Join(stripped, "\n")
	for name, file := range declared {
		if !hasIdent(joined, name) {
			t.Errorf("%s declares %s(), which no file in the package refers to",
				file, name)
		}
	}
}

func hasIdent(s, name string) bool {
	for i := 0; ; {
		j := strings.Index(s[i:], name)
		if j < 0 {
			return false
		}
		start := i + j
		end := start + len(name)
		if (start == 0 || !identByte(s[start-1])) && (end >= len(s) || !identByte(s[end])) {
			return true
		}
		i = start + 1
	}
}

func identByte(b byte) bool {
	return b == '_' || (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9')
}

func stripGoComments(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); i++ {
		switch {
		case i+1 < len(s) && s[i] == '/' && s[i+1] == '/':
			for i < len(s) && s[i] != '\n' {
				i++
			}
			b.WriteByte('\n')
		case i+1 < len(s) && s[i] == '/' && s[i+1] == '*':
			i += 2
			for i+1 < len(s) && !(s[i] == '*' && s[i+1] == '/') {
				i++
			}
			i++
			b.WriteByte(' ')
		default:
			b.WriteByte(s[i])
		}
	}
	return b.String()
}
