package app

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestNoUnusedFunctionsInApp finds unexported functions and methods in this
// package that no other file refers to.
//
// The package had accumulated four: heatmap (a wrapper around heatmapAxed fixed at
// axes=false that no call site used, since both real callers pass true), plus
// renderFFN, renderFFNPositive and drawHeatmap in the tests — helpers left behind
// when the tests around them were rewritten inline. Unused code is not just
// clutter: it is where a reader asking "how does this widget work" lands first,
// and it was describing a mode the app never renders.
//
// Deliberately textual: an identifier occurring anywhere outside its own
// declaration counts as a reference, which over-approximates and therefore never
// reports a function that is actually used.
func TestNoUnusedFunctionsInApp(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	// name -> file where it is declared, for unexported funcs and methods.
	declared := map[string]string{}
	// Source of every file, with each declaration's own name blanked out, so a
	// name that appears only at its definition is not counted as a use.
	stripped := make([]string, 0, len(files))

	for _, name := range files {
		src, err := readFileString(name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		// Strip comments from the pristine source, before any name is blanked. Blanking a
		// declaration's name makes the file unparseable — `func (a *App) foo(` becomes
		// `func (a *App)    (` — so the parser has to see the original, and the offsets it
		// reports are the ones used for both edits.
		fileFset := token.NewFileSet()
		f, err := parser.ParseFile(fileFset, name, src, parser.ParseComments)
		if err != nil {
			t.Fatalf("parse %s: %v", name, err)
		}

		// Blank the comments, keeping newlines so a later failure's line numbers still line
		// up. A space in place of every other comment byte keeps the identifiers on either
		// side from fusing.
		blanked := []byte(src)
		for _, group := range f.Comments {
			for _, c := range group.List {
				off := fileFset.Position(c.Pos()).Offset
				endOff := fileFset.Position(c.End()).Offset
				for i := off; i < endOff && i < len(blanked); i++ {
					if blanked[i] != '\n' {
						blanked[i] = ' '
					}
				}
			}
		}

		for _, d := range f.Decls {
			fd, ok := d.(*ast.FuncDecl)
			if !ok || fd.Name == nil || fd.Name.IsExported() {
				continue
			}
			if strings.HasSuffix(name, "_test.go") {
				// Test, Benchmark and Example functions are entry points, not
				// helpers, and are called by the testing package.
				if strings.HasPrefix(fd.Name.Name, "Test") ||
					strings.HasPrefix(fd.Name.Name, "Benchmark") ||
					strings.HasPrefix(fd.Name.Name, "Example") {
					continue
				}
			}
			declared[fd.Name.Name] = name
			// Blank the declaration's own name, so a name that appears only at its
			// definition is not counted as a use. Same length, so no offset moves.
			off := fileFset.Position(fd.Name.Pos()).Offset
			for i := off; i < off+len(fd.Name.Name) && i < len(blanked); i++ {
				blanked[i] = ' '
			}
		}
		stripped = append(stripped, string(blanked))
	}

	joined := strings.Join(stripped, "\n")
	for name, file := range declared {
		// Word-boundary search: the name followed by a character that cannot
		// continue an identifier.
		if !containsIdentifier(joined, name) {
			t.Errorf("%s declares %s(), which no file in the package refers to",
				file, name)
		}
	}
}

// containsIdentifier reports whether s contains name as a whole identifier.
func containsIdentifier(s, name string) bool {
	for i := 0; ; {
		j := strings.Index(s[i:], name)
		if j < 0 {
			return false
		}
		start := i + j
		end := start + len(name)
		beforeOK := start == 0 || !isIdentChar(s[start-1])
		afterOK := end >= len(s) || !isIdentChar(s[end])
		if beforeOK && afterOK {
			return true
		}
		i = start + 1
	}
}

func isIdentChar(b byte) bool {
	return b == '_' || (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || (b >= '0' && b <= '9')
}

// readFileString reads a file in this directory as a string.
func readFileString(name string) (string, error) {
	b, err := os.ReadFile(name)
	if err != nil {
		return "", err
	}
	return string(b), nil
}
