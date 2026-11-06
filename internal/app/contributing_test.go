package app

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestContributingNamesRealTests keeps the contributor guide from citing tests
// that no longer exist. The guide was written after several rounds of fixing
// "the number does not support the sentence" defects and points at the tests that
// lock each one down; a rename would leave a reader hunting for names that are
// gone.
func TestContributingNamesRealTests(t *testing.T) {
	doc, err := os.ReadFile(filepath.Join("..", "..", "CONTRIBUTING.md"))
	if err != nil {
		t.Fatalf("read CONTRIBUTING.md: %v", err)
	}
	text := string(doc)

	// Every TestXxx the guide mentions must exist in this package or in model.
	sources, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}
	modelSources, _ := filepath.Glob("../model/*.go")
	sources = append(sources, modelSources...)
	var all strings.Builder
	for _, f := range sources {
		b, err := os.ReadFile(f)
		if err != nil {
			continue
		}
		all.Write(b)
	}
	declared := all.String()

	// Every TestXxx the guide names, without guessing at suffixes: the point is
	// to catch renames, and a pattern that only matches certain endings would
	// silently skip most of them.
	re := regexp.MustCompile(`Test[A-Z][A-Za-z0-9]*`)
	named := map[string]bool{}
	for _, m := range re.FindAllString(text, -1) {
		named[m] = true
	}
	if len(named) == 0 {
		t.Fatal("the guide names no tests; this test is no longer checking anything")
	}
	for name := range named {
		if !strings.Contains(declared, "func "+name+"(") {
			t.Fatalf("CONTRIBUTING.md cites %s, which no longer exists", name)
		}
	}
}

// TestContributingCoversTheTeachingChecks checks the guide still carries the
// list of figure-reading mistakes, since that list is the part of the document
// that a test cannot enforce on its own.
func TestContributingCoversTheTeachingChecks(t *testing.T) {
	doc, err := os.ReadFile(filepath.Join("..", "..", "CONTRIBUTING.md"))
	if err != nil {
		t.Fatalf("read CONTRIBUTING.md: %v", err)
	}
	text := string(doc)
	for _, want := range []string{
		"同一组图必须用同一个尺度",
		"颜色深浅要说清上限",
		"图形要标出参照点",
		"数字要支撑它旁边那句话",
		"孤立数字要有量纲参照",
		"切片要说清总体",
		"文档里的数字必须从代码派生",
		"断言要从**渲染出来的文本**里读数字",
		"破坏性验证要用**非默认样本**",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("the guide no longer states %q", want)
		}
	}
}
