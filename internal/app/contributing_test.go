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
		"成对的东西要由一个来源决定",
		"状态的生命周期要定义完整",
		"「发生时对不对」和「结束后对不对」是两件事",
		"断言要从**渲染出来的文本**里读数字",
		"破坏性验证要用**非默认样本**",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("the guide no longer states %q", want)
		}
	}
}

// TestContributingDoesNotRestateTheColourRange keeps one copy of a number that has to move.
//
// The score-colour ceiling depends on the prompt, the layer and the head. README states the measured per-layer spans and
// a test pins them to the model; CONTRIBUTING carried a second, independent copy reading "本模型 5.4–8.3". Neither figure
// matched anything the app produces, and because the two live in different files, changing one never flagged the other.
// The guide now points at the README instead of repeating it.
func TestContributingDoesNotRestateTheColourRange(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "CONTRIBUTING.md"))
	if err != nil {
		t.Fatalf("read CONTRIBUTING.md: %v", err)
	}
	doc := string(b)
	if strings.Contains(doc, "5.4–8.3") {
		t.Fatal("CONTRIBUTING restates the colour-ceiling range that README owns; the two will drift apart")
	}
	// It should still describe the rule, just not with its own numbers.
	if !strings.Contains(doc, "色深") {
		t.Fatal("the colour-depth point disappeared from CONTRIBUTING")
	}
	if !strings.Contains(doc, "README") {
		t.Fatal("CONTRIBUTING no longer points at README for the measured range")
	}
}
