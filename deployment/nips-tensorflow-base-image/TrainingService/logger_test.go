package main

import (
	"strings"
	"testing"
)

func TestLogWithDate(t *testing.T){
	var out string

	out = logWithDate("")
	if !strings.Contains(out, "Can not print empty string!") {
		t.Fail()
	}

	out = logWithDate("This is an output!")
	if !strings.Contains(out, "This is an output!") {
		t.Fail()
	}

}

func TestLogWithDateAndFormat(t *testing.T){
	var out string

	out = logWithDateAndFormat("")
	if !strings.Contains(out, "Can not print empty string!") {
		t.Fail()
	}

	out = logWithDateAndFormat("empty interface")
	if !strings.Contains(out, "interface is nil") {
		t.Fail()
	}

	out = logWithDateAndFormat("%s %s %s %s", "This","is","an","output!")
	if !strings.Contains(out, "This is an output!") {
	t.Fail()
	}
}