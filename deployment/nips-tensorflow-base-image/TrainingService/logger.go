package main

import (
	"fmt"
	"time"
)

func logWithDateAndFormat(format string, a ...interface{}) (out string) {

	if len(format) > 0 {
		if a == nil {
			out = logWithDate("interface is nil")
		} else {
			out = fmt.Sprintf(format, a...)
			out = fmt.Sprintf("[%s] %s", time.Now().Format("15:04:05"), out)
			fmt.Println(out)
		}
	} else {
		out = logWithDate("Can not print empty string!")
	}

	return
}

func logWithDate(output string) (out string) {

	if len(output) > 0 {
		out = fmt.Sprintf("[%s] %s", time.Now().Format("15:04:05"), output)
		fmt.Println(out)
	} else {
		out = logWithDate("Can not print empty string!")
	}

	return
}
