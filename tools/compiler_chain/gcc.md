#	GCC

##	G++

`g++`：是`gcc`的特殊版本，链接时其将自动使用C++标准库而不是
C标准库

```c
$ gcc src.cpp -l stdc++ -o a.out
	// 用`gcc`编译cpp是可行的
```

