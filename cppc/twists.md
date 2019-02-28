#	C++代码技巧

##	输入输出

###	文件输入输出

```cpp
// put code in `main`
#ifdef SUBMIT
freopen("in.txt", "r", stdin);
	// input data
freopen("out.txt", "w", stdout);
	// output
long _begin_time = clock();
#endif

// put code here

#ifdef SUBMIT
long _end_time = clock();
printf("time = %ld ms\n", _end_time - begin_time);
#endif
```
