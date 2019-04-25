#	常用数据结构、相应算法

##	基本类型

###	`float`

####	特殊取值

```python
infty = float("inf")
	# （正）无穷大
nan = float("nan")
	# Not a Number
```

-	特殊取值根据定义`==`、`is`肯定返回`False`
	-	`float.__eq__`内部应该有做检查，保证`==`返回`False`
	-	每次会创建“新”的`nan/infty`

	> - 连续执行`id(float("nan"))`返回值可能相等，这是因为
		每次生成的`float("nan")`对象被回收，不影响

-	`np.nan is np.nan`返回`True`，应该是`numpy`初始化的时候
	创建了一个`float("nan")`，每次都是使用同一个*nan*

##	List

##	`collections`模块

###	`deque`

-	作用：双端队列



