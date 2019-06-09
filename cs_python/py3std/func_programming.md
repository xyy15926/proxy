#	Funtioncal Programming Tools

##	`builtins`

-	`filter(iterable, func)`：过滤`func`返回布尔否值元素
-	`enumerate(iterable)`：添加索引迭代
-	`zip(*iterables)`：打包迭代元素
-	`map(func, *iterables)`：`func`接受各迭代器中元素作为
	参数调用（类似`zip`）
-	`iter`：两种产生迭代器的模式
	-	`iter(iterable)`：返回迭代器
	-	`iter(callable, sentinel)`：调用`callable`直到返回
		`sentinel`停止迭代（不包括）
		-	可用于替代`while`循环

##	`itertools`

`itertools`：包含为高效循环而创建迭代器的函数

> - 参考<https://docs.python.org/zh-cn/3/library/itertools.html>

###	无穷迭代器

-	`count(start, step=1)`：步长累加
-	`cycle(p)`：循环`p`中元素
-	`repeat(elem, n=None)`：重复`elem`

###	迭代元素处理

-	`accumulate(p, func=None)`：累加`p`中元素
-	`chain(*iters)`：链接迭代器
-	`chain.from_iterable(iterable)`：链接可迭代对象中迭代器
-	`compress(data, selelctors)`：根据`selectors`选取`data`
-	`dropwhile(pred, seq)`：保留首个满足`pred`之后所有元素
-	`takewhile(pred, seq)`：保留首个不满足`pred`之前元素
-	`filterfalse(pred, seq)`：反`filter`
-	`groupby(iterable, key=None)`：根据`key(v)`值分组迭代器
-	`islice(seq, start=0, stop=None, step=1)`：切片
-	`starmap(func, seq)`：迭代对`seq`中执行`func(*elem)`
-	`tee(iterable, n=2)`：复制迭代器，默认2个
-	`zip_longes(*iters, fillvalue)`：依最长迭代器`zip`，较短
	循环填充或`fillvalue`填充

###	排列组合

-	`product(*iters, repeat=1)`：笛卡尔积
-	`permutations(p, r=None)`：`r`长度的排列，缺省全排列
-	`combinations(p, r)`：`r`长度组合
-	`combinations_with_replacement(p,r)`：可重复组合

##	`functools`

`functools`：包含高阶函数，即参数、返回值为其他函数的函数

> - 参见<https://docs.python.org/zh-cn/3/library/functools.html>

###	函数转换

-	`cmp_to_key(func)`：将旧式比较函数转换新式*key function*

	-	常用在以下函数`key`参数中转换旧式比较函数
		-	`sorted`
		-	`min`
		-	`max`
		-	`heapq.nlargest`
		-	`heapq.nsmallest`
		-	`itertools.groupby`

	> - *key function*：接收参数，返回可以用于排序的值

-	`partial(func, *args, **kwargs)`：返回partial对象，其
	调用时类似使用`args`、`kwargs`调用`func`

-	`partial.method(func, *args, **kwargs)`：适合类命名空间
	中函数、描述器

-	`update_wrapper(wrapper, wrapped, assigned=WRAPPER_ASSIGNMENTS, udpated=WRAPPER_UPDATES)`：
	更新`wrapper`函数信息为`wrapped`函数信息

###	函数应用

-	`reduce(func, iterable[, initializer])`：使用`func`接受
	两个参数，reduce处理`iterable`

###	函数装饰器

-	`@lru_cache(maxsize=128, typed=False)`：缓存函数执行结果

	-	字典存储缓存：函数固定参数、关键字参数必须可哈希
	-	不同参数调用模式（如仅参数顺序不同）可能被视为不同
		，从而产生多个缓存项

	> - 可以用于方便实现动态规划

-	`@singledispatch`：转换函数为单分派范型函数，实现python
	的重载（接口多态）

	> - *single dispatch*：单分派，基于单个参数类型分派的
		范型函数分派形式

-	`@wraps(wrapped, assigned=WRAPPER_ASSIGNMENTS, updated=WRAPPER_UPDATES)`：
	等价于`partial(update_wrapper, wrapped, assigned, updated)`

###	类装饰器

-	`@total_ordering`：根据类中已定义比较方法实现剩余方法
	-	类必须实现`__eq__`、其他富比较方法中任意一种

##	`operator`

`operator`：提供与python内置运算符对应的高效率函数

-	为向后兼容，保留双下划线版本函数名（同特殊方法名）

###	比较运算

-	`lt(a,b)`/`__lt__(a,b)`
-	`le(a,b)`
-	`eq(a,b)`
-	`ne(a,b)`
-	`ge(a,b)`
-	`gt(a,b)`

###	逻辑运算

-	`not(obj)`
-	`truth(obj)`：等价于使用bool构造器
-	`is_(a,b)`
-	`is_not(a,b)`

###	数值运算 

-	`add(a,b)`
-	`sub(a,b)`
-	`mul(a,b)`
-	`div(a,b)`
-	`pow(a,b)`
-	`mod(a,b)`
-	`floordiv(a,b)`
-	`truediv(a,b)`
-	`matmul(a,b)`
-	`abs(obj)`
-	`neg(obj)`
-	`pos(obj)`

####	在位赋值

> - 在位运算对可变数据类型才会更新参数，否则只返回结果

-	`iadd(a,b)`：等价于`a += b`
-	`isub(a,b)`
-	`imul(a,b)`
-	`idiv(a,b)`
-	`ipow(a,b)`
-	`imod(a,b)`
-	`ifloordiv(a,b)`
-	`itruediv(a,b)`
-	`imatmul(a,b)`

###	位运算

-	`and_(a,b)`
-	`or_(a,b)`
-	`xor(a,b)`
-	`inv(obj)`/`invert(obj)`
-	`lshift(a,b)`
-	`shift(a,b)`

####	在位运算

-	`iand(a,b)`
-	`ior(a,b)`
-	`ixor(a,b)`

###	索引

-	`index(a)`

###	序列运算

-	`concat(a,b)`
-	`contains(a,b)`
-	`countOf(a,b)`
-	`delitem(a,b)`
-	`getitem(a,b)`
-	`indexOf(a,b)`
-	`setitem(a,b,c)`
-	`length_hint(obj, default=0)`

###	访问函数

-	`attrgetter(attr)`/`attrgetter(*attrs)`：返回函数，函数
	返回值为参数属性`attr`
-	`itemgetter(item)`/`itemgetter(*items)`：类似
-	`methodcaller(name[, args...])`：类似



