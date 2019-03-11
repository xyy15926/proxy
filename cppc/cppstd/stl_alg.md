#	C++STL算法库

##	选择函数

###	简单多态函数

-	`max(x,y)`
-	`min(x,y)`
-	`swap(x,y)`
-	`iter_swap(x,y)`

###	迭代范围内操作

-	`binary_search(begin, end, value)`：若迭代返回内包含指定
	`value`，返回`true`
-	`copy(begin, start, out)`：将指定迭代范围内值拷贝给`out`
	开始的迭代器
-	`count(begin, end, value)`：返回迭代范围内与指定`value`
	值相等的数目
-	`fill(begin, end, value)`：将指定迭代范围内元素值置为
	`value`
-	`find(begin, end, value)`：返回指定范围内首个与`value`值
	相同的元素的迭代器，不存在则结束
-	`merge(begin_1, end_2, begin_2, end_2, out)`：将两个有序
	子序列合并为一个以`out`开始的完整有序序列
-	`inplace_merge(begin, middle, end)`：合并同一个集合内的
	两个子序列
-	`min_element(begin, end)`：返回指向迭代范围中最小元素的
	迭代器
-	`max_element(begin, end)`：返回指向迭代范围中最大元素的
	迭代器
-	`random_shuffle(begin, end)`：随机重排迭代范围中的元素
-	`replace(begin, end, old, new)`：将迭代范围中的所有`old`
	替换为`new`
-	`reverse(begin, end)`：逆序指定迭代范围中元素
-	`sort(begin, end)`：将迭代范围中元素升序排列

###	包含函数参数

> - 函数参数可以是函数对象、函数指针

-	`for_each(begin, end, fn)`：对迭代范围中每个元素调用`fn`
-	`count_if (begin, end, pred)`：计算迭代范围内`pred`返回
	`true`数目
-	`replace_if(begin, end, pred)`：将迭代范围内`pred`返回
	`true`所有值替换为`new`
-	`partition(begin, end, pred)`：将所有`pred`返回`true`
	元素放在开头，返回指向边界的迭代器



