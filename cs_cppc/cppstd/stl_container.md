---
title: 集合类
tags:
  - C/C++
categories:
  - C/C++
date: 2019-03-21 17:27:37
updated: 2019-03-08 21:36:46
toc: true
mathjax: true
comments: true
description: 集合类
---

##	*Vector*

Vector类提供了类似数组功能的机制，并对C++中数组做出了改进

> - 数组参见*cppc/mem_ctl*

###	API

```cpp
Vector<int> vec;
	// 创建Vector对象
Vector<int> vec(VEC_LEN);
	// 创建确定长度Vector对象长度，并将值初始化为0

vec.add(10);
	// 在Vector对象里面添加新元素
vec.insert(2, 30);
	// 向Vector对象中index=2插入元素
vec.remove(0)
	// 删除Vector对象中index=0处元素
vec.set(3, 70)
vec[3] = 70
	// 改变Vector对象中index=3处元素
	// `[]`方法简短、和数组操作更相似，Vector被设计模仿
```

####	构造函数

-	`Vector<type>()`
-	`Vector<type>(n, value)`：创建含有n个对象Vector元素，
	每个元素都被初始化为`value`，缺省为相应类型默认值

####	方法

-	`.size()`
-	`.isEmpty()`
-	`.get(index)`
-	`.set(index, valu)`
-	`.add(value)`
-	`.insertAt(index, value)`
-	`.removeAt(index)`
-	`.clear()`

####	操作符

-	`[index]`
-	`v1 + v2`：连接两个Vector，返回包含所有元素的Vector
-	`v1 += e1`：向Vector尾部添加元素

###	注意

####	参数传递Vector对象

```cpp
void print_vector(Vector<int> & vec){
	cout << "[";
	for (int i=0; i<vec.size(); i++){
		if (i>0)
			count << ",";
		count << vec[i];
	}
	count << "]" << endl;
}
```

####	二维结构

```cpp
Vector< Vector<int> > sodok(9, Vector<int>(9))
	// 内部`Vector<int>`两侧需要空格，否则`>>`被错误解析
```

##	*Stack*

###	API

####	构造函数

-	`Stack<type>()`

####	方法

-	`.size()`
-	`.isEmpty()`
-	`.push(value)`
-	`.pop()`
-	`.peek()`：返回栈顶元素但不出栈
-	`.clear()`

##	*Queue*

###	API

####	构造函数

-	`Queue<type>()`

####	方法

-	`.size()`
-	`.isEmpty()`
-	`.enqueue(value)`：将值添加到队尾
-	`.dequeue()`：删除队首元素，并返回给调用者
-	`.peek()`：返回队首元素但不将其熟队列中删除
-	`.clear()`

##	*Map*

###	API

####	构造函数

-	`Map<key_type, value_type>()`

####	方法

-	`.size()`
-	`.isEmpty()`
-	`.put(key, value)`
-	`.get(key)`：返回Map对象中当前与键key相关联的值
	-	若该键没有定义，`get`创建新的键值对，设置值为默认值
-	`.remove(key)`
-	`.containsKey(key)`
-	`.clear()`

####	操作符

-	`map[key]`：同`get`方法

##	*Set*

###	API

####	构造函数

-	`Set<type>()`

####	方法

-	`.size()`
-	`.isEmpty()`
-	`.add(value)`
-	`.remove(value)`
-	`.contains(value)`
-	`.clear()`
-	`.isSubsetof(set)`
-	`.first()`

####	操作符

-	`s1 + s2`：返回两集合并运算结果
-	`s1 * s2`：交
-	`s1 - s2`：差
-	`s1 += s2`
-	`s1 -= s2`
-	`s1 *= s2`


