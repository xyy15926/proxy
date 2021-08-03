---
title: C++函数式编程
categories:
  - C/C++
  - STL
tags:
  - C/C++
  - STL
  - Lambda
date: 2019-03-21 17:27:37
updated: 2019-03-11 12:44:09
toc: true
mathjax: true
comments: true
description: C++函数式编程
---

##	`<functional>`

###	基类

-	`binary_function<arg_type1, arg_type2, result_type>`：
	两个指定类型参数、返回指定类型的函数类的父类
-	`unary_function<arg_type, result_type>`：指定类型参数、
	返回指定类型的函数类的父类

###	实现算数操作符类

-	`plus<arg_type>`：`+`
-	`minus<arg_type>`：`-`
-	`multiples<arg_type>`：`*`
-	`divides<arg_type>`：`/`
-	`modulus<arg_type>`：`%`
-	`negate<arg_type>`：`-`取反

###	实现比较操作

-	`equal_to<arg_type>`：`==`
-	`not_equal_to<arg_type>`：`!=`
-	`less<arg_type>`：`<`
-	`less_equal<arg_type>`：`<=`
-	`greater<arg_type>`：`>`
-	`greater_equal<arg_type>`：`>=`

###	实现逻辑关系

-	`logical_and<arg_type>`：`&&`
-	`logical_or<arg_type>`：`||`
-	`logical_not<arg_type>`：`!`

###	产生函数对象

-	`bind1st(fn, value)`：返回新一元函数对象，用与其绑定的
	`value`作为首个参数调用二元函数对象`fn`
-	`bind2nd(fn, value)`：返回新一元函数对象，用与其绑定的
	`value`作为第二个参数调用二元函数对象`fn`
-	`not1(fn)`：返回新函数对象，且该函数对象为一元函数对象
	时返回`true`
-	`not2(fn)`：返回新函数对象，且该函数对象为二元函数对象
	时返回`true`
-	`ptr_fun(fnptr)`：返回需要调用特定函数指针的新函数对象，
	可能需要一个或两个同类型参数
	-	返回具有**相同效果**的函数对象，可以使得函数指针、
		函数对象概念一体化，避免代码重复

###	例

```cpp
count_if(v.begin(), v.end(), bind2nd(less<int>(), 0));
	// 返回元素类型为整形的矢量对象`v`中负数数量
template<typename FunctionClass>
void func_func(FunctionClass fn);
void func_func(double (*fn)(double)){
	// 函数重载+`ptr_fun`减少代码重复
	func_func(ptr_func(fn));
}
```





