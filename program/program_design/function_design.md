#	函数设计

##	Hook/Callback

-	callback：回调函数，实现与调用者分离
	-	在API中注册、用于在处理流中的合适时刻调用的函数
	-	可以看作是hook的一种

-	hook：钩子函数
	-	更广义的、用于修改对API调用的函数
	-	callback、hook意义其实差不多，很多情况下名词混用

-	register：注册函数
	-	提供参数作为hook的**挂载**点
	-	register通过挂载点调用hook

###	优点

-	让用户实现hook然后注册，框架调用用户自定义的hook，提供
	更好的泛用性

-	框架不再处理hook函数中涉及的问题，降低系统耦合程度

-	hook函数可以随时更换，提升框架的灵活性、扩展性

###	实现

**函数指针**作为函数参数，API通过**函数指针**调用函数实现
**定制**API

####	`C`实现

```c
 # include<stdlib.h>
 # include<stdio.h>

void populate_array(
	int *array, 
	size_t array_size,
	int (*get_next_value_hook)(void)
	// `get_next_value_hook`就是函数指针
	// hook的“挂载点”
){
	for (size_t i=0; i<array_size; i++){
		array[i] = get_next_value();
	}
}

int get_next_random_value(){
	return rand();
}

int main(){
	int array[10];
	poppulate_array(array, 10, get_next_random_value);
		// 这里`get_next_random_value`就是钩子函数
	for(int i=0; i<10; i++){
		printf("%d\n", array[i]);
	}
	printf("\n");
	return 0;
```

####	Python实现

python中虽然没有明确的C中的函数指针，但是实现原理仍然类似，
都是**“函数指针”**作为参数，由API负责调用实现**定制**API

```python
import random

populate_array(l, len, func):
	# `func`就类似于`C`中的函数指针，hook的挂载点
	for i in range(len):
		l[l] = func()

get_next_random_value():
	return random.random()

def test():
	l = [ ]
	populate_array(l, 10, get_next_random_value)
		# `get_next_random_value`就是钩子函数
	print(l)
```

###	*Closure*

闭包：一个函数及与其相关的组合

-	将回调函数与数据封装成单独的单元
-	实现方式
	-	允许在函数定义的同时，使用函数内部变量支持闭包，如：
		python
	-	使用特定数据结构实现闭包，如：C++使用**函数类**实现


