#	Python部分内置模块

##	`atexit`

`atexit`模块主要用于在**程序结束前**执行代码

-	类似于析构，主要做资源清理工作

###	`atexit.register`

```python
def register(
	func,
	*arg,
	**kwargs
)
	# 注册回调函数
```

-	如果程序非正常crash、通过`os._exit()`退出，注册的回调
	函数不会被调用

-	调用顺序按照注册顺序反向

```python
import atexit

df func1():
	print("atexit func 1, last out")

def func2(name, age):
	print("atexit func 2")

atexit.register(func1)
atexit.register(func2, "john", 20)

@atexit.register
def func3():
	print("atexit func 3, first out")
```

####	实现

`atexit`内部时通过`sys.exitfunc`实现的

-	将注册函数放到列表中
-	当程序退出时按照**先进后出**方式调用注册的回调函数，
-	若回调函数执行过程中抛出异常，`atexit`捕获异常然后继续
	执行之后回调函数，知道所有回调函数执行完毕再抛出异常

> - 二者同时使用，通过`atexit.register`注册回调函数可能不会
	被正常调用



