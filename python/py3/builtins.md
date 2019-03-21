#	Python部分内置模块

##	`pickle`

`pickle`

-	将内存中的python对象转换为序列化的字节流，可以写入任何
	输出流中
-	根据序列化的字节流重新构建原来内存中的对象
-	感觉上比较像XML的表示方式，但是是python专用

```python
import pickle
dbfile = open("people.pkl", "wb")
pickle.dump(db, dbfile)
dbfile.close()
dbfile = open("people.pkl", "rb")
db = pickle.load(dbfile)
dbfile.close()
```

##	`shelves`

`shelves`

-	就像能必须打开着的、存储持久化对象的词典
	-	自动处理内容、文件之间的映射
	-	在程序退出时进行持久化，自动分隔存储记录，只获取、
		更新被访问、修改的记录
-	使用像一堆只存储一条记录的pickle文件
	-	会自动在当前目录下创建许多文件

```python
import shelves
db = shelves.open("people-shelves", writeback=True)
	// `writeback`：载入所有数据进内存缓存，关闭时再写回，
		// 能避免手动写回，但会消耗内存，关闭变慢
db["bob"] = "Bob"
db.close()
```

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



