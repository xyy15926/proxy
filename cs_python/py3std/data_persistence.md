---
title: 数据持久化
tags:
  - Python
categories:
  - Python
date: 2019-06-10 01:11:37
updated: 2019-06-10 01:11:37
toc: true
mathjax: true
comments: true
description: 数据持久化
---

##	DBM

DBM文件：python库中数据库管理的标准工具之一

-	实现了数据的随机访问
	-	可以使用键访问存储的文本字符串
-	DBM文件有多个实现
	-	python标准库中`dbm/dbm.py`

###	使用

-	使用DBM文件和使用内存字典类型非常相似
	-	支持通过键抓取、测试、删除对象

##	`pickle`

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

-	不支持`pickle`序列化的数据类型
	-	套接字

##	`shelves`

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
##	`copyreg`

##	`marshal`

##	`sqlite3`


