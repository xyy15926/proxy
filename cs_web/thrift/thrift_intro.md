---
title: Thrift简介
categories:
  - Web
  - Thrift
tags:
  - Web
  - Web Server
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:08
toc: true
mathjax: true
comments: true
description: Thrift简介
---

##	Thrift架构

-	Thrift是跨语言、C/S模式、服务器部署框架

-	使用Interface Definition Language（IDL）定义RPC接口
	、数据类型

-	然后通过Thrift编译器生成不同语言的**代码**，并由生成
	代码负责RPC**协议层**、**传输层**实现

	-	支持服务器端和客户端编译生成代码为不同语言
	-	客户端、服务端代码调用生成代码搭建C/S

-	Thrift支持动态（执行）、静态（编译）

###	Thrift网络栈

![thrift_structure](imgs/thrift_structure)

####	TTransport

传输层，定义数据传输方式

-	可以是TPC/IP、共享内存、共享文件等，作为运行时库

-	提供了一个简单的网络读写抽象层，使Thrift底层TTransport
	从系统其他部分（如：序列化、反序列化）解耦

-	接口方法包括
	-	`open`
	-	`close`
	-	`read`
	-	`write`
	-	`listen`
	-	`accept`
	-	`flush`

#####	传输协议

-	TSocket：阻塞式socket
-	TFramedTransport：frame为单位进行传输，非阻塞式
-	TFileTransport：文件形式传输
-	TMemoryTransport：直接对内存进行I/O
	-	Java实现时使用了简单的ByteArrayOutputStream
-	TZlibTransport：使用zlib进行压缩，同其他方式联合使用
	-	目前无java实现

####	TProtocol

协议层，定义数据传输格式

-	定义了一种将内存的数据结构映射成可传输格式的机制，即定义
	数据类型在TTransport和自身间进行解、编码

-	需要实现编码机制，负责对数据进行序列化、反序列化

#####	数据格式

-	TBinaryProtocal：二进制格式
-	TCompactProtocal：压缩格式
-	TJSONProtocol：JSON格式
-	TSimpleJSONProtocal：提供JSON只写协议，生成文件容易
	通过脚本语言解析
-	TDebugProtocol：简单易懂的可读文本格式，便于debug

####	TProcessor

封装了从输入数据流中读取、向输出数据流中写的操作

-	读写数据流用TProtocol对象表示
-	和服务相关的TProcessor由Thrift编译器产生
-	工作流程
	-	使用输入TProtocol从连接中读取数据
	-	将处理授权给用户实现的handler
	-	使用输出TProtocol向连接中写入数据


###	服务模型

-	创建TTransport对象
-	为TTransport对象创建输入、输出TProtocol对象
-	基于输入、输出TProtocol对象创建TProcessor对象
-	等待连接请求，交由TProcessor处理

####	支持的服务模型

-	TSimpleServer：简单单线程服务模型
-	TThreadPoolServer：多线程服务模型，使用标准阻塞式I/O
-	TNonblockingServer：多线程服务模型，使用非阻塞式I/O
	-	需使用TFrameTransport传输方式

##	Thrift语法

###	句法

-	支持shell的`#`注释、C/C++的`//`/`/**/`注释

-	struct等复杂类型定义中
	-	类型名和`{`之间必须有空格
	-	各字段之间`,`分割（末尾不能有`,`）
	-	方法可以使用`;`、`,`结尾

###	函数

####	参数

-	可以是基本类型、结构体
-	参数是常量`const`，不能作为返回值

####	返回值

-	可以是基本类型、结构体

##	数据类型

###	基本类型

不支持无符号整形

-	`bool`
-	`byte`
-	`i16`
-	`i32`
-	`i64`
-	`double`
-	`string`
-	`binary`：字节数组

###	泛型（容器）

容器中元素类型可以是除了service以外的任何类型（包括结构体、
异常）

-	`map<t1, t2>`：字典
-	`list<t1>`：列表
-	`set<t1>`：集合

###	结构体

Thrift结构体概念上同C结构体：将相关数据封装

```thrift
struct Work {
	1: i32 num1=0,
	2: i32 num2,
	3: Operation op,
	4: optional string comment,
}
```

-	编译为面向对象语言时：将被转换为类

-	结构体中，每个字段包含
	-	整数ID
	-	数据类型
	-	字段名
	-	可选的默认值

-	字段可选：**规范**的struct定义中，每个域都使用`optional`
	、`required`关键字标识
	-	`optional`：字段未设置时，序列化输出时不被包括
	-	`required`：字段未设置时，Thrift给与提示

-	不支持继承

###	Exception

```thrift
exception InvalidOperation {
	1: i32 what,
	2: string why
}
```

-	异常在语法、功能上类似于结构体，使用`exception`声明，但
	语义不同

###	Service

Thrift编译器根据选择的目标语言为server产生服务接口代码，为
client产生桩代码

-	函数、参数列表定义方式同struct
-	支持继承：`extends`

```thrift
service Twitter {
	# Twitter和`{`中需要有空格
	void ping(),
	bool postTweet(1: Tweet tweet);
	TweetSearchResult searchTweets(1: string query);
	oneway void zip();
```

###	enum

枚举类型

-	枚举常量必须时32位正整数

```thrift
enum TweetType {
	TWEET,
	RETWEET = 2,
	DM = 0xa,
	REPLY
}
```

###	const

常量

-	复杂类型、结构体可以使用JSON标识

```thrift
const i32 INT_CONST = 1234
const map<string, string> MAP_CONST = {"hello": "world", "1": "2"}
```

###	typedef

```md
typedef i32 new_type
```

##	namespace

Thrift命名空间同C++中namespace类似

-	均提供组织（隔离）代码的方式
-	因为不同的语言有不同的命名空间定义方式（如：python中
	module），Thrift允许针对特定语言定义namespace

```thrift
namespace cpp com.example.project
namespace java com.example.project
```



