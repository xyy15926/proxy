---
title: IO
categories:
  - C/C++
  - STL
tags:
  - C/C++
  - STL
date: 2019-05-10 01:01:26
updated: 2019-05-10 01:01:26
toc: true
mathjax: true
comments: true
description: IO
---

-	C++中数据输入/输出操作是通过I/O流库实现

-	流：数据之间的传输操作
	-	输出流：数据从内存中传送到某个载体、设备中
	-	输入流：数据从某个载体、设备传送到内存缓冲区

-	C++中流类型
	-	标准流I/O流：内存与标准输入、输出设备之间信息传递
	-	文件I/O流：内存与外部文件之间信息传递
	-	字符串I/O流：**内存变量**与表示字符串流的字符数组
		之间信息传递

##	`<ios>`

###	`class ios`

`ios`：流基类

-	所有流的父类
-	保存流状态、处理错误

####	方法

-	`.fail()`：判断流是否失效
	-	尝试超出文件的结尾读取数据时
	-	输入流中字符串无法被正确解析

-	`.eof()`：判断流是否处于文件末尾
	-	基于C++流库语义，`.eof`方法只用在`.fail`调用之后，
		用于判断错故障是否是由于到达文件结尾引起的

-	`.clear()`：重置与流相关状态位
	-	故障发生后，任何时候重新使用新流都必须调用此函数

-	`if(stream)`：判断流是否有效
	-	大部分情况下等同于`if(!stream.fial())`

-	`.open(filename)`：尝试打开文件filename并附加到流中
	-	流方向由流类型决定：输入流对于输入打开、输出流对于
		输出打开
	-	可以调用`.fail`判断方法是否失败

-	`.close()`：关闭依附于流的文件

#####	`.[un]setf`

```cpp
UKNOWN setf(setflag, unsetfield);
UKNOWN unsetf(unsetflag);
```

-	用途
	-	`.setf`：设置某个流操纵符
	-	`.unsetf()`：取消某个流操纵符

-	参数
	-	`setflag`：需要设置的操纵符
	-	`unsetflag`：取消设置的操纵符
	-	`unsetfield`：需要清空的格式设置位组合

> - 不能像`<<`、`>>`中省略操纵符`ios::`前缀

#####	`.rdbuf`

```cpp
template <class Elem, class Traits>
class basic_ios: public ios_base{
	basic_streambuf <_Elem, _Traits> *_Mystrbuf,

	_Mysb * rdbuf() const{
		return (_Mystrbuf);
	}

	_Mysb * rdbuf(_Mysb * _Strbuf){
		_Mysb * _Oldstrbuf = _Mystrbuf;
		_Mystrbuf = _Strbuf;
		return (_Oldstrbuf);
	}
}
```

-	用途：获得输入、输出流对象中指向缓冲区类`streambuf`指针
	-	`>>`、`<<`操作符对其有重载，可以方便读取、写入

###	`class istream`

`istream`：输入流基类

-	将流缓冲区中数据作格式化、非格式化之间的转换
-	输入

####	方法

-	`.unget()`：复制流的内部指针，以便最后读取的字符能再次
	被下个`get`函数读取

#####	`.get`

```cpp
int_type get();
basic_istream& get(E& c);
basic_istream& get(E *s, streamsize n);
basic_istream& get(E *s, streamsize n, E delim);
basic_istream& get(basic_stream<E, T> &sb);
basci_istream& get(basci_stream<E, T> &sb, E delim);
```

-	用途：从输入流中获取字符、字符串

-	参数
	-	`delim`：分隔符，缺省`\n`
	-	`n`：

####	（友元）函数

#####	`getline`

```cpp
template<class E, class T, class A>
basic_istream<E, T>& getline(
	basic_istream<E, T>& is,
	basic_string<E, T, A>& str,
);

template<class E, class T, class A>
basic_istream<E, T>& getline(
	basic_istream<E, T>& is,
	basic_string<E, T, A>& str,
	E delim,
);
```

-	用途：从流`is`读取以`delim`为界，到字符串中
	-	保留开头空白字符、丢弃行尾分割符
	-	读取字符直到分隔符，若首字符为分隔符则返回空字符串

-	参数
	-	`delim`：分隔符，缺省为换行符`\n`

###	`class ostream`

`ostream`：输出流基类

-	将流缓冲区中数据作格式化、非格式化之间的转换，输出

####	方法

-	`.put(ch)`：将字符`ch`写入输出流

###	`class iostream`

`iosstream`：多目的输入、输出流基类

###	*Operator*

####	*Insertion Operator*

`<<`：插入操作符，将数据插入流中

-	左操作数是输出流

-	右操作数是需要插入流中的数据

	-	基本类型：`<<`会将其自动转换为字符串形式
		-	整形：默认10进制格式
		-	`[unsigned ]char`类型：总是插入单个字符

	-	`streambuf`类型指针：插入缓冲区对象中所有字符

####	*Extraction Operator*

`>>`：提取操作符，从输入流中读取**格式化数据**

-	左操作数为输入流

-	右操作数存储从输入流中读取的数据

	-	缺省
		-	`skipws`：忽略开头所有空白字符
		-	空白字符分隔：读取字符直到遇到空白字符

	-	`streambuf`类型指针：把输入流对象中所有字符写入该
		缓冲区

> - 几乎不提供任何支持检测用户输入是否有效的功能
> > -	数据格式由**变量类型控制**

###	缓冲

####	缓冲类型

> - ISO C要求
> > -	当且仅当不涉及交互设备时，标准输入、输出全缓存
> > -	标准错误绝不是全缓存

-	无缓冲：不缓冲字符

	-	适用情况：标准错误

	-	标准库不缓冲不意味着系统、设备驱动不缓冲

-	行缓冲：在输入、输出遇到换行符时才会执行I/O操作

	-	适用情况：涉及交互设备，如标准输入、输出

-	全缓冲：I/O操作只会在缓冲区填满后才会进行

	-	适用情况：大部分情况，如驻留在磁盘的文件

	-	*flush*描述I/O缓冲写操作
		-	标准I/O函数自动*flush*
		-	手动调用对流调用死`fflush`函数

> - 缓冲区一般是在第一次对流进行I/O操作时，由标准I/O函数调用
	`malloc`函数分配得到

####	文件自定义缓冲区

-	文件必须已打开、未做任何操作

#####	`setbuf`

```c
void setbuf(FILE * restrict fp, char * restrict buf);
```

-	用途：打开或关闭缓冲区
	-	打开：`buf`必须为大小为`BUFSIZ`的缓存
		-	`BUFSIZ`：定义在`stdio.h`中，至少256
	-	关闭：将`buf`设置为`NULL`

#####	`setvbuf`

```c
int setvbuf(FILE * restrict fp, char * restrict buf,
	int mode, size_t size);
```

-	用途：设置缓冲区类型

####	流自定义缓冲区

#####	`setbuf`

```cpp
virtual basic_streambuf * setbuf(E *s, streamsize n);
```

##	*Manipulator*

（流）操纵符：控制格式化输出的一种特定类型值

###	输出

> - 短暂的：只影响下个插入流中的数据
> - 持久的：直到被明确改变为止

> - 双操纵符条目中，前者为默认
> - `setw`、`setprecision`、`setfill`还需要包含`<iomanip>`

####	组合格式

-	`adjustfield`：对齐格式位组合
-	`basefield`：进制位组合
-	`floatfield`：浮点表示方式位组合

####	位置

-	`endl`：将行结束序列插入输出流，确保输出字符被写入目的流
-	`setw(n)`：短暂的
-	`setfill(ch)`：持久的，指定填充字符，缺省空格
-	`left`：持久的，指定有效值靠左
-	`right`：持久的，指定有效值靠右
-	`internal`：持久的，指定填充字符位于符号、数值间

####	数值

-	`showbase`：为整数添加表示其进制的前缀
-	`fixed`：持久的，完整输出浮点数
-	`scientific`：持久的，科学计数法输出浮点数
-	`setprecision(digits)`：持久的，精度设置依赖于其他设置
	-	`fixed`/`scientific`：指定小数点后数字位数
	-	其他：有效数字位数

-	`hex`：持久的，16进制输出**无符号整形**
-	`oct`：持久的，8进制输出**无符号整形**
-	`dec`：持久的，10进制输出整形

-	`noshowpoint`/`showpoint`：持久的，否/是强制要求包含
	小数点
-	`noshowpos`/`showpos`：持久的，要求正数前没有/有`+`
-	`nouppercase`/`uppercase`：持久的，控制作为数据转换部分
	产生任意字符小/大写，如：科学计数法中的`e`
-	`noboolalpha`/`boolalpha`：持久的，控制布尔值以数字/
	字符形式输出

####	控制

-	`unitbuf`：插入、提取操作之后清空缓冲
-	`stdio`：每次输出后清空stdout、stderr

###	输入

-	`skipws`/`noskipws`：持久的，读取之前是/否忽略空白字符
-	`ws`：从**输入流中读取空白字符**，直到不属于空白字符为止

##	`<iostream>`

-	`ifstream_withassign`：标准输入流类
	-	`cin`：标准文件stdin

-	`ofstream_withassign`：标准输出、错误、log流
	-	`cout`：标准文件stdout
	-	`cerr`：标准文件stderr
	-	`clog`：标准文件stderr

##	`<fstream>`

-	`ifstream`：文件输入流类
	-	默认操作：`ios::in`

-	`ofstream`：文件输出流类
	-	默认操作：`ios::out|ios::trunc`

-	`fstream`：文件流输入、输出类

###	例

```cpp
#include <fstream>

int main(){
	ifstream infile;
	ofstream outfile;
		// 声明指向某个文件的流变量

	infile.open(filename)
		// 打开文件：在所声明变量和实际文件间建立关联

	infile.close()
		// 关闭文件：切断流与所关联对象之间联系
}
```

####	流操作复制文件

-	逐字符复制

	```cpp
	#include<fstream>
	std::ifstream input("in", ios::binary);
	std::ofstream output("out", ios::binary);
	char ch;
	while(input.get(ch)){
		output << ch;
	}
	```

	-	使用`input >> ch`默认会跳过空白符，需要使用
		`input.unsetf(ios::skipws)`取消

-	逐行复制

	```cpp
	#include<string>
	std::string line;
	while(getline(input, line)){
		output << line << "\n";
	}
	```

	-	若文件最后没有换行符，则复制文件会末尾多`\n`

-	迭代器复制

	```cpp
	#include<iterator>
	#include<algorithm>
	input.unsetf(ios::skipws);
	copy(istream_iterator(input), istream_iterator(),
		ostream_iterator(output, "")
	);
	```

-	缓冲区复制

	```cpp
	output << input.rdbuf();
	```

	-	丢失`\n`

####	标准输出文件内容

-	`<<`操作符

	```cpp
	#include<iostream>
	#include<fstream>

	ifstream input("in");
	cout << input.rdbuf();
	```

-	`.get`方法

	```cpp
	while(input.get(*cout.rdbuf()).eof()){
		// 读取一行
		if(input.fail()){
			// `get`遇到空行无法提取字符，会设置失败标志
			input.clear();
			// 清除错误标志
		}
		cout << char(input.get());
		// 提取换行符，转换为`char`类型输出
	}
	```

-	`.get`方法2

	```cpp
	input.get(*cout.rdbuf(), EOF);
	```

##	`<sstream>`

-	基于C类型字符串`char *`编写
	-	`istrstream`：串输入流类
	-	`ostrstream`：串输出流类
	-	`strstream`：串输入、输出流类

-	基于`std::string`编写：推荐
	-	`istringstream`：串输入流类
	-	`ostringstream`：串输出流类
	-	`stringstream`：串输入、输出流类

###	例

```c
#include <sstream>

int string_to_integer(string str){
	instringstream istream(str)
		// 类似`ifstream`，使用流操作符从字符串中读取数据
	int value;

	istream >> value >> ws;
		// `>>`忽略流开头空白字符，`ws`读取尾部空白
	if(stream.fail() || !stream.eof()){
		// 如果字符串不能作为整数解析，`.fail`返回`true`
		// `.eof`返回`false`，说明字符串包含其他字符
		error("string to integer: illegal integer format");
	}
	return value;
}

string integer_to_string(int n){
	ostringstream ostream;
	ostream << n;
	return stream.str();
		//
}
```
