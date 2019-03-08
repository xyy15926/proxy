#	IO

##	`<iostream>`

###	标准流

-	`cin`
-	`cout`
-	`cerr`

###	*Operator*

-	`<<`：*insertion operator*，插入操作符，将数据插入流中

	-	左操作数是输出流，右操作数是需要插入流中的数据
	-	`<<`被重载后可以是任意类型的数据，如果不是字符串类型
		，`<<`会将其自动转换为字符串形式
	-	`<<`将数值转换为十进制表示形式？？？

-	`>>`：*extraction operator*，提取操作符，从输入流中读取
	**格式化数据**
	-	数据格式由**变量类型控制**
	-	一旦遇到*whitespace character*就停止读取
	-	读取数据前默认忽略所有空白字符
	-	几乎不提供任何支持检测用户输入是否有效的功能

###	*Manipulater*

（流）操纵符：控制格式化输出的一种特定类型值

####	输出

-	`endl`：将行结束序列插入输出流，确保输出字符被写入目的流
-	`setw(n)`：短暂的
-	`setprecision(digits)`：持久的，精度设置依赖于其他设置
	-	`fixed`/`scientific`：指定小数点后数字位数
	-	其他：有效数字位数
-	`setfill(ch)`：持久的
-	`left`：持久的
-	`right`：持久的
-	`fixed`：持久的，完整输出浮点数
-	`scientific`：持久的，科学计数法输出浮点数
-	`noshowpoint`/`showpoint`：持久的，否/是强制要求包含
	小数点
-	`noshowpos`/`showpos`：持久的，要求正数前没有/有`+`
-	`nouppercase`/`uppercase`：持久的，控制作为数据转换部分
	产生任意字符小/大写，如：科学计数法中的`e`
-	`noboolalpha`/`boolalpha`：持久的，控制布尔值以数字/
	字符形式输出
-	`skipws`/`noskipws`：持久的，读取之前是/否忽略空白字符
-	`ws`：从**输入流中读取空白字符**，直到不属于空白字符为止

> - 短暂的：只影响下个插入流中的数据；持久的：直到被明确改变
	为止
> - `setw`、`setprecision`、`setfill`还需要包含`<iomanip>`
> - 双操纵符中，前者为默认


##	`<ifstream>`

```cpp
#include <ifstream>

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

###	API

####	所有流

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

####	输入流

-	`>>`：读入格式化数据数据到变量中
-	`.get(var)`：读取下个字符到字符变量中，返回流本身
-	`.get()`：返回流的下个字符（`int`）
	-	使用整形返回字符是为了，可以返回一个合法字符代码范围
		外的值（`EOF`）表示文件结束
-	`.unget()`：复制流的内部指针，以便最后读取的字符能再次
	被下个`get`函数读取
-	`getline(stream, str)`：从流`stream`读取一行到字符串中

####	输出流

-	`<<`：格式化数据写入到输出流
-	`.put(ch)`：将字符`ch`写入输出流

##	`<sstream>`

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

