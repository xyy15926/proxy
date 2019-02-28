#	GDB

##	概述

GDB是GNU发布的UNIX程序调试工具，可以帮助完成

-	自定义运行程序
-	让程序在指定断点处停止
-	检查停止程序的内部状态
-	动态改变程序执行环境

##	调试准备

-	gdb会根据文件名后缀确认调试程序的语言，设置gdb自身语言
	环境，并随之改变语言环境

	-	如果程序由多种语言编译而成，gdb能根据不同语言自动
		切换语言环境

-	可以使用`info`、`show`、`set`命令查看设置当前语言环境

-	可能会缺少`glibc-debuginfo`，无法显示全部调试信息
	-	OpenSuse安装可能需要添加源

###	编译选项

-	调试C++/C程序时，要求在编译时就把调试信息添加到可执行
	文件中

	-	使用gcc/g++的`-g`参数添加源码信息
	-	否则调试时将看不见函数名、变量名，而是全部以运行时
		内存地址代替

-	关闭优化选项

	-	否则优化器会删改程序，使得某些变量不能访问、取值错误

###	启动GDB调试

-	`gdb <exe>`：直接gdb启动可执行文件`<exe>`
-	`gdb <exe> core`：用gdb同时调试可执行文件、core文件
-	`gdb <exe> <PID>`：给定进程PID，gdb自动attach该进程，
	调试已经运行的程序
	-	也可不指定PID，直接关联源码，在gdb中使用`attach`命令
		手动挂接，调试已运行程序

###	配置调试环境

####	源文件

-	gdb启动程序后，gdb会在`PATH`、当前目录中搜索程序的源文件
	`-directory`/`-d`添加源文件搜索路径
-	在gdb交互中使用`dir[rectory] <dirname>`指定源文件搜索
	路径
-	gdb中使用`list`/`l`检查gdb是否能列出源码

####	程序运行环境

在gdb中run程序之前，可能需要设置、查看程序运行环境

-	程序运行参数

	-	`set <args>`：设置程序运行时参数
	-	`show <args>`：查看已设置参数

-	环境变量

	-	`path [<dir>]`：查看、添加`<dir>`至`PATH`
	-	`show paths`：查看`PATH`
	-	`set environment var [=value]`：设置环境变量
	-	`show environment var`：查看环境变量

-	工作目录

	-	`cd <dir>`：等价于`cd`
	-	`pwd`：等价于`pwd`

-	输入输出

	-	`info terminal`：显示程序所有终端模式
	-	`run > outfile`：重定向程序输出
	-	`tty /dev/`：指定输入、输出设备

###	参数

-	`-s <file>`/`-symbols <file>`：从指定文件读取符号表
-	`-se <file>`：从指定文件读取符号表信息，并用于可执行文件
-	`-c <file>`/`-core <file>`：调试core dump产生的core文件
-	`-d <dir>`/`-directory <dir>`：添加源文件搜索路径
	-	默认搜索路径是`PATH`

###	表达式

-	表达式语法应该是**当前所调试语言**的语法

-	gdb另外还支持一些通用操作符

	-	`@`：指定存储在堆上、动态分配内存大小的长度

		```shell
		int *array = (int*)malloc(len * sizeof(int));
			# 源码
		> p *array@len
			# 打印数组`*array`
		```

	-	`::`：指定某个具体文件、函数中的变量
		-	常用于取出被隐藏的全局变量

	-	`[(type)]<addr>`：内存地址`addr`处为一个`type`类型
		变量

##	停止点

###	设置*Breakpoint*

####	`break`

`break`：在某位置设置断点

```shell
b[reak] [<probe_modifier>] [<location>] [thread <threadno>]
	[if <condition>]
```

> -	`probo_modifier`：命令处于*probe point*时需要
> > -	`-probe`：通用、自动推测探测类型
> > -	`-probe-stap`：*SystemTap*探测
> > -	`-probe-dtrace`：*DTrace*探测
> - `location`：设置断点位置
> > -	缺省：当前栈帧中的执行位置
> > -	*linespecs*：**冒号分隔**绝对位置参数（逐步精确）
> > >	- 其中可以包括：文件名、函数名、标签名、行号
> > -	`+-[n]`标识相对当前行位置
> > -	`*addr`：在程序运行的内存地址`addr`处
> > - *explicit*：类似于*linespecs*，通过多个参数给出
> - `threadno`：设置断点的线程号
> > -	缺省断点设置在所有线程上
> > -	**gdb分配**的线程编号，通过`info threads`查询
> > -	多线程被gdb停止时，**所有**运行线程都被停止，方便
		查看运行程序总体情况
> - `if <condition>`：满足条件`condition`时在断点处停止，
	缺省立即停止程序

```shell
> break factorial.c:fact:the_top
	# factorial.c文件、fact函数、the_top标签
> break +2
	# 当前行之后两行
> break *main + 4
	# `main`函数4B之后
> break -source factorial.c -function fact	
	-label the_top
	# explicit方式，同linespecs
```

-	标签：C++/C中配合`goto`使用（`label_1:`）
-	`<tab>`：补全函数名称、函数原型（重载函数）
-	若指定函数名称不唯一，gdb会列出函数原型供选择

####	`tbreak`

`tbreak`：设置一次性断点，生效后立刻被删除

####	`rbreak`

`rbreak`：对匹配正则表达式的行均设置断点

####	`condition`

`condition`：修改、设置断点`n`生效条件

```shell
> condition <bpnum> <condition>
	# 设置断点`bpnum`生效条件为`condition`
```
####	`ignore`

`ignore`：设置忽略断点`n`次数

```shell
> ignore <bpnum> <count>`
	# 忽略断点`bpnum` `count`次
```

###	设置*Watchpoint*

####	`watch`

`watch`：为某个表达式设置观察点

-	观察某个表达式，其值发生变化时停止程序

```shell
> watch [-l|-location] <expr> [thread <threadno>]
	[mask <maskvalue>]
```

> - `-l`：表示将`expression`视为地址，观察表达式指向的地址
	中的值
> - `expr`：表达式、`*`开头表示的内地地址
> - `threadno`：同`break`
> - `maskvalue`：观察值mask，只观察部分bit值

```shell
> watch foo mask 0xffff00ff
	# 观察`foo`
> watch *0xdeadbeef 0xffffff00
	# 观察内存地址`0xdeadbeef`处的值
```

-	取决于系统，观察点有可能以硬件、软件方式实现，大部分
	PowerPC、X86支持硬件观察点
	-	软件观察点通过逐步测试变量实现，程序执行速度慢得多
	-	硬件观察点不会降低程序执行速度
-	`mask`参数需要架构支持

####	`rwatch`

`rwatch`：为某表达式设置读观察点

-	当表达式值被读取时停止程序

```shell
> rwatch [-l|location] <expression> [thread <threadno>]
	[mask <maskvalue>]
```

####	`awatch`

`awatch`：为某表达式设置写观察点

-	当表达式值被修改时停止程序

```shell
> awatch [-l|-location] <expression> [thread <threadno>]
	[mask <maskvalue>]
```

###	设置*Catchpoint*

####	`catch`

`catch`：设置捕捉点捕捉事件

```shell
> cat[ch] <event>
```

> - 通用事件
> > -	`catch`：捕获异常
> > -	`exec`：调用系统调用`exec`（仅*HP-UX*下有用）
> > -	`fork`：调用系统调用`fork`（仅*HP-UX*下有用）
> > -	`load [<libname>]`：载入共享库（仅*HP-UX*下有用）
> > -	`unload [<libname>]`：卸载共享库（仅*HP-UX*下有用）
> > -	`throw`：抛出异常
> > -	`rethrow`：再次抛出异常
> > -	`vfork`：调用系统调用`vfork`时（仅*HP-UX*下有用）
> > -	`syscall`：被系统调用
> - Ada
> > -	`assert`：捕获Ada断言失败
> > -	`exception [arg]`：捕获Ada和参数匹配的异常
> > -	`handlers`：捕获Ada异常

-	捕捉点：捕捉程序运行时的一些事件，如：载入动态链接库、
	异常事件`<event>`发生时停止程序

####	`tcatch`

`tcatch`：设置一次性捕捉点，程序停止后自动删除

###	信号处理

gdb可以在调试程序时处理任何信号，可以要求gdb在收到指定信号后
停止正在执行的程序以供调试

```shell
> handle <signal> <keywords>
```

> - `signal`：需要处理的信号
> > -	信号可选使用`SIG`开头
> > -	可以定义需要处理的信号范围`SIGIO-SIGKILL`
> > -	可以使用`all`表示处理所有信号
> - `keywords`：被调试程序接收到信号后gdb的动作
> > -	`nostop`：不停止程序运行，打印消息告知收到信号
> > -	`stop`：停止程序运行
> > -	`print`：显示信息
#todo
> > -	`noprint`：不显示信息
> > -	`noigore`：gdb处理信号，交给被调试程序处理
> > -	`nopass`/`ignore`：gdb截留信号，不交给被调试程序处理

###	维护停止点

####	`clear`

`clear`：清除指定位置断点

```shell
> clear [<location>]
```

> - `location`：指定清除断点位置
> > -	缺省：清除当前栈帧中正在执行行断点
> > -	其余设置同`break`

####	`commands`

`commands`：设置断点生效时自动执行命令

-	利于自动化调试

```shell
> commands [bpnum]
> ...command-list...
> end
```

> - `bpnum`：断点序号
> > -	缺省：最近设置的断点
> > -	`5-7`：指定断点区间

##	执行

###	文件

####	`list`

`list`：打印源代码

```shell
> l[ist] [<location>]
```

> - `location`：输入的源代码
> > -	缺省/`+`：显示当前行后源代码
> > -	`-`：显示当前行前源代码
> > -	`[[start], [end]]`：显示范围内源代码（缺省表当前行）
> > -	指定单行类似`break`中`location`参数

-	一般默认显示10行，可以通过`set listsize <count>`设置

####	`forward-search`/`search`

`search`/`forward-search`：从打印出最后行开始正则搜索源码

```shell
> search/forward-search <regexp>
```

####	`revserse-search`

`reverse-search`：从打印出的最后行反向正则搜索源码

```shell
> reverse-search <regexp>
```

####	`directory`

`directory`：指定源文件搜索路径

```shell
> dir[rectory] <dir>
```

> - `dir`：源文件路径
> > -	可以指定多个搜索路径，linux下`:`分隔，windows下`;`
> > -	缺省：清除自定义源文件搜索路径信息

-	`show`查看当前

###	继续执行

####	`run`

`run`：启动程序开始调试

```shell
> r[un] [<args>]
```

> - `args`
> > -	缺省：上次`run`、`set args`指定的参数
> > -	参数中包含的`*`、`...`将被用于执行的shell扩展
> > 	（需清除参数，使用`set args`置空）

-	允许输入、输出重定向

####	`continue`

`continue`：继续执行直到之后断点、程序结束

```shell
> c[ontinue]/fg [<ignore-count>]
```

> - `ignore-count`：执行直到之后第`ingore-count`个断点
	（忽略`ignore-count-1`个断点）
> > -	缺省：1

####	`step`/`next`

`step`/`next`：单步/单行跟踪

```shell
> s[tep]/[n]ext [<count>]
```

> - `count`：执行代码行数
> > -	缺省：1

-	有函数调用，`step`进入函数（需要函数被编译有debug信息）
> > ，`next`不进入函数调用，视为一代代码

####	`stepi`/`nexti`

`stepi`/`nexti`：单条*intruction*（机器指令、汇编语句）跟踪

```shell
> s[tep]i/n[ext]i [<count>]
```

> - `count`：执行指令条数
> > -	缺省：1

####	`finish`

`finish`：运行直到当前栈帧/函数返回，并打印函数返回值、存入
值历史

####	`return`

`return`：强制函数忽未执行语句，并返回

```shell
> return [expr]
```

> - `expr`：返回的表达式值
> > -	缺省：不返回值

####	`util`

`until/u`：运行程序直到退出循环体

####	`jump`

`jump`：修改程序执行顺序，跳转至程序其他执行处

```shell
> jump [<location>]
```

> - `location`：同`break`

-	`jump`不改变当前程序栈中内容，所以在函数间跳转时，函数
	执行完毕返回时进行弹栈操作式必然发生错误、结果错误、
	core dump，所以最好在同一个函数中跳转
-	事实上，`jump`就是改变了寄存器中保存当前代码所在的内存
	地址，所以可以通过`set $pc`更改跳转执行地址

####	`call`

`call`：强制调用函数，并打印函数返回值（`void`不显示）

```shell
> call <expr>
```

-	`print`也可以调用函数，但是如果函数返回`void`，`print`
	显示并存储如历史数据中

###	查看信息

####	`print`/`inspect`

```shell
> p[rint] [/<f>]<expr>[=value]
```

> - `f`：输出格式
> > -	`x`：16进制格式
> > -	`d`：10进制格式
> > -	`u`：16进制格式显示无符号整形
> > -	`o`：8进制格式
> > -	`t`：2进制
> > -	`a`：16进制
> > -	`c`：字符格式
> > -	`f`：浮点数格式
> > -	`i`：机制指令码
> > -	`s`：
> - `expr`：输出表达式、**gdb环境变量、寄存器值、函数**
> > -	输出表达式：gdb中可以随时查看以下3种变量值
> > >	-	全局变量：所有文件可见
> > >	-	静态全局变量：当前文件可见
> > >	-	局部变量：当前scope可见
> > >	-	局部变量会隐藏全局变量，查找被隐藏变量可以使用
			`::`指定
> > -	编译程序时若开启优化选项，会删改程序，使得某些变量
		不能访问
> > -	输出环境变量、寄存器变量时，需要使用`$`前缀
> > -	函数名称：强制调用函数，类似`call`
> - `value`：修改被调试程序运行时变量值
> > -	缺省：打印变量值
> > -	`=`是C++/C语法，可以根据被调试程序改为相应程序赋值
	 	语法
> > -	可以通过`set var`实现（当变量名为gdb参数时，必须
		使用`set var`）

-	每个`print`输出的表达式都会被gdb记录，gdb会以`$1`、`$2`
	等方式记录下来，可以使用此编号访问以前的表达式

####	`examine`

`examine`/`x`：查看内存地址中的值

```shell
> examine/x /[<n/f/u>] <addr>
```

> - 输出参数：可以三者同时使用
> > -	`n`：查看内存的长度（**单元数目**）
> > -	`f`：展示格式，同`print`
> > >	-	`u`：内存单元长度
> > >	-	`b`：单字节
> > >	-	`h`：双字节
> > >	-	`w`：四字节，默认
> > >	-	`g`：八字节
> - `addr`：内存地址

```shell
> x/3uh 0x54320
	# 从内存地址`0x54320`开始，16进制展示3个双字节单位
```

####	`display`

`display`：设置自动显示变量，程序停止时变量会自动显示

```shell
> display/[<fmt>] [<expr>] [<addr>]
```

> - `fmt`：显示格式
> > -	同`print`
> - `exprt`：表达式
> - `addr`：内存地址

```shell
> display/i $pc
	# `$pc`：gdb环境变量，表示指令地址
	# 单步跟踪会在打印程序代码时，同时打印出机器指令
```

####	`undisplay`

`undisplay`：删除自动显示

```shell
> undisplay [<num>]
```

> - `num`：自动显示编号
> > -	`info`查看
> > -	可以使用`a-b`表示范围

##	查看、设置GDB环境

###	`info`

####	停止点

-	`locals`：打印当前函数中所有局部变量名、值
-	`args`：打印**当前函数**参数名、值
-	`b[reak][points] [n]`：查看断点
-	`watchpoints [n]`：列出所有观察点
-	`catch`：打印当前函数中异常处理信息
-	`line [<location>]`：查看源代码在内存中地址
-	`f[rame]`：可以打印更详细当前栈帧信息
	-	大部分为运行时内存地址
-	`display`：查看`display`设置的自动显示信息

####	线程

-	`threads`查看在**正在运行程序**中的线程信息

####	信号

-	`info signals/handle`：查看被gdb检测的信号
-	`frame`：查看当前函数语言
-	`source`：查看当前文件程序语言

####	其他

-	`terminal`：显示程序所有终端模式
-	`registers [reg]`：查看寄存器情况
	-	缺省：除浮点寄存器外所有寄存器
	-	还可以通过`print`实现
-	`all-registers`：查看所有寄存器情况（包括浮点寄存器）

###	`set`

####	停止点

-	`step-mode [on] [off]`：开启/关闭`step-mode`模式
	-	程序不会因为没有debug信息而不停止，方便查看机器码

####	环境

-	`language [lang]`：设置当前语言环境
-	`args [<args>]`：设置被调试程序启动参数
-	`environment var [=value]`：设置环境变量
-	`listsize <count>`：设置最大打印源码行数
-	`var <var=value>`：修改被调试程序运行时变量值
	-	还可以通过`print`变量修改

####	`print`

-	`address [on/off]`：打开地址输出
	-	即程序显示函数信息时，显示函数地址
	-	默认打开

-	`array [on/off]`：打开数组显示
	-	打开数组显示后，每个函数占一行，否则以逗号分隔
	-	默认关闭

-	`elements <num-of-elements>`：设置数组显示最大长度
	-	`0`：不限制数组显示

-	`null-stop [on/off]`：打开选项后，显示字符串时遇到结束符
	则停止显示
	-	默认关闭

-	`pretty [on/off]`：打开选项后，美化结构体输出
	-	打开选项后，结构体成员单行显示，否则逗号分隔

-	`sevenbit-strings [on/off]`：字符是否按照`/nnn`格式显示
	-	打开后字符串/字符按照`/nnn`显示
#todo
-	`union [on/off]`：显示结构体时是否显示其内联合体数据
	-	打开时联合体显示结构体各种值，否则显示`...`

-	`object [on/off]`：打开选项时，若指针对象指向其派生类，
	gdb自动按照虚方法调用的规则显示输出，否则gdb忽略虚函数表
	-	默认关闭

-	`static-members [on/off]`：是否对象中静态数据成员
	-	默认打开

-	`vtbl [on/off]`：选项打开，gdb将用比较规则的格式输出
	虚函数表
	-	默认关闭

###	`show`

####	执行

-	`args`：查看被调试程序启动参数
-	`paths`：查看gdb中`PATH`
-	`environtment [var]`：查看环境变量
-	`directories`：显示源文件搜索路径
-	`convenience`：查看当前设置的所有环境变量

#####	`print`

-	`address`：查看是否打开地址输出
-	`array`：查看是否打开数组显示
-	`element`：查看再打数组显示最大长度
-	`pretty`：查看是否美化结构体输出
-	`sevenbit-strings [on/off]`：查看字符显示是否打开
-	`union`：查看联合体数据输出方式
-	`object`：查看对象选项设置
-	`static-members`：查看静态数据成员选项设置
-	`vtbl`：查看虚函数显示格式选项设置

###	`shell`

`shell`：执行shell命令

```shell
> shell <shell-cmd>
```

> - `cd`：等同`> shell cd`
> - `pwd`
> - `make <make-args>`

-	Linux：使用环境变量`SHELL`、`/bin/sh`执行命令
-	Windows：使用`cmd.exe`执行命令

####	`path`

`path`：添加路径至gdb中`PATH`（不修改外部`PAHT`）

```shell
> path <dir>
```

###	GDB环境

####	环境变量

可以在gdb调试环境中自定义环境变量保存调试程序中需要的数据

```shell
> set $foo = *object_ptr
	# 设置环境变量
> show convenience
	# 查看当前设置的所有环境变量
> set $i=0
> print bar[$i++] -> contents
	# 环境变量、程序变量交互使用
	# 只需要回车重复上条命令，环境变量自动累加，逐个输出变量
```

-	环境变量使用`$`开头（定义时也需要）
-	gdb会在**首次使用时**创建该变量，在以后使用直接对其赋值
-	环境变量没有类型，可以定义任何类型，包括结构体、数组

####	寄存器

寄存器：存放了程序运行时数据

> - `ip`：程序当前运行指令地址
> - `sp`：程序当前堆栈地址

```shell
> info registers [<reg-name>]
	# 输出寄存器值，缺省除浮点外所有
> info all-registers
	# 输出所有寄存器值
> print $eip
	# 输出寄存器`eip`值
```

###	其他

-	`disassemble`：查看程序当前执行的机器码
	-	此命令会dump当前内存中指令
-	`si[gnal] <signal>`：产生信号量**发给被调试程序**
	-	`signal`：取值1-15，即Unix信号量
	-	此命令直接发送信号给被调试程序，而系统信号则是发送给
		被调试程序，但由**gdb截获**，

##	调试设置

###	`delete`

`delete`：删除断点（缺省）、自动输出表达式等

```shell
> delete [breakpoints] [bookmark] [checkpoints]
	[display] [mem] [tracepoints] [tvariable] [num]
```

> - `breakpoints`：删除断点
> - `bookmark`：从书签中删除书签
> - `checkpoints`：删除检查点
> - `display`：取消程序停止时某些输出信息
> - `mem`：删除存储区
> - `tracepoint`：删除指定追踪点
> - `tvariable`：删除追踪变量

> - `num`
> > -	缺省：删除所有断点/自动输出/书签等
> > -	指定的序号
> > >	-	`info`查看
> > >	-	可以使用`a-b`表示范围

###	`disable`

`disable`：禁用断点（缺省）、输出表达式等

```shell
> disable [breakpoints] [display] [frame-filter]
	[mem] [pretty-printer] [probes] [type-printer]
	[unwinder] [xmethod] [num]
```

####	`breakpoints`

禁用断点

```shell
> disable [breakpoints] [num]
```
-	缺省：禁用所有断点
-	仅指定的序号（`info`查看）

####	`display`

禁用程序停止时某些输出信息

####	`frame-filter`

禁用某些帧过滤器

####	`mem`

禁用存储区

####	`pretty-printer`

禁用某些打印美化

####	`probes`

禁用探测

####	`type-printer`

禁用某些类型打印

####	`unwinder`

禁用某些unwinder

####	`xmethod`

禁用某些xmethod

###	`enable`

`enable`：启用断点（缺省）、输出表达式等

```shell
> enable [breakpoints] [display] [frame-filter]
	[mem] [pretty-printer] [probes] [type-printer]
	[unwinder] [xmethod] [num]
```

####	`breakpoints`

启用断点

```shell
> enable [breakpoints] [num] [once] [delete]`
	-	`[delete]`：启用断点，生效后自动被删除
```

-	`num`：断点序号
	-	缺省：启用所有断点
-	`once`：启用断点一次
-	`delete`：启用生效后自动删除
-	`count`：启用断点`count`次

####	`display`

启用程序停止时某些输出信息

####	`frame-filter`

启用某些帧过滤器

####	`mem`

启用存储区

####	`pretty-printer`

启用某些打印美化

####	`probes`

启用探测

####	`type-printer`

启用某些类型打印

####	`unwinder`

启用某些unwinder

####	`xmethod`

启用某些xmethod

##	栈

###	`backtrace`

`backtrace`：打印函数栈

```shell
> backtrace/bt [-][<n>]
```

> - `-`：打印栈底信息
> > -	缺省：打印栈顶信息
> - `n`：打印栈数量
> > -	缺省：打印当前函数调用栈所有信息

-	一般而言，程序停止时，最顶层栈就是当前函数栈

###	`frame`

`frame`：切换当前栈

```shell
> f[rame] [<n>]
```
> - `n`：切换到第`n`个栈帧
> > -	缺省打印当前栈编号、断点信息（函数参数、行号等）

###	`up`

`up`：上移当前栈帧

```shell
> up [<n>]
```

> - `n`：上移`n`层栈帧
> > -	缺省：上移1层

###	`down`

`down`：下移当前栈帧

```shell
> down [<n>]
```
> - `n`：下移`n`层栈帧
> > -	缺省：下移1层

##	GDB命令大全

####	*aliases*

####	*breakpoints*

####	*data*

####	*files*

####	*internals*

####	*obscure*

####	*running*

####	*stack*

####	*status*

####	*support*

####	*tracepoints*

####	*user-defined*


```shell
$ gdb tst
	# `gdb`启动可执行文件tst
(gdb) l
	# `l`：list，列出源码
(gdb)
	# 直接回车：重复上次命令
(gdb) break 16
	# `break [n]`：在第`n`行设置断点
(gdb) break func
	# `break func`：在函数`func`入口处设置断点
(gdb) info break
	# `info break`：查看断点信息
(gdb) r
	# `r`：run，执行程序，会自动在断点处停止
(gdb) n
	# `n`：next，单条语句执行
(gdb) c
	# `c`：continue，继续执行（下个断点、程序结束为止）
(gdb) p i
	# `p i`：print，打印变量i的值
(gdb) bt
	# `bt`：查看函数栈
(gdb) finish
	# `finish`：退出**函数**
(gdb) 1
	# `q`：quit，退出gdb
```


