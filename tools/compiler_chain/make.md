#	Make

##	Make基础

Make：根据指定的Makefile文件**构建**新文件

```makefile
$ make [-f makefile] [<target>]
	# 指定使用某文件中的规则，默认`makefile`/`Makefile`
```

-	make默认寻找当前目中`GNUmakefile`/`makefile`/`Makefile`
	文件作为配置
	文件
-	默认用makefile中首个目标文件作为**最终目标文件**，否则
	使用`<target>`作为目标文件

###	Make参数

-	`-b`/`-m`：忽略和其他版本make兼容性

-	`-B`/`--always-make`：总是更新/重编译所有目标

-	`-C <dir>`/`--directory=<dir>`：指定读取makefile的目录，
	相当于`$ cd <dir> && make`
	-	指定多个`-C <dir>`，make将按次序合并为最终目录
	-	`-C`时，`-w`选项自动打开

-	`--debug[=<options>]`：输出make调试信息
	-	`a`：all，输出所有调试信息
	-	`b`：basic，基本信息
	-	`v`：verbose，基本信息之上
	-	`i`：implicit，所有隐含规则
	-	`j`：jobs，执行规则中命令的详细信息，如：PID、返回码
	-	`m`：makefile，make读取、更新、执行makefile的信息
-	`-d`：等价于`--debug=a`

-	`-e`/`--environment-overrides`：环境变量覆盖makefile中值

-	`-f <file>`/`--file=<file>`/`--makefile=<file>`：指定
	makefile
	-	可以多次传递参数`-f <filename>`，所有makefile合并
		传递给make

-	`-h`/`--help`：帮助

-	`-i`/`--ignore-errors`：忽略执行时所有错误

-	`-I <dir>`/`--include-dir=<dir>`：搜索`include`makefile
	路径
	-	可以多个`-I <dir>`指定多个目录

-	`-j [<jobsum>]`/`-jobs[=<jobsum>]`：指定同时运行命令数
	-	进程数
	-	默认同时运行尽量多命令
	-	多个`-j`时最后者生效

-	`-k/--keep-going`：出错不停止运行
	-	若目标生成失败，依赖于其上目标不会继续执行

-	`-l <load>`/`--load-average[=<load>]`
-	`--max-load[=<load>]`：make命令负载

-	`-n`/`--just-print`/`--dry-run`/`--recon`：仅输出执行
	过程中命令序列，不执行

-	`-o <file>`/`--old-file=<file>`/`--assume-old=<file>`：
	不重新生成指定的`<file>`，即使目标依赖其

-	`-p`/`--print-data-base`：输出makefile中所有数据：规则、
	变量等
	```shell
	$ make -qp
		# 只想输出信息，不执行makefile
	$ make -p -f /dev/null
		# 查看执行makefile前的预设变量、规则
	```

-	`-q`/`--question`：不执行命令、不输出，仅检查指定目标
	是否需要更新
	-	`0`：需要更新
	-	`2`：有错误发生

-	`-r`/`--no-builtin-rules`：禁用make任何隐含规则

-	`-R`/`--no-builtin-variables`：禁用make任何作用于变量上
	的隐含规则

-	`-s`/`--silent`/`quiet`：禁止显示所有命令输出

-	`-S`/`--no-keep-going`/`--stop`：取消`-k`作用

-	`-t`/`--touch`：修改目标日期为最新，即组织生成目标的命令
	执行

-	`-v`/`--version`：版本

-	`-w`/`--print-directory`：输出运行makefile之前、之后信息
	-	对跟踪嵌套式调用make有用

-	`--no-print-directory`：禁止`-w`选项

-	`-W <file>`/`--what-if=<file>`/`--new-file=<file>`/`--assume-file=<file>`
	-	联合`-n`选项，输出目标更新时的运行动作
	-	没有`-n`，修改`<file>`为当前时间

-	`--warn-undefined-variables`：有未定义变量是输出警告信息

###	步骤

-	读入所有makefile
-	读入被`include`其他makefile
-	初始化（展开）文件中变量、函数，计算条件表达式
-	展开模式规则`%`、推导隐式规则、分析所并规则
-	为所有目标文件创建依赖关系链
-	根据依赖关系，决定需要重新生成的目标
-	执行生成命令

###	相关环境变量

-	`MAKEFILES`：make会将此环境变量中的文件自动`include`

	-	不建议使用，会影响所有的make动作
	-	其中文件缺失不会报错

-	`MAKEFLAGS`：make命令行参数，自动作为make参数

##	Makefile基本语法

###	控制符号

-	`#`：注释

-	`@`：消除echoing，默认make会打印每条命令

-	`-`：忽略命令出错

-	通配符同bash

	-	`*`：任意字符
	-	`?`：单个字符
	-	`[...]`：候选字符
	-	`~`：用户目录
		-	`~`：当前用户目录
		-	`~xxx`：用户`xx`目录

-	`%`：模式匹配

	```makefile
	%.o: %.c
		# 匹配所有c文件，目标为`.o`文件
	```

-	`$`：引用、展开变量，执行函数

###	引用其他Makefile

```makefile
include <filename>
```

> - `<filename>`可以是默认shell的文件模式，包含通配符、路径
> - `include`之前可以有空格，但是不能有`<tab>`（命令提示）

-	make寻找其他makefile，将其内容放当前位置

-	若文件没有明确指明为绝对路径、相对路径，make会在以下目录
	中寻找

	-	`-I`、`--include-dir`参数
	-	`/usr/local/bin/include`、`/usr/include`

-	make会`include`环境变量**MAKEFILES**中文件

	-	不建议使用环境变量`MAKEFILES`，会影响所有make

-	文件未找到，make会生成一条警告信息，但继续载入其他文件，
	完成makefile读取之后，make会重试寻找文件，失败报错

	-	可以使用`-include`/`sinclude`代替，忽略include过程
		中的任何错误

##	Makefile显式规则

```makefile
<target>: <prerequisite>
[tab]<commands>
```
> - `<target>`：目标
> - `<prerequisites>`：前置条件
> - `<commands>`：命令，和前置条件至少存在一个

```makefile
a.txt: b.txt c.txt
	cat b.txt c.txt > a.txt
```

-	makefile中规则是**生成目标**的规则

-	make自顶向下寻找可以用于生成目标的规则，生成最终目标类似
	调用函数栈

	-	前置条件/依赖类似于被调用函数
	-	命令类似于函数体
	-	目标类似于**函数返回值**

###	*Target*

目标：make的目标

-	目标通常是文件名，指明需要构建的对象
	-	文件名可以是多个，之间使用空格分隔
-	不是文件名的目标称为伪目标，视为某种操作

####	多目标

多目标规则意义是多个目标**共享规则**依赖、声明命令，并
**不是需要同时生成**多个目标

-	需要多目标中的**任何一个**时，多目标规则就会被应用，其中
	命令被执行

-	**每次只生成单独目标**的多目标规则，目标之间只是单纯的
	可以**合并简化**规则中的命令

	```makefile
	bigoutput littleoutput: text.g
		generate text.g -$(subst output,,$@) > $@

		# 等价于

	bigoutput: text.g
		generate text.g -big > bigoutput
	littleoutput: text.g
		generate text.g -little > littleoutput
	```

-	**同时生成多个目标**的多目标规则，多个目标应该满足
	**需要同时生成、不能单独修改**，否则没有必要定义为多目标
	，当然这其实也是**合并简化**规则中的命令

	```makefile
	%.tab.c %.tab.h: %.y
		bison -d $<
	```

####	*Phony Target*

#todo

伪目标：目标是某个操作的名字，每次执行都会执行命令

```makefile
.PHONY: clean
	# 明确声明是*伪目标*，可省略
clean:
	rm *.o
```

-	若省略`.PYHONY`，要求当前目中不存在同名文件，否则`make`
	认为目标已存在，不会执行命令

####	GNU规范

GNU推荐makefile中包含的伪目标、命名

-	`all`：编译所有目标
-	`clean`：删除所有被make创建的文件
-	`install`：安装已编译好程序，即将目标执行文件拷贝到指定
	目标中
-	`print`：列出改变过的源文件
-	`tar`：打包源程序备份
-	`dist`：创建压缩文件
-	`tags`：更新所有目标，以备完整地编译使用
-	`check`/`test`：测试makefile流程

####	静态库

目标`archive(member)`：指定静态库文件、及其组成

-	这种定义目标的方法就是方便`ar`命令

```makefile
foolib(hack.o kludge.o): hack.o kludge.o
	ar cr foolib hack.o kludge.o

foolib(hack.o): hack.o
	ar cr foolib hack.l kudge.o
foolib(kludge.o): kludge.o
	ar cr foolib kludge.o

	# 确实等价，但是这个看起来有点不对劲，只要传了任何一个
		# 静态库的构成，就执行命令???

foolib(*.o): hack.o kludge.o
	# 这样更好???
```

###	*Prerequisites*

前置条件/依赖：生成目标的依赖

-	通常是一组空格分隔的文件名，为空则说明目标的生成和其他
	文件无关

-	指定目标是否重新构建的判断标准，只要有一个前置文件不存在
	、有过更新（时间戳比较），目标就需要更新

-	若前置文件不存在，则需要建立以其作为目标的规则用于生成，
	`make target`时会自动调用

```makefile
source: file1 file2 file3
	# 利用伪目标+前置条件，同时构建多个文件
```

###	*Commands*

命令：更新、构建文件的方法

-	在linux下默认使用环境变量`SHELL`（`/bin/sh`）执行命令，
-	在MS-DOS下没有`SHELL`环境变量，会在`PATH`环境变量中寻找
	，并自动加上`.exe`、`.bat`、`.sh`等后缀

####	`<tab>`

每行命令前必须有一个`<tab>`，否则需要使用提前声明

```makefile
.RECIPEPREFIX=>
all:
> echo Hello, world
```

####	Shell进程

**每行**命令在单独的shell进程中执行，其间没有继承关系
（即使是同一个规则中）

-	多条命令可以使用`;`分隔

	```makefile
	var-kept:
		export foo=bar; echo "foo=[$$foo]"
	```

-	可类似python`\`换行

	```makefile
	var-kept:
		export foo=bar; \
		echo "foo=[$$foo]"
	```

-	使用`.ONESHELL`命令

	```makefile
	.ONESHELL
	var-kept:
		export foo=bar
		echo "foo=[$$foo]"
	```

####	嵌套执行Make

大工程中不同模块、功能源文件一般存放在不同目录，可以为每个
目录单独建立makefile

-	利于维护makefile，使得其更简洁
-	利于分块/分段编译
-	最顶层、调用make执行其他makefile的makefile称为总控

```makefile
subsystem:
	cd subdir && $(MAKE)
	# 等价
subsystem:
	$(MAKE) -C subdir

subsystem:
	cd subdir && $(MAKE) -w MAKEFLAGS=
	# 将命令行参数`MAKEFLAGS`置空，实现其不向下级传递
	# 指定`-w`参数输出make开始前、后信息
```

###	搜索路径

####	`VPATH`

`VPATH`：makefile中定义的特殊环境变量，指明寻找依赖文件、
目标文件的路径

```makefile
VPATH = src:../src
```

-	`:`分隔路径
-	当前目录优先级依旧最高

####	`vpath`

`vpath`：make关键字，指定不同**模式**文件不同搜索目录

```c
vpath <pattern> <directories>
vpath %.h ../headers
	# 指明`<pattern>`模式文件搜索目录
vpath <pattern>
	# 清除`<pattern>`模式文件搜索目录设置
vpath
	# 清除所有`vapth`设置
```

> - `<pattern>`中使用`%`匹配0或若干字符
> - `vpath`可以重复为某个模式指定不同搜索策略，按照出现顺序
	先后执行搜索

##	隐含规则

-	隐含规则是一种惯例，在makefile中没有书写相关规则时自动
	照其运行

	-	隐含规则中优先级越高的约经常被使用
	-	甚至有些时候，显式指明的目标依赖都会被make忽略
		```makefile
		foo.o: foo.p
			# Pascal规则出现在C规则之后	
			# 若当前目录下存在foo.c文件，C隐含规则生效，生成
				# foo.o，显式依赖被忽略
		```
	-	很多规则使用**后缀规则**定义，即使使用`-r`参数，其
		仍会生效

-	隐含规则会使用系统变量

	-	`CPPFLAGS`/`CFLAGS`：C++/C编译时参数

-	可以通过**模式规则**自定义隐含规则，更智能、清晰

	-	**后缀规则**有更好的兼容性，但限制更多

###	常用隐含规则

####	编译C

-	目标：`<n>.o`

-	依赖包含：`<n>.c`

-	生成命令
	```makefile
	$(CC) -c $(CPPFLAGS) $(CFLAGS)
	```

####	编译CPP

-	目标：`<n>.o`

-	依赖包含`<n>.cc`/`<n>.c`

-	生成命令
	```makefile
	$(CXX) -c $(CPPFLAGS) $(CFLAGS)
	```

####	编译Pascal

-	目标：`<n>.p`

-	依赖包含：`<n>.p`

-	生成命令
	```makefile
	$(PC) -c $(PFLAGS)
	```

####	编译Fortran/Ratfor

-	目标：`<n>.o`

-	依赖包含：`<n>.f`/`<n>.r`

-	生成命令
	```makefile
	$(FC) -c $(FFLAGS)
		# `.f`
	$(FC) -c $(FFLAGS) $(CPPFLAGS)
		# `.F`
	$(FC) -c $(FFLAGS) $(RFLAGS)
		# `.r`
	```

####	预处理Fortran/Ratfor

-	目标：`<n>.f`

-	依赖包含：`<r>.r`/`<n>.F`

-	生成命令
	```makefile
	$(FC) -F $(CPPFLAGS) $(FFLAGS)
		# `.F`
	$(FC) -F $(FFLAGS) $(RFLAGS)
		# `.r`
	```

> - 转换Ratfor、有预处理的Fortran至标准Fortran

####	编译Modula-2

-	目标：`<n>.sym`/`<n>.o`

-	依赖包含：`<n>.def`/`<n>.mod`
-	生成命令
	```makefile
	$(M2C) $(M2FLAGS) $(DEFFLAGS)
		# `.def`
	$(M2C) $(M2FLAGS) $(MODFLAGS)
		# `.mod`
	```

####	汇编汇编

-	目标：`<n>.o`

-	依赖包含：`<n>.s`

-	生成命令：默认使用编译器`as`
	```makefile
	$(AS) $(ASFLAGS)
		# `.s`
	```

####	预处理

-	目标：`<n>.s`

-	依赖包含：`<n>.S`

-	生成命令：默认使用预处理器`cpp`
	```makefile
	$(CPP) $(ASFLAGS)
		# `.S`
	```

####	链接object

-	目标：`<n>`

-	依赖包含：`<n>.o`

-	生成命令：默认使用C工具链中链接程序`ld`
	```makefile
	$(CC) <n>.o $(LOADLIBS) $(LDLIBS)
	```

####	Yacc C

-	目标：`<n>.c`

-	依赖包含：`<n>.y`

-	生成命令
	```makefile
	$(YACC) $(YFALGS)
	```

####	Lex C

-	目标：`<n>.c`

-	依赖包含：`<n>.c`

-	生成命令
	```makefile
	$(LEX) $(LFLAGS)
	```

####	Lex Ratfor

-	目标：`<n>.r`

-	依赖包含：`<n>.l`

-	生成命令
	```makefile
	$(LEX) $(LFLAGS)
	```

####	创建Lint库

-	目标：`<n>.ln`

-	依赖包含：`<n>.c`/`<n>.y`/`<n>.l`

-	生成命令
	```makefile
	$(LINT) $(LINTFLAGS) $(CPPFLAGS) -i
	```

####	创建静态链接库

-	目标：`<archive>(member.o)`

-	依赖包含：`member`

-	生成命令
	```makefile
	ar cr <archive> member.o
	```

> - 即使目标传递多个`memeber.o`，隐含规则也只会解析出把首个
	`.o`文件添加进静态链接库中的命令

	```makefile
	(%.o): %.o
		$(AR) rv $@ $*.o
		# 此命令可以得到添加所有`member.o`的命令
		# 但是此时`$*=member.o member`
	```

###	隐含规则使用变量

隐含规则使用的变量基本都是预先设置的变量

-	makefile中改变
-	make命令环境变量传入
-	设置环境变量
-	`-R`/`--no-builtin-variable`参数取消变量对隐含规则作用

####	命令

-	`AR`：函数打包程序，默认`ar`
-	`AS`：汇编语言编译程序，默认`as`
-	`CC`：C语言编译程序，默认`cc`
-	`CXX`：C++语言编译程序，默认`c++`/`g++`
-	`CPP`：C程序预处理器，默认`$(CC) -E`/`cpp`
-	`FC`：Fortran、Ratfor编译器、预处理程序，默认`f77`
-	`PC`：Pascal语言编译程序，默认`pc`
-	`LEX`：Lex文法分析器程序（针对C、Ratfor），默认`lex`
-	`YACC`：Yacc文法分析器程序（针对C、Ratfor），默认
	`yacc -r`
-	`GET`：从`SCCS`文件中扩展文件程序，默认`get`
-	`CO`：从`RCS`文件中扩展文件程序，默认`co`
-	`MAKEINFO`：转换Texinfo `.texi`到Info程序，默认
	`makeinfo`
-	`TEX`：转换TeX至Tex DVI程序，默认`tex`
-	`TEXI2DVI`：转换Texinfo至Tex DVI程序，默认`texi2dvi`
-	`WEAVE`：转换Web至TeX程序，默认`weave`
-	`TANGLE`：转换Web至Pascal程序，默认`tangle`
-	`CTANGEL`：转换C Web至C，默认`ctangle`
-	`RM`：删除文件命令，默认`rm -f`

####	命令参数

未指明默认值则为空

-	`ARFLAGS`：静态链接库打包程序AR参数，默认`rv`
-	`ASFLAGS`：汇编语言汇编器参数
-	`CFLAGS`：C编译器参数
-	`CXXFLAGS`：C++编译器参数
-	`CPPFLAGS`：C预处理参数
-	`LDFLAGS`：链接器参数
-	`FFLAGS`：Fortran编译器参数
-	`RFLAGS`：Fatfor的Fortran编译器参数
-	`LFLAGS`：Lex文法分析器参数
-	`YFLAGS`：Yacc文法分析器参数
-	`COFLAGS`：RCS命令参数
-	`GFLAGS`：SCCS `get`参数

###	隐含规则链

make会努力**自动推导**生成目标的一切方法，无论**中间目标**
数量，都会将显式规则、隐含规则结合分析以生成目标

-	中间目标不存在才会引发中间规则

-	目标成功产生后，中间目标文件被删除
	-	可以使用`.SECONDARY`强制声明阻止make删除该中间目标
	-	指定某模式为伪目标`.PRECIOUS`的依赖目标，以保存被
		隐含规则生成的**符合该模式中间文件**

-	通常makefile中指定成目标、依赖目标的文件不被当作中间目标
	，可以用`.INTERMEDIATE`强制声明目标（文件）是中间目标

-	make会优化特殊的隐含规则从而不生成中间文件，如从文件
	`foo.c`直接生成可执行文件`foo`

##	模式规则

模式规则：隐式规则可以看作**内置**模式规则

> - 目标定义包含`%`，表示任意长度非空字符串
> - 依赖中同样可以使用`%`，但是其取值取决于目标
> - 命令中不使用模式`%`，使用*自动化变量*

-	模式规则**没有确定目标**，不能作为最终make目标

	-	但是符合模式规则的某个具体文件可以作为最终目标
	-	不需要作为显式规则的目标，如：`archive(member)`作为
		静态库隐含规则目标

-	模式的**启用取决于其目标**，`%`的**解析同样取决于目标**
	（因为根据目标查找、应用模式规则）

-	模式规则类似于隐含规则，给出符合**某个模式**的某类目标
	的依赖、生成命令

-	`%`的**展开**发生在变量、函数展开后，发生在运行时

###	静态模式

静态模式：**给定目标候选范围**的模式，限制规则只能应用在以
给定范围文件作为目标的情况

```makefile
<target>: <target-pattern>: <prereq-patterns>
	<commands>
```

> - `<target>`：目标候选范围，可含有通配符
> - `<target-pattern>`：**所有目标文件**满足的模式
> - `<prereq-pattern>`：目标相应依赖

-	简单例子

	```makefile
	objects = foo.o bar.o
	all: $(objects)
	$(objects): %.o: %.c
		$(CC) -c $(CFLAGS) $< -o $@

		# 等价于

	foo.o: foo.c
		$(CC) -c $(CFLAGS) foo.c -o foo.o
	bar.o: bar.c
		$(CC) -c $(CFLAGS) bar.c -o bar.o
	```

-	静态模式+`filter`函数筛选范围

	```makefile
	files = foo.elc bar.o lose.o
	$(filter %.o,$(files)): %.o: %.c
		$(CC) -c $(CFLAGS) $< -o $@
	$(filter %.elc,$(files)): %.elc: %.el
		emacs -f batch-byte-compile $<
	```

###	重载内建隐含规则

```makefile
%.o: %c
	$(CC) -c $(CPPFLAGS) $(CFLAGS) -D $(date)
	# 重载内建隐含规则
%o: %c
	# 命令置空，取消内建隐含规则
```

###	后缀规则

> - 双后缀规则：定义一对目标文件后缀、依赖后缀
> - 单后缀规则：定义一个依赖后缀

```makefile
.c.o:
	# 等价于`%.o: %c`
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<
.c:
	# 等价于`%: %.c`
	$(CC) -c $(CFLAGS) $(CPPFLAGS) -o $@ $<
```

-	后缀规则中不能有任何依赖文件，否则不是后缀规则，后缀被
	认为是目标文件

-	后缀规则中必须有命令，否则没有任何意义，这不会移去内建
	的隐含规则

-	后缀规则定义中的后缀需要是make所认识的，可以使用伪目标
	`.SUFFIXES`修改make认识的后缀

	```makefile
	.SUFFIXES:
		# 删除默认后缀
	.SUFFIXES: .c .o .h
		# 添加自定义后缀
	```

	-	变量`$SUFFIXE`用于定义默认后缀列表，不应该修改

	-	`-r`/`--no-builtin-rules`同样会清空默认后缀列表

-	后缀规则是老式定义隐含规则的方法，会被逐步取代，事实上
	后缀规则在makefile载入后会被转换为模式规则

###	模式规则搜索算法

设目标为`src/foo.o`

-	将目标目录部分、文件名部分分离，得到`src/`、`foo.o`

-	搜索所有模式规则列表，创建目标和`src/foo.o`匹配的模式
	规则列表

	-	若模式规则列表中有目标匹配所有文件的模式（如`%`），
		则从列表中移除其他模式
	-	移除列表中没有命令的规则

-	对列表中首个模式规则

	-	将`src/foo.o`或`foo.o`匹配目标，推断`%`匹配非空部分
		**茎S**
	-	把依赖中`%`替换为茎S，如果依赖项中没有包含目录，
		尝试将`src`添加在其前
	-	检查所有依赖文件存在、理当存在（文件被定义为其他规则
		的目标文件、显式规则的依赖文件）
	-	若有依赖文件存在、理当存在或没有依赖文件，此规则被
		采用，退出算法

-	若没有找到合适模式规则，则检查列表中下个规则是否可行

-	若没有模式规则可以使用，检查`.DEFAULT`规则存在性，存在
	则应用

##	变量、赋值

Makefile中定义的变量类似C++/C中的宏

-	代表一个字符串，在makefile中执行的时候展开在所在位置
-	变量可以改变值

###	赋值

Makefile内自定义变量

```makefile
txt = Hello World
	# 自定义变量
test:
	@echo $(txt)
	echo ${txt}
	# 调用变量
	# 若变量名为单个字符，可以省略括号，但不建议省略
```

-	`=`：*lazy set*，在**执行**时扩展

	-	可以使用**任意位置定义（可能还未定义）**的变量赋值
	-	允许递归扩展，make报错

-	`:=`：*immediate set*，在**定义/赋值**时扩展完毕

	-	只允许使用**之前已定义**变量赋值（否则为空）

-	`?=`：*set if absent*，只有变量为空时才设置值

-	`+=`：*append*，将值追加到变量的尾部

	-	若前面变量有定义，`+=`会继承前一次操作符`:=`/`=`
	-	对于`=`定义变量，make自动处理“递归”

####	`define`

`define`可以换行定义变量

-	变量类似宏的行为、可换行定义变量，方便定义命令包

```makefile
define run-yacc
	# `define`后跟变量名作为命令包名称
	yacc $(firstword $^); \
	mv y.tab.c $@
endef
	# 结束定义

foo.c: foo.y
	$(run-yacc)
	# 使用命令包
```

####	`override`

-	若变量由make命令行参数`-e`设置，makefile中默认忽略对其
	赋值
-	需要显式使用`override`关键字设置

```makefile
override <variable> = <value>
override <variable> := <value>
override define <variable>
```

####	`export`

上级makefile中变量可以显式`export`传递到下层makefile中，
但是不会覆盖下层中定义的变量（除指定`-e`参数）

```makefile
export <variable>[=value]
	# 传递变量至下级makefile中
unexport <variable>
	# 禁止变量传递至下级makefile中

export variable = value
	# 等价
variable = value
export variable
```

-	`export`后面不指定具体传递变量，表示传递所有变量
-	`MAKEFLAGS`、`SHELL`两个变量总是会传递到下级makefile中

###	系统环境变量

make运行时系统环境变量、命令行环境变量可以被载入makefile

-	默认makefile中定义变量覆盖系统环境变量
-	`-e`参数则表示makefile中变量被覆盖

```makefile
test:
	@echo $$HOME
	# 需要对`$`转义
```

###	*Target-Specific Variable*

目标/局部变量：作用范围局限于规则、连带规则中

```makefile
<target ...>: [override] <variable-assignment>

prog: CFLAGS  = -g
prog: prog.o foo.o bar.o
	$(CC) $(CFLAGS) prog.o foo.o bar.o

prog.o: prog.c
	$(CC) $(CFLAGS) prog.c

foo.o: foo.c
	$(CC) $(CFLAGS) foo.c

bar.o: bar.c
	$(CC) $(CFLAGS) bar.c
```

###	*Pattern-Specific Variable*

模式变量：给定模式，变量定义在符合模式的**所有目标**上

```makefile
<pattern ...>: [override]<variable-assignment>

%.o: CFLAGS = -o
```

###	*Implicit Variables*

内置变量：主要是为了跨平台的兼容性

-	`$(CC)`：当前使用的编译器

	```makefile
	output:
		$(CC) -o output input.c
	```

-	`$(MAKE)`：当前使用的Make工具

-	`$(MAKECMDGOLA)`：make目标列表

###	*Automatic Variables*

自动化变量：应用规则时被自动赋予相应值（一般是文件）的变量

-	`$@`：**当前需要生成**的目标文件
	-	多目标规则中，`$@`也只表示**被需要**目标
-	`$*`：匹配符`%`匹配部分
	-	若目标中没有`%`模式符，`$*`不能被推导出，为空
	-	GNU make中，目标中没有`%`，`$*`被推导为除后缀部分，
		但是很可能不兼容其他版本，谨慎使用
-	`$<`：首个前置条件
-	`$%`：仅当目标是函数库文件，表示目标成员名，否则为空
	-	目标为`foo.a(bar.o)`：`$%`为`bar.o`、`$@`为`foo.a`

-	`$?`：比目标更新的前置条件，空格分隔
-	`$^`：所有前置条件，会取出其中重复项
-	`$+`：类似于`$^`，但是剔除重复依赖项

> - 自动化变量只应出现在规则的**命令**中
> - 自动化变量值与当前规则有关
> - 其中`$@`、`$*`、`$<`、`$%`扩展后只会为单个文件，`$?`、
	`$^`、`$+`扩展后可能是多个文件

```makefile
dest/%.txt: src/%.txt
	@[ -d test ] || mkdir dest
	cp $< $@
```

####	D、F

-	7个自动化变量可以搭配`D`、`F`取得相应路径中目录名、
	文件名
-	新版本GNU make可以使用函数`dir`、`notdir`代替`D`/`F`

> - `D`/`dir`：目录带有最后`/`，若为当前目录则为`./`
> - `F`/`nodir`：文件名
> - 对可能会扩展为多文件的`$?`、`$^`、`$+`，`D`/`F`处理后
	返回同样是多个目录/文件

```makefile
$(@D)
$(dir $@)
	# `$@`的目录名
$(@F)
$(nodir $@)
	# `$@`的文件名

$(?D)
$(dir $?)
	# `$?`中所有目录，空格分隔
$(?F)
$(nodir $?)
	# `$?`中所有文件，空格分隔
```

##	控制语句

###	`if`

```makefile

<conditional-directive>
<text-if-true>
[
else
<text-if-false>
]
endif
```


-	`ifeq`：比较参数是否相等

-	`ifneq`：比较参数是否不等

	```makefile
	ifeq ($(CC), gcc)
		# 也可以用单/双引号括起，省略括号
		libs=$(libs_for_gcc)
	else
		libs=$(normal_libs)
	endif
	```

-	`ifdef`

	```makefile
	bar =
	foo = $(bar)
	# `foo`有定义
	ifdef foo
		frobozz = yes
		# 此分支
	else
		frobozz = no
	endif

	foo =
	# `foo`未定义
	ifdef foo
		frobozz = yes
	else
		frobozz = no
		# 此分支
	endif
	```

-	`ifndef`

> - `<conditional-directive>, else, endif`行可以有多余空格，
	但是不能以`<tab>`开头，否则被认为是命令
> - make在读取makefile时就计算表达式值、选择语句，所以最好
	别把自动化变量放入条件表达式中
> - make不允许把条件语句分拆放入两个文件中

###	`for`

```makefile
LIST = one two three

all:
	for i in $(LIST); do \
		echo $$i;
		// 这里是shell中的变量，需要转义
	done
all:
	for i in one two three; do
		echo $$i;
	done
```

##	内建函数

```makefile
$(function parameters)
${function paremeters}
```

###	Make控制函数

提供一些函数控制make执行

-	检测运行makefile的运行时信息，根据信息决定make继续执行
	、停止

####	`error`

产生错误并退出make，错误信息`<text>`

```makefile
$(error <text...>)

ifdef ERROR_001
$(error error is $(ERROR_001))
endif

ERR = $(error found an error)
.PHONY: err
err:
	err: ; $(ERR)
```

####	`warning`

类似`error`函数，输出警告信息，继续make

###	其他函数

####	`shell`

执行shell命令的输出作为函数返回值

```makefile
srcfiles := $(shell echo src/{00..99}.txt)
```

-	函数会创建新shell执行命令，大量使用函数可能造成性能下降
	，尤其makefile的隐晦规则可能让shell函数执行次数过多

####	`wildcard`

在变量中展开**通配符**`*`

```makefile
srcfiles := $(wildcard src/*.txt)
	# 若不展开，则`srcfiles`就只是字符串
	# 展开后则表示所有`src/*.txt`文件集合
```

###	字符串处理函数

####	`subst`

文本替换

```makefile
$(subst <from>,<to>,<text>)
	# `subst`函数头

$(subst ee,EE,feet on the street)
	# 替换成*fEEt on the strEET*
```

####	`patsubst`

模式匹配的替换

```makefile
$(patsubst <pattern>,<replacement>,<text>)
	# 函数头文件
$(patsubst %.c,%o,x.c.c bar.c)
	# 替换为`x.c.o bar.o`

foo := a.o b.o c.o
$(variable: <pattern>=<replacement>)
	# `patsubst`函数的简写形式
bar := $(foo:%.o=%.c)
	# `$(bar)`变为`a.c b.c c.c`

$(variable: <suffix>=<replacement>)
	# 没有模式匹配符`%`则替换结尾
bar := $(foo:.o=.c)
	# `$(bar)`变为`a.c b.c c.c`
```

####	`strip`

去字符串头、尾空格

```makefile
$(strip <string>)
$(strip a b c)
```

####	`findstring`

在`<in>`中查找`<find>`，找到返回`<find>`，否则返回空串

```makefile
$(findstring <find>,<in>)
$(findstring a,a b c)
```

####	`filter`

以`<pattern>`模式过滤`<text>`字符串中单词，返回符合模式的
单词

```makefile
$(filter <pattern..>,<text>)

sources := foo.c bar.c baz.s ugh.h
foo: $(sources)
	cc $(filter %.c %.s, $(sources)) -o foo
	# 返回`foo.c bar.c baz.s`
```

####	`filter-out`


以`<pattern>`模式过滤`<text>`字符串中单词，返回不符合模式的
单词

```makefile
objects := main1.o foo.o main2.o bar.o
mains=main1.o main2.o
$(filter-out $(mains), $(objects))
	# 返回`foo.o bar.o`
```

####	`sort`

对`<list>`中单词升序排序

```makefile
$(sort <list>)

$(sort foo bar lose)
	# 返回`bar foo lose`
```

####	`word`

取字符串`<text>`中第`<n>`个单词

```makefile
$(word <n>,<text>)

$(word 2, foo bar baz)
	# 返回`bar`
```

####	`wordlist`

从`<text>`中取`<s>-<e>`单词（闭区间）

```makefile
$(wordlist <s>,<e>,<text>)

$(wordlist 2, 3, foo bar baz)
	# 返回`bar baz`
```

####	`words`

统计`<text>`中单词个数

```makefile
$(word <text>)

$(word, foo bar baz)
	# 返回3
```

####	`firstword`

取`<text>`中首个单词

```makefile
$(firstword <text>)

$(firstword foo bar)
	# 返回`foo`
```

###	文件名操作函数

####	`dir`

从文件名**序列**中取出目录部分

-	最后`/`之前部分
-	若没有`/`则返回`./`

```makefile
$(dir <names...>)

$(dir src/foo.c hacks)
	# 返回`src/ ./`
```

####	`notdir`

从文件名**序列**中取出非目录部分（最后`/`之后部分）

```makefile
$(notdir <names...>)

$(notdir src/foo.c hacks)
	# 返回`foo.c hacks`
```

####	`suffix`

从文件名**序列**中取出各文件名后缀

```makefile
$(suffix <names...>)

$(suffix src/foo.c src-1.0/bar.c hacks)
	# 返回`.c .c`
```

####	`basename`

从文件名**序列**中取出各文件名“前缀”（除后缀外部分）

```makefile
$(basename <names...>)

$(basename src/foo.c src-1.0/bar.c hacks)
	# 返回`src/foo src-1.o/bar hacks`
```

####	`addsuffix`

把后缀`<suffix>`添加到文件名**序列**中每个单词后

```makefile
$(addsuffix <suffix>,<names...>)

$(addsuffix .c, foo bar)
	# 返回`foo.c bar.c`
```

####	`addprefix`

把后缀`<prefix>`添加到文件名**序列**中每个单词后

```makefile
$(addprefix <prefix>,<names...>)

$(addprefix src/, foo bar)
	# 返回`src/foo src/bar`
```

####	`join`

把`<list2>`中单词**对应**添加到`<list1>`中单词后

-	较多者剩余单词保留

```makefile
$(join <list1>,<list2>)

$(join aaa bbb, 111 222 333)
	# 返回`aaa111 bbb222 333`
```

###	控制函数

####	`foreach`

循环函数，类似于Bash中的`for`语句

-	把`<list>`中单词逐一取出放到参数`<var>`所指定的变量中
-	再执行`<text>`所包含的表达式，每次返回一个字符串
-	循环结束时，返回空格分隔的整个字符串

```makefile
$(foreach <var>,<list>,<text>)

names := a b c d
files := $(foreach n,$(names),$(n).o)
	# 返回`a.o b.o c.o d.o`
```

> - `<var>`是临时局部变，函数执行完后将不再作用

####	`if`

类似于make中的`ifeq`

-	`<condition>`为真（非空字符串），计算`<then-part>`返回值
-	`<condition>`为假（空字符串），计算`<else-part>`、返回空
	字符串

```makefile
$(if <condition>,<then-part>,[<else-part>])
```

####	`call`

创建新的**参数化函数**的函数 

-	创建表达式`<expression>`，其中可以定义很多参数
-	用`call`函数向其中传递参数，`<expression>`返回值即`call`
	返回值

```makefile
$(call <expression>,<param1>,<param2>,...>

reverse = $(1) $(2)
foo = $(call reverse,a,b)
	# 返回`a b`
reverse = $(2) $(1)
foo = $(call reverse,a,b)
	# 返回`b a`
```

> - `<expression>`要先创建为变量，然后不带`$`传递

####	`origin`

返回变量的来源

-	`undefined`：`<variable>`未定义
-	`default`：make默认定义变量
-	`environment`：环境变量，且`-e`选项未开
-	`file`：定义在makefile中
-	`command line`：命令行定义环境变量
-	`override`：`override`关键字定义
-	`atomatic`：命令运行中自动化变量

```makefile
$(origin <variable>)

ifdef bletch
ifeq "$(origin bletch)" "environment"
bletch = barf, gag, etc
endif
endif
```

> - `<variable>`不操作变量值，不带`$`传递

##	Makefile技巧

###	案例

```makefile
edit: main.o kdd.o command.o display.o \
	insert.o search.o files.o utils.o
	cc -o edit main.o kbd.o command.o dispaly.o\
		insert.o search.o files.o utils.o

main: main.c defs.h
	cc -c main.c
kbd.o: kbd.c defs.h
	cc -c kbd.c
command.o: command.c defs.h command.h
	cc -c command.c
display.o: display.o defs.h buffer.h
	cc -c display.c
insert.o: insert.c defs.h buffer.h
	cc -c insert.c
search.o: search.c defs.h buffer.h
	cc -c search.c
files.o: files.c defs.h buffer.h command.h
	cc -c files.c
utils.o utils.c defs.h
	cc -c utils.c

clean:
	rm edit main.o kbd.o command.o display.o \
		insert.o search.o files.o utils.o

.PHONY: edit clean
	# 设置`edit`、`clean`为伪目标
```

###	利用变量简化目标

```makefile
objects = main.o kbd.o command.o display.o \
	insert.o search.o files.o utils.o

edit: $(objects)
	cc -o edit $(objects)
	# 以下同上
```

###	隐式模式自动推导

```makefile
objects = main.o kbd.o command.o display.o \
	insert.o search.o files.o utils.o
edit: $(objects)
	cc -o edit $(objects)
main.o: defs.h
kbd.o: defs.h command.h
command.o: defs.h command.h
display: defs.h buffer.h
insert.o: defs.h buffer.h
search.o: defs.h buffer.h
files.o: defs.h buffer.h command.h
utils.o: defs.h

clean:
	rm edit $(objects)

.PHONY: clean
```

> - 利用隐式模式自动推导文件、文件依赖关系

###	利用变量提取依赖

```makefile
objects = main.o kbd.o command.o display.o \
	insert.o search.o files.o utils.o
edit: $(objects)
	cc -o edit $(objects)

$(objects): defs.h
kbd.o command.o files.o: command.h
display.o insert.o search.o files.o: buffer.h

clean:
	rm edit $(objects)

.PHONY: clean
```

> - 文件变得简单，但是依赖关系不再清晰

###	自动生成依赖

-	大部分C++/C编译器支持`-M`选项，自动寻找源文件中包含的
	头文件，生成依赖关系

-	GNU建议为每个源文件自动生成依赖关系，存放在一个文件中，
	可以让make自动更新依赖关系文件`.d`，并包含在makefile中

```makefile
%.d: %.c
	@set -e; rm -f $@; \
	$(cc) -M $(CPPFLAGS) $< > $@.$$$$; \
		# 生成中间文件
		# `$$$$`表示4位随机数
	sed 's,/($*/)/.o[ :]*,/1.o $@ : ,g' < $@.$$$$ > $@; \
		# 用`sed`替换中间文件target
		# `xxx.o` -> `xxx.o xxx.d`
	rm -f $@.$$$$
		# 删除中间文件

source = foo.c bar.c
include $(sources: .c=.d)
```

