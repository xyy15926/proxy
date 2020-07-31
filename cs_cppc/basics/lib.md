---
title: 库
tags:
  - C/C++
categories:
  - C/C++
date: 2019-03-21 17:27:37
updated: 2019-03-07 19:56:39
toc: true
mathjax: true
comments: true
description: 库
---

##	C++库结构

定义C++库时，需要提供：*interface*、*implementation*

-	类库向用户提供了一组函数、数据类型，以实现
	*programming abstraction*

-	类库像函数一样，提供了可用于降低复杂度的方法，但也需要
	在建库时考虑更多细节，简化程度取决于接口设计的优劣

##	接口、实现

-	接口：允许库用户在不了解库实现细节的情况下使用库中库函数

	-	典型接口可提供多种定义、声明，称为*interface entry*
		-	函数声明
		-	类型定义
		-	常量定义

-	实现：说明库的底层实现细节

-	C++接口通常写在`.h`头文件中，实现在同名`.cpp`文件

###	接口设计原则

-	*unified*：统一性，接口必须按照统一主题来定义一致的抽象
-	*simple*：简单性，接口必须向用户隐藏实现的复杂性
-	*sufficient*：充分性，接口必须提供足够功能满足用户的需求
-	*general*：通用性，良好设计的接口必须有高度适用性
-	*stable*：稳定性，接口在函数底层实现改变时也要有不变的
	结构、功能

###	使用

```cpp
$ g++ -o a.out src.cpp -L /path/to/library -l lib_name
	// 动态、静态链接库均可
	// `-L`指定（额外）库文件搜索路径
$ g++ -o a.out src.cpp liblib_name.a
	// 静态链接库可以类似于`.o`文件使用
```

-	编译时使用指定链接库名只需要指定`lib_name`，编译器自动
	解析为`lib[lib_name].so`

-	gcc/ld为可执行文件链接库文件时搜索路径

	-	`/lib`、`/usr/lib64`、`/usr/lib`、`/usr/lib64`
	-	`LIBRARY_PATH`中包含的路径

##	*Static Link Library*

静态链接库`.a`/`.lib`：二进制`.o`中间目标文件的集合/压缩包

-	链接阶段被引用到静态链接库会和`.o`文件一起，链接打包到
	可执行文件中
-	程序运行时不再依赖引用的静态链接库，移植方便、无需配置
	依赖
-	相较于动态链接库浪费资源、空间
-	相较于`.o`二进制的文件，管理、查找、使用方便

###	生成静态链接库

-	linux下使用`ar`、windows下使用`lib.exe`，即可将目标文件
	压缩得到静态链接库
	
-	库中会对二进制文件进行编号、索引，以便于查找、检索

	```shell
	$ g++ -c src.cpp src.o
	$ ar -crv libsrc.a src.o
		// 生成静态库`libsrc.a`
	```

-	linux下静态库命名规范：`lib[lib_name].a`（必须遵守，因为
	链接时按此规范反解析名称）

##	*Dynamic Link Library*

动态链接/共享库`.so`/`.dll`

-	动态链接库在程序链接时不会和二进制中间文件一起，被打包
	进可执行文件中，而时在程序运行时才被
	*dynamic linker/loader*载入

-	不同程序调用相同库，在内存中只需要该共享库一份实例，规避
	空间浪费、实现资源共享

-	解决了静态库对程序更新、部署、发布的麻烦，可以通过仅仅
	更新动态库实现增量更新

-	执行环境需要安装依赖、配置环境变量（或者是编译时指定依赖
	搜索路径）

###	生成动态链接库

-	直接使用编译器即可创建动态库

	```cpp
	$ g++ -f PIC -c src.cpp -o src.o
		# PIC: position independent code
		# 创建地址无关的二进制目标文件w
	$ g++ -shared -nosname libsrc.so -o libsrc.so.1 src.o
		# 生成动态链接库
	```

-	动态链接库命名规则：`lib[libname].so`（必须按照此规则
	命名，因为链接时按照此规则反解析库名称）

###	*dynamic linker/loader*

动态载入器：先于executable模块程序工作，并获得控制权

-	对于linux下elf格式可行程序，即为`ld-linux.so*`
-	按照一定顺序搜索需要动态链接库，定位、加载

####	搜索次序

`ld-linux.so*`依次搜索以下，定位动态链接库文件、载入内存

-	elf文件的`DT_RPATH`段：**链接/编译**时指定的搜索
	库文件的路径，存储在elf文件中

	-	g++通过添加`-Wl,rpath,`、ld通过`-rpath`参数指定添加
		的路径
	-	若没有指定rpath，环境变量`LD_RUN_PATH`中路径将被添加

	```c
	$ objdump -x elf_exe
		# 查看elf文件的`DT_RPATH`
	$ g++ -Wl,-rpath,/path/to/lib
		# 在g++命令中直接给出链接参数
		# 也可以使用链接器`ld`链接时给出
	$ g++ -Wl,--enable-new-tags,-rpath,'$ORIGIN/../lib'
		# 使用相对路径
		# Makefile中使用时需要转义`$$ORIGIN/../lib`
	```

-	`LD_LIBRARY_PATH`环境变量

	-	优先级较高，可能会覆盖默认库，应该避免使用，会影响
		所有动态链接库的查找
	-	不需要root权限，同时也是影响安全性

-	`/etc/ld.so.cache`文件列表（其中包括所有动态链接库
	文件路径）

	-	`/lib`、`/lib64`、`/usr/lib`、`/usr/lib64`隐式
		默认包含，优先级较低、且逐渐降低

	-	其由`ldconfig`根据`/etc/ld.so.conf`生成，库文件添加
		进已有库路径、添加路径至`/et/ld.so.conf`后，需要通过
		`ldconfig`更新缓存才能被找到

	> - `ldconfig`具体参见*linux/shell/cmd_sysctl*

> - 因为`LD_LIBRARY_PATH`的缺点，建议使用`LD_RUN_PATH`，在
	链接时就指定动态库搜索路径


