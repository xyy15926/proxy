#	关于Python安装和配置的问题

##	Python

###	Python3包管理安装

-	`zlib-devel`
-	`bzip2-devel`
-	`readline-devel`
-	`openssl-devel`
-	`sqlite(3)-devel`

原始Centos7缺少的依赖，以上的依赖也有各自的依赖

###	Python Implementation

名称中flag体现该发行版中python特性

-	`-d`：with pydebug

-	`-m`：with pymalloc
	-	pymalloc是specialized object allocator
	-	比系统自带的allocator快，且对python项目典型内存
		分配模式有更少的开销 
	-	其使用c的malloc函数获得更大的内存池
	-	original
		Pymalloc, a specialized object allocator written by Vladimir Marangozov, 
		was a feature added to Python 2.1. Pymalloc is intended to be faster 
		than the system malloc() and to have less memory overhead for allocation 
		patterns typical of Python programs. The allocator uses C's malloc() 
		function to get large pools of memory and then fulfills smaller memory 
		requests from these pools.J

-	`-u`：with wide unicode

>	注意：有时也有可能只是hard link

###	Python源码安装依赖

根据包用途、名称可以确认对应的应用，缺少的估计就是相应的
`-devel(-dev)`

##	Python、Python包配置

###	Pip

配置文件：`~/.config/pip/pip.conf`

-	index-url：pip下载python包地址
-	format：pip list输出格式（legacy，columns）

###	Conda

配置文件：`~/.condarc`

