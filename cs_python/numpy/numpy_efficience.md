---
title: 
categories:
  - 
tags:
  - 
date: 2021-03-11 09:47:45
updated: 2021-03-11 09:47:45
toc: true
mathjax: true
description: 
---

##	Miscellaneous

###	性能调优

|Function|Desc|
|-----|-----|
|`setbufsize(size)`|设置*ufunc*使用的缓冲区大小|
|`getbufsize()`||
|`shares_memory(a,b[,max_work])`||
|`may_share_memory(a,b[,max_work])`||
|`byte_bounds(a)`|返回指向数组结尾的指针|

###	Array Mixin

|Function|Desc|
|-----|-----|
|`lib.mixins.NDArrayOperatorsMixin`|定义了所有使用`array_ufunc`特殊方法|
|`lib.NumpyVersion(vstring)`|解析、比较NumPy版本|
|`get_include()`|返回头文件目录|
|`deprecate(*args,**kwargs)`|废弃警告|
|`deprecate_with_doc(msg)`||
|`who([vardict])`|在指定字典中打印数组|
|`disp(mesg[,device,linefee])`|展示信息|

##	浮点错误处理

-	错误处理
	-	设置硬件平台上注册的错误处理，如：除零错误
	-	基于线程设置

|Function|Desc|
|-----|-----|
|`seterr([all,divide,over,under,invalid])`|设置浮点错误处理|
|`seterrcall(func)`|设置浮点错误回调或log|
|`geterr()`|获取当前处理浮点错误的方法|
|`geterrcall()`|获取当前处理浮点错误回调函数|
|`errstate(**kwargs)`|浮点错误处理上下文|
|`seterrobj(errobj)`|设置定义浮点错误处理的对象|
|`geterrobj()`|获取定义浮点错误处理的对象|

##	NumPy帮助

|Function|Desc|
|-----|-----|
|`lookfor(what[,module,import_modules])`|在文档中搜索关键词|
|`info([object,maxwidth,output,toplevel])`|获取帮助信息|
|`source(object[,output])`|获取源码|

