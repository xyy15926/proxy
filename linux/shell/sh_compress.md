---
title: Linux 归档、压缩
categories:
  - Linux
  - Shell
tags:
  - Linux
  - Tool
  - Compress
  - Tar
date: 2021-07-29 21:46:01
updated: 2021-07-29 21:46:01
toc: true
mathjax: true
description: 
---

##	归档、压缩

###	`tar`

多个文件保存进行归档、压缩

###	`gzip`

压缩、解压缩gzip文件

###	`gunzip`

解压缩gzip文件

###	`zcmp`

调用diff比较gzip压缩文件

###	`unzip`

解压缩zip文件

###	`zip`

压缩zip文件

###	`zcat`

查看zip压缩文件

###	`zless`

查看zip压缩文件

###	`zipinfo`

列出zip文件相关详细信息

###	`zipsplit`

拆分zip文件

###	`zipgrep`

在zip压缩文件中搜索指定字符串、模式

###	`zmore`

查看gzip/zip/compress压缩文件


###	`rpm2cpio`

将rpm包转变为cpio格式文件，然后可以用cpio解压

```
$ rpm2cpio rpm_pkg | cpio -div
```

