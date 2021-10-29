---
title: Shell 常用工具
categories:
  - Linux
  - Shell
tags:
  - Linux
  - Shell
  - Tool
  - Datetime
  - Network
  - Crond
date: 2019-07-31 21:10:52
updated: 2021-08-31 20:02:15
toc: true
mathjax: true
comments: true
description: Shell 常用工具
---

##	网络传输

###	`wget`、`curl`

-	`wget`：从指定 URL 下载文件
	-	`-b`：后台执行
	-	`-c`：继续执行上次任务
	-	`-r`：递归下载
	-	`-O`：下载文件重命名
	-	`-nc`：不覆盖同名文件
	-	`-nv`：不显示执行过程
	-	`-P`：指定下载路径
	-	`--no-check-certificate`：跳过证书检查过程

-	`curl`：综合文件传输工具

##	日期时间

###	`date`

```sh
date -d <time> "+<format>"
```

-	`date`：显示、设置系统日期时间
	-	`-d`：指定时间，缺省今天
	-	`+<format>`：指定输出格式
		-	`%Y-%m-%d %h-%M-%S`：年月日时（24时）分秒
		-	`%a/%A`：星期缩写、完整
		-	`%b/%B`：月份缩写、完整
		-	`%D`：`MM/DD/YY`
		-	`%F`：`YYYY-MM-DD`

##	定时任务

###	`crontab`

-	`crontab`：提交和管理用户需要周期性执行的任务
	-	`crontab` 服务为 `crond`

-	选项参数
	-	`-e`：编辑
	-	`-l`：显示
	-	`-r`：删除
	-	`-u <USER-NAME>`：指定用户名称


####	`crontab` 配置文件

-	用户配置文件
	-	`/etc/cron.deny`：禁止使用 `crontab` 命令用户
	-	`/etc/cron.allow`：允许使用 `crontab` 命令用户

-	`/var/spool/cron/<user_name>`：用户名标识的每个用户任务调度文件
	-	文件中每行 6 个字段设置 `cron` 调度内容
		-	前 5 个字段为执行调度时间分配
			-	5 个字段：分钟、小时（24）、日期（31）、月份（12）、星期
			-	`*`：每个当前时间
			-	`-`：时间区间
			-	`,`：时间列表分隔
			-	`<range>/<n>`：范围内每隔多久执行一次，范围为 `*` 可省
		-	最后字段为调度内容

		```cnf
		# 任务运行的环境变量
		SHELL=/bin/bash
		PATH=/sbin:/bin:/usr/sbin:/usr/bin
		# 任务执行信息通过电子邮件发送给用户
		MAILTO=""
		# 命令执行的主目录
		HOME=/


		# 调度命令：mm hh DD MM Week Command
		51 *  *  *  *  <command-to-run>
		```

	-	案例

		```sh
		3,15 8-11 */2 * * command
			# 每隔两天的8-11点第3、15分执行
		3,15 8-11 * * 1 command
			# 每个星期一的上午8-11点的第3、15分钟执行
		```

-	`/etc/crontab`：系统任务调度文件
	-	类似于用户周期任务文件，但多第 7 个用户名字段

###	`at`

-	`at`：在指定时间执行任务

-	案例

	```shell
	at now+2minutes/ now+5hours
		# 从现在开始计时
	at 5:30pm/ 17:30 [today]
		# 当前时间
	at 17:30 7.11.2018/ 17:30 11/7/2018/ 17:30 Nov 7
		# 指定日期时间
		# 输入命令之后，`<c-d>`退出编辑，任务保存
	```


##	命令总结

-	日期、时间
	-	`cal`：显示日历信息
	-	`date`：显示、设置系统日期时间
	-	`hwclock`：查看、设置硬件时钟
	-	`clockdiff`：主机直接测量时钟差
	-	`rdate`：通过网络获取时间
	-	`sleep`：暂停指定时间

-	数值计算
	-	`bc`：任意精度计算器
	-	`expr`：将表达式值打印到标准输出，注意转义

-	网络工具
	-	`wget`：从指定 URL 下载文件
	-	`curl`：综合文件传输工具

-	获取命令系统帮助
	-	`help`：查看 shell 内建命令帮助信息 **（*bash* 内建）**
	-	`man`：显示在线帮助手册（常用）
	-	`info`：info 格式的帮助文档

-	归档压缩
	-	`tar`：文件归档、压缩
	-	`gzip`：压缩、解压缩 *gzip* 文件
		-	`gunzip`：解压缩 *gzip* 文件
		-	`zcmp`：调用 `diff` 比较 *gzip* 压缩文件（*Shell* 脚本）
	-	`zip`：压缩zip文件
		-	`unzip`：解压缩zip文件
		-	`zcat`：查看zip压缩文件
		-	`zless`：查看zip压缩文件
		-	`zipinfo`：列出zip文件相关详细信息 
		-	`zipsplit`：拆分zip文件
		-	`zipgrep`：在zip压缩文件中搜索指定字符串、模式
	-	`rpm2cpio`：将 *rpm* 包转变为 *cpio* 格式文件

-	定时任务
	-	`atq`：列出用户等待执行的作业 
	-	`atrm`：删除用户等待执行的作业
	-	`watch`：定期执行程序
	-	`at`：设置在某个时间执行任务
	-	`crontab`：提交、管理周期性任务


