---
title: Java安装设置
tags:
  - Java
categories:
  - Java
date: 2019-07-10 00:48:32
updated: 2021-09-03 18:40:59
toc: true
mathjax: true
comments: true
description: Java安装设置
---

##	*Java* 概念

-	*Java* 平台包括
	-	*Standard Edition* / *SE*：桌面、简单服务器应用平台
	-	*Enterprise Edition* / *EE*：在 *SE* 基础上添加企业级技术标准、模块
		-	包括 *JDBC*、*EJB* 等
		-	适合复杂服务器应用
		-	*Oracle* 已停止更新，类似一般模块
	-	*Micro Edition* / *ME*：手机等小型设备

-	*Java* 版本号包含小数、整数两种方式
	-	*Java* 平台版本：以 *SE* 平台为例
		-	*J2SE 1.<X>*：使用小数点后一位标识大版本，*J2SE 1.4*
		-	*Java SE <X>*：*Java SE 5* 后使用整数标识大版本
	-	*JDK* 版本：版本数同 *SE* 平台，但
		-	*JDK 1.9* 及之前：使用小数点后一位标识大版本
		-	*JDK 10* 及之后：使用整数标识大版本

> - *J2* / *Java 2* 曾经用于标识 *Java* 版本

###	*Java Virtual Machine*

-	*JVM*：运行java字节码（`.class`)的虚拟机
	-	无法直接执行`.java`文件，需要编译为`.class`
	-	java能够跨平台的关键，在所有平台上都有JVM的实现，可以
		直接执行编译后的`.class`文件

-	*JVM*版本
	-	*Google dalvik*
	-	*ART*
	-	*Oracle JRE* 自带 *JVM*

###	*Java Runtime Environment*

-	*JRE*：*Java* 运行环境，运行 *Java* 编写的应用
	-	*JRE* = *JVM* + `.class`库文件 + 其他支持文件

###	*Java Development Kits*

-	*JDK*：*Java* 开发工具集合
	-	*JDK* = *Java* 开发工具 + *JRE*
	-	包括 *complier*、*debugger*，用于开发 *Java* 应用

###	*Java* 发行版

-	*Java* 常见的发行版本两个：*OpenJDK* 和 *JDK*
	-	*JDK(Sun JDK)*：*Sun* 公司发布
	-	*OpenJDK*：*JDK* 的开源版本，其实也是Sun公司发布的Java版本，Sun被Oracle收购之后也称为Oracle OpenJDK
		-	*JDK* 的一个“发行版”，*Java SE 7 JSR* 的一个开源实现
		-	现在同 *Oracle JDK* 区别很小
##	配置

###	JAVA

```shell
export JAVA_HOME=/opt/jdk
	# 安装是jdk，所以目录直接由jdk就好
	# 或者保留原版本号，创建符号链接`java`执行原目录
	# 可自定以部分
export JRE_HOME=$JAVA_HOME/jre
export PATH=$PATH:$JAVA_HOME/bin
export CLASSPATH=$CLASSPATH:$JRE_HOME/lib/rt.jar:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar
	# jar包是java文件的集合，可以/需要看作是文件夹
	# java的`CLASSPATH`所以需要添加的是jar包
```

###	Scala

```shell
export SCALA_HOME=/opt/scala
export PATH=$PATH:$SCALA_HOME/bin:$SCALA_HOME/sbin
```

#	MAVEN



