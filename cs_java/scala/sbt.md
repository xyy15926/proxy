---
title: SBT
tags:
  - Java
  - Scala
categories:
  - Java
  - Scala
date: 2019-07-10 00:48:32
updated: 2019-07-10 00:48:32
toc: true
mathjax: true
comments: true
description: SBT
---

##	综述

SBT：*simple build tools*，Scala世界的Maven

-	下载、解压、将`SBT_HOME/bin`添加进`$PATH`即可
-	SBT中包含的Scala可通过`$ scala console`交互式启动解释器

###	参数命令

-	`clean`：删除所有构建文件
-	`compile`：编译源文件：`src/main/scala`、`src/main/java`
-	`test`：编译运行所有测试
-	`console`：进入包含所有编译文件、依赖的classpath的scala
	解释器
	-	`:q[uit]`：退出
-	`run <args>`：在和SBT所处同一虚拟机上执行项目main class
-	`package`：将`src/main/resources`、`src/main/scala`、
	`src/main/java`中编译出class打包为jar
-	`help <cmd>`：帮助
-	`reload`：重新加载构建定义：`build.sbt`、`project.scala`
	、`project.sbt`
-	`inspect <key>`：查看key描述
-	`show <key[:subkey]>`：查看key对应value执行结果
	-	例：`show compile:dependencyClasspath`


> - 以上命令可以在shell中作参数、在SBT命令行作命令
> - SBT命令行中历史记录查找同shell
> - `~`前缀执行命令表示监视变化，源文件改变时自动执行该命令

###	项目结构

-	`src`：`src`中其他目录将被忽略
	-	`main`：主源码目录
		-	`scala`
		-	`java`
		-	`resources`：存储资源，会被打进jar包
			-	分布式执行则每个节点可以访问全体资源，相当于
				broadcast
			-	访问时`resources`作为根目录，直接`/`列出相对
				`resource`路径
	-	`test`：测试源码目录
		-	`scala`
		-	`java`
		-	`resources`

-	`project`：可以包含`.scala`文件，和`.sbt`文件共同构成
	完整构建定义
	-	`Build.scala`
	-	`plugins.sbt`：添加sbt插件

-	`target`：包含构建出的文件：classes、jar、托管文件、
	caches、文档

-	`lib`：存放非托管依赖
	-	其中jar被添加进`CLASSPATH`环境变量

-	`build.sbt`：包含构建定义
	-	其中隐式导入有
		-	`sbt.Keys._`：包含内建key
		-	`sbt._`

###	SBT配置

```cnf
 # ~/.sbt/repositories
 # 默认依赖仓库设置

[repositories]
	local
	<maven_repo_name>: <repo_address>
	<ivy_repo_naem>: <repo_address>, <address_formation>

 # 地址格式可能如下
[orgnanization]/[module]/(scala_[scalaVersion]/)(sbt_[sbtVersion]/)[revision]/[type]s/[artifact](-[classifier]).[ext]

 # ali Maven2 repo
aliyun: https://maven.aliyun.com/nexus/content/groups/public/
```

> - 需要添加sbt启动参数`-Dsbt.override.build.repos=true`使
	覆盖默认生效

##	*Build Definition*

构建定义：定义在`build.sbt`中、包含项目构建信息

> - 多工程`.sbt`构建定义：可为代码中定义多个子项目，结构灵活
> - bare `.sbt`构建定义

-	构建定义可以包含位于`project`目录下的`.scala`下的文件，
	用于定义常用函数、值

###	多工程`.sbt`构建定义

```c
// 自定义`TaskKey`
lazy val hello = taskKey[Unit]("example task")
// 定义库ID
val derby = "org.apache.derby" % "derby" % "10.4.1.3"

// 创建名为`root`、位于当前目录的子项目
lazy val root = (project in file("."))
	// 在`.settings`方法中设置*setting expression*键值对
	.settings(
		name := "hello",
		hello := {prinln("hello")},
		libraryDependencies += derby
	)
```

-	构建定义拥有不可变的`Setting[T]`类型映射描述项目属性，
	是会影响sbt保存键值对的map的转换

	-	`T`：映射表中值类型
	-	Setting描述`.settings`是对映射表的转换，增加新键值、
		追加至已有值，转换后的映射成为sbt新映射

-	`build.sbt`文件中除设置外，可以包含`val`、`def`定义
	-	所有定义都在设置之前被计算，无论在文件中所处位置
	-	一般使用`lazy val`避免初始化顺序问题

###	Bare`.sbt`构建定义

```scala
name := "hello"
version := "1.0"
scalaVersion := "2.12.8"
library.Dependencies += "org.apache.derby" % "derby" % "10.4.1.3"
```

-	bare `.sbt`构建定义由`Setting[_]`表达式列表组成

###	Keys

> - `SettingKey[T]`：值仅在子项目载入时计算一次
> - `TaskKey[T]`：值每次都需要被计算，可能有副作用
> - `InputKey[T]`：值以命令行参数作为输入的任务

-	可以通过各自创建方法创建自定义key

	```scala
	// 给定value（返回）类型、键值对描述
	lazy val hello = taskKey[Unit]("task key demo")
	```

-	在sbt交互模式下，可以输入key名称获取、执行value
	-	setting key：获取、显示value
	-	task key：执行value，但不会显示执行结果，需要
		`show <task>`才会打印执行结果

> - Key可以视为**为项目定义的属性、trigger**
> - `taskiness`（每次执行）可以视为task key的属性

###	`sbt.Keys`内建Key

-	内建Key中泛型参数已经确定，定制需要满足类型要求

####	项目属性

-	`name`：项目名称，默认为项目变量名
-	`baseDirectory`：项目根目录
-	`sourceDirectories`：源码目录
	-	`compile:_`：编译时设置
-	`sourceDirectory`：源码上层目录？

####	依赖相关

-	`unmanageBase`：指定非托管依赖目录
-	`unmanagedJars`：列举`unmanagedBase`目录下所有jar
	的task key
-	`dependencyClasspath`：非托管依赖classpath
	-	`compile:_`：编译时设置
	-	`runtime:_`：运行时设置
-	`libraryDependecies`：指定依赖、设置classpath
	-	直接列出
	-	Maven POM文件
	-	Ivy配置文件
-	`resolvers`：指定额外解析器，Ivy搜索服务器指示
-	`externalResolvers`：组合`resolvers`、默认仓库的task key
	-	定制其以覆盖默认仓库

####	运行相关

-	`package`：打包系列Task
	-	类型：`TaskKey[File]`的task key
	-	返回值：生成的jar文件

-	`compile`：编译系列Task

###	`.sbt`特殊方法

> - 常用类型`String`等的方法仅在`.sbt`中可用
> - 方法的具体行为、返回值略有差异

####	`XXXXKey[T]`：

-	`:=`：给`Setting[T]`赋值、计算，并返回`Setting[T]`
	-	`SettingKey[T].:=`返回`Setting[T]`
	-	`TaskKey[T].:=`返回`Setting[Task[T]]`

-	`in`：获取其他Key的子Key

	```scala
	sourceDirectories in Compile += Seq(file("1"), file("2"))
	```

####	`SettingKey[T]`：

-	`+=`：**追加**单个元素至列表
-	`++=`：连接两个列表

####	`String`

-	`%`：从字符串构造Ivy `ModuleID`对象
-	`%%`：sbt会在actifact中加上项目的scala版本号
	-	也可手动添加版本号替代
	-	很多依赖会被编译给多个Scala版本，可以确保兼容性
-	`at`：创建`Resolver`对象

##	依赖

###	非托管依赖

非托管依赖：`lib/`目录下的jar文件

-	其中jar被添加进classpath
	-	对`compile`、`test`、`run`、`console`都成立
	-	可通过`dependencyClasspath`改变设置[某个]classpath

####	相关Key使用

```scala
// 定制非托管依赖目录
dependencyClasspath in Compile += <path>
dependencyClasspath in Runtime += <path>
// 定制非托管目录
unmanagedBase := baseDirectory.value / "custom_lib"
// 清空`compile`设置列表
unmanagedJars in Compile := Seq.empty[sbt.Attributed[java.io.File]]
```

###	托管依赖

托管依赖：由sbt根据`build.sbt`中设置自动维护依赖

-	使用*Apache Ivy*实现托管依赖
-	默认使用Maven2仓库

####	格式

```scala
dep_exp ::= <groupID> % <artifactID> % <revision>[% <configuraion>] [% "test" | % Test] [% "provided"]

resolver_exp ::= <name> at <location>
```

-	`groupID`：
-	`acrtifactID`：工件名称
-	`revision`：版本号，无需是固定版本号
	-	`"latest.integration"`
	-	`"2.9.+"`
	-	`[1.0,)`
-	可选选项
	-	`"test"|Test`：仅在`Test`配置的classpath中出现
	-	`"provided"`：由环境提供，assembly打包时将不打包该
		依赖
-	`name`：指定Maven仓库名称
-	`location`：服务器地址

####	依赖添加

```scala
// `sbt.Keys`中依赖声明
val libraryDependencies = settingKey[Seq[ModuleID]]("Delares managed dependencies")

// 添加单个依赖
libraryDependencies += dep_exp
// 添加多个依赖
libraryDependencies ++= Seq(
	dep_exp,
	dep_exp,
	<groupID> %% <artifactID> % <revision>
)
```

####	解析器添加

```scala
// `sbt.Keys`中解析器声明
val resolvers += settingKeys[Seq[Resolver]]("extra resolvers")

// 添加本地Maven仓库
resolvers += resolver_exp
resolvers += Resolver.mavenLocal
resolvers += "Loal Maven Repo" at "file://" + Path.userHome.absolutePath+"/.m2/repository"

// 将自定义解析器至于默认`externalResolvers`前
externalResolvers := Seq(
	resolver_exp
) ++ externalResolvers.values
```

> - `externalResolvers`中包含默认解析器，sbt会将此列表值拼接
	至`resolvers`之前，即仅修改`resolvers`仍然会有限使用默认
	解析器

##	其他配置

###	`project/plugins.sbt`

```scala
// assembly插件
// `assembly`：将依赖加入jar包，修改jar包配置文件
addSbtPlugin("com.eed3si9n" % "sbt-assembly" % "0.14.6")
```


