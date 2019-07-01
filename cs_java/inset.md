#	Java安装设置

##	Java概念

###	Java

java有两种理解方式

-	java语言：指java编程语言，都是整数版本，如：`Java5`、
	`Java8`等
-	java环境：指jre/jdk环境，版本一般从`1.x`开始，如：
	`Java1.5.x`

###	JVM（Java Virtual Machine）

运行java字节码（`.class`)的虚拟机

-	无法直接执行`.java`文件，需要编译为`.class`
-	java能够跨平台的关键，在所有平台上都有JVM的实现，可以
	直接执行编译后的`.class`文件

JVM有很多版本：

-	Google dalvik
-	ART
-	Oracle JRE 自带JVM

###	JRE（Java Runtime Environment）

-	JRE = JVM + `.class`库文件 + 其他支持文件
-	java运行环境，运行java编写的应用

###	JDK（Java Development Kits）

-	JDK = java开发工具 + JRE
-	包括：complier、debugger，用于开发java应用

####	OpenJDK

-	**JDK**的一个“发行版”，Java SE 7 JSR的一个开源实现
-	现在同Oracle JDK区别很小

####	Java JDK版本

Java常见的发行版本两个：OpenJDK和JDK

-	JDK（Sun JDK）：Sun公司发布，

-	OpenJDK：JDK的开源版本，其实也是Sun公司发布的Java版本，
	Sun被Oracle收购之后也称为Oracle OpenJDK

#####	授权协议不同

-	JDK有两个协议发布

	-	SCSL（Sun Community Source License）：允许商用

	-	JRL（Java Research License）：开放源码，但仅允许个人
		研究使用，随JDK6发布（2004）开始使用

-	OpenJDK：GPL V2协议发布，开放源码，允许商业使用


#####	源代码完整性

OpenJDK：采用GPL协议发布发布，JDK的部分源代码因产权问题无法
开放OpenJDK使用（最重要部分为JMX中的可选元件SNMP部分源码）

-	这些无法开放的源码可作为plugin供OpenJDK编译时使用

-	替换为功能相同的开源代码

	-	Icedtea为很多不完整的部分开发了相同功能的源码
		（OpenJDK6）

	-	字体栅格化引擎使用Free Type代替

代码完整度排序：Sun JDK > SCSL > JRL > OpenJDK

-	Sun JDK有少量代码是完全不开放的，在SCSL中也没有，但少
-	SCSL代码比JRL多一些`closed`目录中的代码
-	JRL比OpenJDK多一些受lisense影响而无法以GPLv2开放的

#####	功能

OpenJDK相较于JDK有些功能缺失

-	不包含部署功能，包括：Browser Plugin、Java Web Start、
	Java控制面板

-	不包含其他软件包，如：Rhino、Java DB、JAXP，并且可以
	分离的软件包也都尽量分离

但是从JDK7/OpenJDK7开始，两者实质差异非常小，相较于“OpenJDK
是不完整JDK”，更像是JDK7在OpenJDK7上带有一些value-add

#####	商标

OpenJDK不能使用Java商标，安装OpenJDK的机器上的，输入
`$ java --version`，输出`OpenJDK`

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

#	SBT

SBT：*simple build tools*，Scala世界的Maven

-	下载、解压、将`SBT_HOME/bin`添加进`$PATH`即可
-	SBT中包含的Scala可通过`$ scala console`交互式启动解释器

##	参数命令

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

##	项目结构

-	`src`：`src`中其他目录将被忽略
	-	`main`
		-	`scala`
		-	`java`
		-	`resource`
	-	`test`
		-	`scala`
		-	`java`
		-	`resource`

-	`project`：可以包含`.scala`文件，和`.sbt`文件共同构成
	完整构建定义
	-	`Build.scala`

-	`target`：包含构建出的文件：classes、jar、托管文件、
	caches、文档

-	`lib`：存放非托管依赖
	-	其中jar被添加进`CLASSPATH`

-	`build.sbt`：包含构建定义
	-	其中隐式导入有
		-	`sbt.Keys._`：包含内建key
		-	`sbt._`

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
dep_exp ::= <groupID> % <artifactID> % <revision>[% <configuraion>] [% "test" | % Test]

resolver_exp ::= <name> at <location>
```

-	`groupID`：
-	`acrtifactID`：工件名称
-	`revision`：版本号，无需是固定版本号
	-	`"latest.integration"`
	-	`"2.9.+"`
	-	`[1.0,)`
-	`"test"|Test`：仅在`Test`配置的classpath中出现
-	`name`：指定Maven仓库名称
-	`location`：服务器地址

####	依赖添加

```scala
// `sbt.Keys`中声明
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
// `sbt.Keys`中声明
val resolvers += settingKeys[Seq[Resolver]]("extra resolvers")
// 添加本地Maven仓库
resolvers += resolver_exp
resolvers += Resolver.mavenLocal
resolvers += "Loal Maven Repo" at "file://" + Path.userHome.absolutePath+"/.m2/repository"
```


