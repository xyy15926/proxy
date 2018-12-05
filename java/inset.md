#	Java安装设置

##	Java名词

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

###	OpenJDK

-	**JDK**的一个“发行版”，Java SE 7 JSR的一个开源实现
-	现在同Oracle JDK区别很小

##	Java JDK版本

Java常见的发行版本两个：OpenJDK和JDK

-	JDK（Sun JDK）：Sun公司发布，

-	OpenJDK：JDK的开源版本，其实也是Sun公司发布的Java版本，
	Sun被Oracle收购之后也称为Oracle OpenJDK

###	授权协议不同

-	JDK有两个协议发布

	-	SCSL（Sun Community Source License）：允许商用

	-	JRL（Java Research License）：开放源码，但仅允许个人
		研究使用，随JDK6发布（2004）开始使用

-	OpenJDK：GPL V2协议发布，开放源码，允许商业使用


###	源代码完整性

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

###	功能

OpenJDK相较于JDK有些功能缺失

-	不包含部署功能，包括：Browser Plugin、Java Web Start、
	Java控制面板

-	不包含其他软件包，如：Rhino、Java DB、JAXP，并且可以
	分离的软件包也都尽量分离

但是从JDK7/OpenJDK7开始，两者实质差异非常小，相较于“OpenJDK
是不完整JDK”，更像是JDK7在OpenJDK7上带有一些value-add

###	商标

OpenJDK不能使用Java商标，安装OpenJDK的机器上的，输入
`$ java --version`，输出`OpenJDK`


###	配置

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

##	Scala

###	配置

```shell
export SCALA_HOME=/opt/scala
export PATH=$PATH:$SCALA_HOME/bin:$SCALA_HOME/sbin
```



