#	Terminal、CLI、Shell、TTY

##	*Terminal*

终端：计算机领域的终端是指用于**与计算机进行交互的输入输出
设备**，本身不提供运算处理功能

###	早期终端

连接到计算上的带输入、输出功能的外设

-	普通终端：终端机（实体）
	-	电传打字机：键盘输入指令、纸带打印信号

-	*Console*：控制台，和计算机一体的特殊终端，用于管理主机，
	比普通终端权限更高

-	说明：
	-	一台主机一般只有一个控制台，但可以连接多个普通终端
	-	Console、Terminal随着PC普及已经基本时同义词

###	分类

-	*Character/Text Terminal*：字符终端，只能接受、显示文本
	信息的终端

	-	*Dumb Terminal*：哑终端
	-	*Intelligent Terminal*：智能终端，相较于哑终端
		-	理解转义序列
		-	定位光标、显示位置

-	*Graphical Terminal*：图形终端，可以显示图形、图像
	-	现在专门图形终端已经少见，基本被**全功能显示器**取代

###	Terminal Emulator

终端模拟器：默认传统终端行为的程序，用于与传统的、不兼容图形
接口命令行程序（如：GNU工具集）交互

-	对CLI程序，终端模拟器假装为传统终端设备

-	对现代图形接口，终端模拟器假装为GUI程序

	-	捕获键盘输入
	-	将输入发送给CLI程序（bash）
	-	得到命令行程序输出结果
	-	调用图形接口（如：X11），将输出结果渲染至显示器

> - Linux系统中，`<C-A—F1..6>`组合键切换出的的全屏终端界面
	也是终端模拟器，只是不运行在图形界面中、由内核直接提供，
	也称虚拟控制台

###	其他概念

####	*CLI*

*Command-Line Interface*：命令行界面，图形用户界面普及前使用
最广泛的用户界面，通常不支持鼠标，通过键盘输入指令

####	*TTY*

TTY是teletype（电传打字机）的简称

-	UNIX为支持电传打字机设计了**名为tty的子系统**，后面tty
	名称被保留成为终端的统称（虽然终端设备不局限于tty）

-	具体终端硬件设备抽象为操作系统内部`/dev/tty*`设备文件
	-	`/dev/tty1..6`即对应6个虚拟控制台

-	早计算机上*Serial Port*最大用途就是连接终端设备，所以
	Unix会把**串行端口上设备同样抽象为tty设备**
	-	`/dev/ttyS*`对应串口设备

> - [Linux TTY/PTS概述](https://segmentfault.com/a/1190000009082089)

####	*Shell*

Shell：提供**用户界面**的程序，接受用户输入命令和内核沟通

![pc_architecture](imgs/pc_architecture.png)

-	避免普通用户随意操作导致系统崩溃

-	但是Shell提供用户操作系统入口，一般通过是通过Shell调用
	其他应用程序，其他应用程序再调用系统调用，虽然不是Shell
	直接于内核交互，但是广义上可以认为是Shell提供了
	**与内核交互的**用户界面

#####	分类

-	命令行Shell：提供CLI
	-	*sh*：*Bourne Shell*
	-	*bash*：*Bourne-Again Shell*
	-	*zhs*：*Z Shell*
	-	*fish*：*Friendly Interactive Shell*
	-	*cmd.exe*：这个应该看作是Shell，而不是terminal，考虑
		其和内核进行交互，其可以接受键盘输入应该是其宿主的
		功能（即宿主作为隐式terminal emulator）
	-	*PowerShell*

-	图形Shell：提供GUI
	-	Windows下的*explorer.exe*

#####	说明

-	大部分情况下终端将用户的字符以外键盘输入转为控制序列，
	但是终端接受到`<C-C>`等特殊组合键时，发送特殊信号

	> -	`$ stty -a`查看当前终端设置

-	除非被重定向，否则Shell不会知道命令执行结果

-	Shell相关
	-	命令提示符Prompt
	-	行编辑、输入历史、自动补全（但是也有些终端自己实现
		此功能）

-	Terminal相关
	-	上、下翻页查看内容
	-	终端中复制、粘贴功能





