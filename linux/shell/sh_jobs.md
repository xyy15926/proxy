---
title: Shell 任务
categories:
  - Linux
  - Shell
tags:
  - Linux
  - Shell
  - Crontab
  - Systemd
date: 2021-07-29 21:30:25
updated: 2021-07-29 21:51:23
toc: true
mathjax: true
description: 
---

##	定时任务

###	`atq`

列出用户等待执行的作业

###	`atrm`

删除用户等待执行的作业

###	`watch`

定期执行程序

###	`at`

设置在某个时间执行任务

```shell
at now+2minutes/ now+5hours
	# 从现在开始计时
at 5:30pm/ 17:30 [today]
	# 当前时间
at 17:30 7.11.2018/ 17:30 11/7/2018/ 17:30 Nov 7
	# 指定日期时间
	# 输入命令之后，`<c-d>`退出编辑，任务保存
```

###	`crontab`

针对用户维护的`/var/spool/cron/crontabs/user_name`文件，其中
保存了cron调度的内容（根据用户名标识）

-	`-e`：编辑
-	`-l`：显示
-	`-r`：删除

任务计划格式见文件部分


##	任务、作业

###	Crontab文件

####	`/var/spool/cron/crontabs/user_name`

使用用户名标识的每个用户的定时任务计划文件，文件中每行为一个
cron调度内容（也可以使用`crontab -e`直接编辑）

-	共6个字段，前5个字段为执行调度时间分配
	-	分钟、小时（24）、日期（31）、月份（12）、星期
	-	`*`表示每个当前时间
	-	`-`时间区间
	-	`,`时间分隔
	-	`[range]/[n]`：范围内每隔多久执行一次，范围为`*`可省

-	最后字段为调度内容

-	例

	```shell
	3,15 8-11 */2 * * command
		# 每隔两天的8-11点第3、15分执行
	3,15 8-11 * * 1 command
		# 每个星期一的上午8-11点的第3、15分钟执行
	```

####	`/etc/crontabs`

系统级crontab文件，类似于用户crontab文件，只是多一个用户名
字段位于第6个

##	服务

###	系统服务

-	服务`systemctl`脚本目录优先级从低到高

	-	`[/usr]/lib/systemd/system`：系统、应用默认存放位置
		，随系统、应用更新会改变

	-	`/etc/systemd/system`：**推荐在此自定义配置**，避免
		被更新覆盖

		```cnf
		.include /lib/systemd/system/<service_name>.service
		# customized changes
		```

	-	`/run/systemd/system`：进程运行时创动态创建服务文件
		目录，仅修改程序运行时参数才会修改

-	自定义系统服务方法

	-	创建`<service_name>.service`文件，其中写入配置
	-	创建`<service_name>.service.d`目录，其中新建`.conf`
		文件写入配置

> - 系统服务控制命令`systemctl`参见*linux/shell/cmd_sysctl*

####	`.service`

`<some_name>.service`：服务配置unit文件

```cnf
[Unit]					# 包含对服务的说明
Description=			# 服务简单描述
Documentation=			# 服务文档
Requires=				# 依赖unit，未启动则当前unit启动失败
Wants=					# 配合unit，不影响当前unit启动
BindsTo=				# 类似`Requires`，若其退出则当前unit
							# 退出
Before=					# 若该字段指定unit要启动，则必须在
							# 当前unit之后启动
After=					# 若该字段指定unit要启动，则必须在
							# 当前unit之前启动
Conflicts=				# 冲突unit
Condition=				# 条件，否则无法启动
Assert=					# 必须条件，否则启动失败


[Service]				# 服务具体运行参数设置，仅
							# *service unit*有该字段
Type=					# 服务启动类型
PIDFile=				# 存放PID文件路径
RemainAfterExit=		# 进程退出后是否认为服务仍处于激活

ExecStartPre=			# 服务启动前执行的命令
ExecStartPost=			# 服务启动后执行的命令
ExecStart=				# 服务具体执行、重启、停止命令
ExecReload=
ExecStop=

Restart=				# 重启当前服务的情况
RestartSec=				# 自动重启当前服务间隔秒数
TimeoutSec=				# 停止当前服务之前等待秒数

PrivateTmp=				# 是否给服务分配临时空间
KillSignal=SIGTERM
KillMode=mixed

Environmen=				# 指定环境变量

[Install]
WantedBy=				# unit所属的target，`enable`至相应
							# target
RequiredBy=				# unit被需求的target
Alias=					# unit启动别名
Also=					# `enable`当前unit时，同时被`enable`
							# 的unit
```

-	`Type`：服务启动类型

	-	`simple`：默认，执行`ExecStart`启动主进程
		-	Systemd认为服务立即启动
		-	适合服务在前台持续运行
		-	服务进程不会fork
		-	若服务要启动其他服务，不应应该使用此类型启动，

	-	`forking`：以fork方式从父进程创建子进程，然后父进程
		立刻退出
		-	父进程退出后Systemd才认为服务启动成功
		-	适合常规守护进程（服务在后台运行）
		-	启动同时需要指定`PIDFile=`，以便systemd能跟踪
			服务主进程

	-	`oneshot`：一次性进程
		-	Systemd等待当前服务退出才继续执行
		-	适合只执行一项任务、随后立即退出的服务
		-	可能需要同时设置`RemainAfterExit=yes`，使得
			systemd在服务进程推出后仍然认为服务处于激活状态

	-	`notify`：同`simple`
		-	但服务就绪后向Systemd发送信号

	-	`idle`：其他任务执行完毕才会开始服务

	-	`dbus`：通过D-Bus启动服务
		-	指定的`BusName`出现在DBus系统总线上时，Systemd
			即认为服务就绪

-	`WantedBy`：服务安装的用户模式，即希望使用服务的用户
	-	`multi-user.target`：允许多用户使用服务

####	`.target`

`<some_name>.target`：可以理解为系统的“状态点”

-	target通常包含多个unit
	-	即包含需要启动的服务的组
	-	方便启动一组unit

-	启动target即意味将系统置于某个状态点

-	target可以和传统init启动模式中RunLevel相对应
	-	但RunLevel互斥，不能同时启动
	
	|RunLevel|Target|
	|-----|-----|
	|0|`runlevel0.target`或`poweroff.target`|
	|1|`runlevel1.target`或`rescue.target`|
	|2|`runlevel2.target`或`multi-user.target`|
	|3|`runlevel3.target`或`multi-user.target`|
	|4|`runlevel4.target`或`multi-user.target`|
	|5|`runlevel5.target`或`graphical.target`|
	|6|`runlevel6.target`或`reboot.target`|

-	`enable`设置某个服务自启动，就是在服务配置`.service`中
	`WantedBy`、`RequiredBy`指明的target注册

	-	`WantedBy`向target对应目录中
		`/etc/systemd/system/<target_name>.target.wants`
		添加符号链接
	-	`RequiredBy`向target对应目录中
		`/etc/systemd/system/<target_name>.target.required`
		添加符号链接
	-	`systemctl disable <unit>`就是移除对应symlink

####	`.socket`

###	日志

####	`/etc/systemd/journal.conf`

Systemd日志配置文件

##	Systemd（服务）

-	Systemd作为新系统管理器，替代service更加强大
	-	支持服务并行启动，提高效率
	-	具有日志管理、快照备份和恢复、挂载点管理
	-	包括一组命令

-	Systemd可以管理所有系统资源，不同资源统称为*Unit*
	-	*service unit*：系统服务
	-	*target unit*：多个unit构成的组
	-	*devie unit*：硬件设备
	-	*mount unit*：文件系统挂载点
	-	*automount unit*：自动挂载点
	-	*path unit*：文件、路径
	-	*scope unit*：不是由Systemd启动外部进程
	-	*slice unit*：进程组
	-	*snapshot unit*：Systemd快照，可以切换某个快照
	-	*socket unit*：进程间通信socket
	-	*swap unit*：swap文件
	-	*timer unit*：定时器

-	unit**配置文件状态**（无法反映unit是否运行）
	-	`enabled`：已建立启动链接
	-	`diabled`：未建立启动链接
	-	`static`：配置文件没有`[Install]`部分，只能作为
		其他unit配置文件依赖，不能独立注册
	-	`masked`：被禁止建立启动链接

> - `systemd`进程ID为1，掌管所有进程
> - *unit*配置文件参见*linux/shell/config_files*

###	`systemctl`

> - `systemctl`通过`d-bus`和systemd交流，在docker和wsl中可能
	没有systemd-daemon，此命令可能不能使用，使用`service`
	代替

####	查看服务、系统状态

-	`status <unit>`：查看系统、unit（服务）状态
	-	`-H <host>`：查看远程主机unit状态
-	`show <unit>`：查看unit底层参数
	-	`-p <attr>`：查看某个具体属性参数
-	三个查询状态的简单方法，方便脚本使用
	-	`is-active <serivce>`
	-	`is-failed <serivce>`
	-	`is-enabled <serivce>`
-	`reset-failed <unit>`：清除错误状态

-	`list-units`：列出所有已加载unit
-	`list-unit-files`：列出所有系统中unit配置文件、状态
	-	`--type=`：指定特定类型文件
	-	`--status=`：指定特定状态文件
-	`list-dependencies <unit>`：查看依赖关系
	-	`-all`：展开target类型

####	状态点/启动级别

-	`get-default`：查看默认target
-	`set-default <target>`：设置默认target
-	`isolate <target>`：切换target，其他target的unit将
	被停止

####	设置服务、系统

-	`start <service>`：启动服务
-	`stop <service>`：关闭服务
-	`enable <service>`：开机自启动服务

	> - 行为参见*linux/shell/config_files*

-	`disable <service>`：关闭开机自启动服务
	-	`enable`启动后的服务仅仅`disable`不会立刻停止服务
-	`kill <service>`：杀死服务的所有子进程
-	`mask <service>`：禁用服务
	-	无法通过`start`、`restart`启动服务
	-	可以防止服务被其他服务间接启动
-	`umask <service>`：启用已禁用服务
-	`daemon-reload`：重新读取所有服务项
	-	修改、删除、添加服务项后执行该命令
-	`poweroff`：关机
-	`reboot`：重启
-	`rescue`：进入rescue模式

####	配置文件

-	`cat <service_name>.service`：查看Unit定义文件
-	`edit <service_name>.service`：编辑Unit定义文件
-	`reload <service_name>.service`：重新加载Unit定义文件

###	`journalctl`

`journalctl`：统一管理所有日志：内核日志、应用日志

-	`-k`：仅内核日志
-	`--since`：指定日志时间
-	`-n`：指定查看日志行数
-	`-f`：最新日志
-	`_PID=`：进程日志
-	`_UID=`：用户日志
-	`-u`：unit日志
-	`-p`：日志有限集
	-	`emerg`：0
	-	`alert`：1
	-	`crit`：2
	-	`err`：3
	-	`warning`：4
	-	`notice`：5
	-	`info`：6
	-	`debug`：7
-	`--nopager`：非分页输出
-	`-o`：日志输出格式
	-	`json`
	-	`json-pretty`
-	`--disk-usage`：日志占用空间
-	`--vacuum-size=`：指定日志占用最大空间
-	`--vacuum-time=`：指定日志保持时间

###	`systemd-analyze`

`systemd-analyze`：查看系统启动耗时

-	`blame`：查看各项服务启动耗时
-	`critical-chain`：瀑布状启动过程流

###	`hostnamectl`

`hostnamectl`：查看主机当前信息

-	`set-hostname <hostname>`：设置主机名称

###	`localectl`

`localctl`：查看本地化设置

-	`set-locale LANG=en_GB.utf8`
-	`set-keymap en_GB`

###	`timedatectl`

`timedatectl`：查看当前时区设置

-	`list-timezones`：显示所有可用时区
-	`set-timezone America/New_York`
-	`set-time YYYY-MM-DD`
-	`set-time HH:MM:SS`

###	`loginctl`

`loginctl`：查看当前登陆用户

-	`list-sessions`：列出当前session
-	`list-users`：列出当前登陆用户
-	`show-user <user>`：显示用户信息

###	`service`

控制系统服务


