---
title: Systemd V.S. System V
categories:
  - Linux
  - Kernel
tags:
  - Linux
  - Kernel
  - Systemd
  - System V
date: 2021-08-31 10:16:19
updated: 2021-09-29 15:17:57
toc: true
mathjax: true
description: 
---

##	*System V*

-	*System V*：最初是 *Unix* 的分支，这里指 *System V* 上的系统服务管理风格
	-	由 `init` 守护进程根据运行级别 *run level* 管理系统启动

###	*System V* 运行级别

-	*System V* 的 7 个运行级别
	-	*LV 0*：停机状态
		-	设为默认运行级别设置时不能正常启动
	-	*LV 1*：单用户工作状态
		-	*root* 权限
		-	禁止远程登录
		-	用于系统维护
	-	*LV 2*：多用户状态
		-	无 *NFS* 支持
	-	*LV 3*：完整的多用户模式
		-	有 *NFS* 支持
		-	登录后进入控制台命令模式
		-	标准运行级别之一
	-	*LV 4*：系统未使用
		-	保留，不使用
	-	*LV 5*：*X11* 控制台
		-	登录后进入 *GUI* 模式
		-	标准运行级别之一
	-	*LV6*：系统正常关闭并重启
		-	设为默认运行级别设置时不能正常启动

###	*System V* 运行原理

-	*System V* 运行原理
	-	`init` 进程根据 `inittab` 确定当前运行级别
		-	`initdefault:` 行为默认运行级别
	-	`/etc/init.d` 下存储系统服务脚本
		-	服务脚本必须存放在此目录下才能被相关工具感知
	-	`rc<N>.d` 目录下文件为不同运行级别下需要启动、禁止的服务
		-	其中文件基本为 `/etc/init.d` 下脚本的符号链接
		-	`S` 开头文件：启动服务
		-	`K` 开头文件：禁止服务
		-	数字表示执行顺序

###	*System V* 基本工具

####	`chkconfig`

-	`chkconfig`：管理系统服务，查询系统服务运行级别信息
	-	不是立刻禁止、激活服务，仅改变符号链接

-	选项参数
	-	`--add <SERVICE-NAME>`：增加系统服务，即在 `/etc/rc<N>.d` 下设置符号链接
	-	`--del <SERVICE-NAME>`：删除系统服务
	-	`--list <SERVICE-NAME>`：列出系统服务信息，即列出 `/etc/init.d` 目录下脚本服务
	-	`--level <LEVEL> <SERVICE-NAME> [on/off/reset]`：设置服务在某运行级别启动、停止、重置

####	`service`

-	`service`：启动、停止、重启服务，即执行 *System V* 初始化脚本
	-	`$ service <SERVICE-NAME> [start/stop/restart]`：启动、停止、重启服务
	-	`$ service --status-all`：

####	`upate-rc.d`

-	`update-rc.d`：安装、删除服务，即 *System V* 风格初始化脚本链接
	-	`$ update-rc.d [-f] <SERVICE-NAME> remove`：删除服务
	-	`$ update-rc.d <SERVICE-NAME> [defaults/defaults-disabled]`：设置服务启动、禁用
	-	`$ update-rc.d <SERVICE-NAME> [disable|enable] [S|2|3|4|5]`：设置服务启动、禁用运行级别

##	*Systemd*

-	*Systemd*：替代 *System V* 风格的系统启动、管理的解决方案
	-	*Systemd* 可以管理所有系统资源，不同系统资源统称为 *Unit*

> - 启用 *Systemd* 启动时，`systemd` PID 为 1，掌管所有进程，否则其工具无法正常使用

###	*Unit* 类型

-	*Systemd* 中 *Unit* 分为 12 种
	-	*Service Unit*：系统服务
	-	*Target Unit*：多个 *Unit* 组成的组
	-	*Device Unit*：硬件设备
	-	*Mount Unit*：文件系统挂载点
	-	*Automount Unit*：自动挂载点
	-	*Path Unit*：文件、路径
	-	*Scope Unit*：非 *Systemd* 启动的外部进程
	-	*Slice Unit*：进程组
	-	*Snapshot Unit*：*Systemd* 快照
	-	*Socket Unit*：进程间通信 socket
	-	*Swap Unit*：swap 文件
	-	*Timer Unit*：定时器

####	*Unit* 配置文件

```cnf
 # 首个区块，定义 unit 的元数据、与其他 unit 关系
[Unit]
Description=
Documentation=
Requires=					# 依赖，未运行则启动失败
Wants=						# 配合，未运行也成功启动
BindsTo=					# 类似依赖，退出则退出
Before=						# 若需启动，则需在此 unit 之后启动
After=						# 若需启动，则需在此 unit 之前启动
Conflicts=					# 冲突，不能同时启动
Condition=					# 前置条件，否则不能运行
Assert=						# 前置条件，否则启动失败
 # 最后一个区块，定义如何启动、是否开机启动
[Install]
WantedBy=					# 值为一个或多个 Target，Unit 被 enable 后其符号链接在
							# `/etc/systemd/system/<Target>.wants` 中被创建
RequiredBy=					# 值是一个或多个 Target，Unit 被 enable 后期符号链接在
							# `/etc/systemd/system/<Target>.required` 中被创建
Alias=						# 可用于启动的别名，Unit 被 enable 后会在
							# `/etc/systemd/system` 创建名称为别名的符号链接
Also=						# Unit 被激活时，同时也会被激活的其他 Unit
```

-	*Unit* 配置文件：指示如何启动 unit
	-	`/usr/lib/systemd/system`：存放系统、应用默认 *Unit* 配置文件
		-	在此目录下才可被 *systemd* 感知
		-	配置文件后缀名指示 unit 类型，缺省为 `.service`
	-	`/etc/systemd/system`：*systemd* 默认读取配置文件的目录
		-	此目录中 *Unit* 配置文件开机启动时被读取
		-	可在此自定配置，避免应用更新覆盖默认配置

			```cnf
			# `<service_name>.service` 或 `<service_name>.service.d/.conf`
			.include /lib/systemd/system/<service_name>.service
			# customized changes
			```

	-	`/run/systemd/system`：进程运行时创动态创建的服务文件目录
		-	仅修改程序运行时参数才会修改

-	*Unit* 配置文件状态
	-	`enabled`：已建立启动链接
	-	`disable`：未建立启动链接
	-	`static`：配置文件中无 `[Install]` 部分，只能作为其他配置文件依赖
	-	`masked`：被禁止建立启动链接

####	*Service*

```cnf
 # 用于 Service 的配置，只有 Service 类型 Unit 有
[Service]
Type=						# 定义启动时进程的行为
							# simple：默认值，执行 `ExecStart` 指定的指令，启动主进程
							# forking：以 fork 方式从父进程创建子进程，之后父进程退出
							# oneshot：一次性进程，Systemd 回等待当前服务退出，再继续执行
							# dbus：通过 D-bus 启动
							# notify：当前服务启动完毕，会通知 Systemd 继续执行
							# idle：其他任务执行完毕才会继续往下执行
PIDFile=					# 存放PID文件路径
ExecStart=					# 启动当前服务的命令
ExecStartPre=				# 启动当前服务前执行的命令
ExecStartPost=				# 启动当前服务之后执行的命令
ExecReload=					# 重启当前服务时执行的命令
ExecStop=					# 停止当前服务时执行的命令
ExecStopPost=				# 停止当前服务之后执行的命令
RestartSec=					# 自动重启当前服务间隔
Restart=					# 重启当前服务的情况
							# always、on-success、on-failure、on-abnormal、on-abort、on-watchdog
TimeoutSec=					# 停止当前服务前等待描述
Environment=				# 指定环境变量
RemainAfterExit=			# 进程退出后是否认为服务仍处于激活
PrivateTmp=					# 是否给服务分配临时空间
KillSignal=SIGTERM
KillMode=mixed
Environment=				# 指定环境变量
```

-	`Type`：服务启动类型
	-	`simple`：默认，执行 `ExecStart` 启动主进程
		-	*Systemd* 认为服务立即启动
		-	适合服务在前台持续运行
		-	服务进程不会 fork
		-	若服务要启动其他服务，不应使用此类型启动

	-	`forking`：以 fork 方式从父进程创建子进程，然后父进程立刻退出
		-	父进程退出后 *Systemd* 才认为服务启动成功
		-	适合常规守护进程（服务在后台运行）
		-	启动同时需要指定 `PIDFile=`，以便 *Systemd* 能跟踪服务主进程

	-	`oneshot`：一次性进程
		-	*Systemd* 等待当前服务退出才继续执行
		-	适合只执行一项任务、随后立即退出的服务
		-	可能需要同时设置 `RemainAfterExit=yes`，使得 *Systemd* 在服务进程推出后仍然认为服务处于激活状态

	-	`notify`：同 `simple`
		-	但服务就绪后向 *Systemd* 发送信号

	-	`idle`：其他任务执行完毕才会开始服务

	-	`dbus`：通过 *D-Bus* 启动服务
		-	指定的 `BusName` 出现在 *D-Bus* 系统总线上时，*Systemd* 即认为服务就绪

####	*Target*

-	Target 类型 unit：包含很多相关 unit 的 unit 组
	-	启动某个 target 时和启动其中所有 unit

-	Target 作用类似 *System V* 风格中的运行级别
	-	但是多个 target 可以同时启动

	|传统运行级别|对应 Target|Target 链接目标|
	|-----|-----|-----|
	|*RunLevel 0*|`runlevel0.target`|`poweroff.target`|
	|*RunLevel 1*|`runlevel1.target`|`rescue.target`|
	|*RunLevel 2*|`runlevel2.target`|`multi-user.target`|
	|*RunLevel 3*|`runlevel3.target`|`multi-user.target`|
	|*RunLevel 4*|`runlevel4.target`|`multi-user.target`|
	|*RunLevel 5*|`runlevel5.target`|`graphical.target`|
	|*RunLevel 6*|`runlevel6.target`|`reboot.target`|

###	*Systemd* 相关命令

####	`systemctl`

-	`systemctl`：管理系统

-	系统启动命令参数
	-	`reboot`：重启系统
	-	`poweroff`：关闭系统
	-	`halt`：停止 CPU 工作
	-	`suspend`：挂起系统
	-	`hibernate`：系统冬眠
	-	`hybrid-sleep`：系统交互式休眠
	-	`rescue`：启动进入救援状态（单用户）

-	资源管理命令参数
	-	`list-units`：列出 unit
		-	缺省：列出正在运行的 unit
		-	`--all`：列出所有 unit，包括未找到配置、启动失败
		-	`--failed`：列出加载失败 unit
		-	`--state=<STATE>`：列出指定状态 unit
		-	`--type=<TYPE>`：列出指定类型 unit
	-	`status <UNIT>`：查看系统、各 unit 状态
		-	`is-active <UNIT>`：`UNIT` 是否正在运行
		-	`is-failed <UNIT>`：`UNIT` 是否启动失败
		-	`is-enabled <UNIT>`：`UNIT` 是否启动链接
	-	`list-dependencies <UNIT>`：列出 `UNIT` 依赖
		-	`--all`：展开 `.target` 类型依赖
	-	`get-default`：列出启动时默认的 target
	-	`set-default <TARGET>`：设计启动时默认 target

-	*Unit*（主要是系统服务 `.service`）控制
	-	`enable <UNIT>`：开机启动 `UNIT`，即在 `/etc/systemd/system` 建立指向 `/usr/lib/systemd/system` 中 *Unit* 配置文件的符号链接
	-	`disable <UNIT>`：撤销开机启动 `UNIT`，即删除 `/etc/systemd/system` 中 *Unit* 配置文件符号链接
	-	`start <UNIT>`：启动 `UNIT`
	-	`stop <UNIT>`：停止 `UNIT`
	-	`restart <UNIT>`：重启 `UNIT`
	-	`kill <UNIT>`：杀死 `UNIT` 所有子进程
	-	`mask <UNIT>`：禁用 `UNIT`
		-	无法通过 `start`、`restart` 启动 `UNIT`
		-	可以防止服务被其他服务间接启动
	-	`umask <UNIT>`：启用已禁用服务 `UNIT`
	-	`reload <UNIT>`：重新加载的 `UNIT` 的配置文件（重载之后仍需重启才生效）
	-	`daemon-reload <UNIT>`：重载所有配置
	-	`show <UNIT>`：显示 `UNIT` 所有底层参数
		-	`-p <OPTION>`：显示指定属性值
	-	`setproperty <UNIT> <OPTION=VALUE>`：设置 `UNIT` 指定属性值
	-	`isolate <TARGET>`：关闭前一个 target 中不属于当前 target 的所有进程

-	*Unit* 配置文件命令参数
	-	`list-unit-files`：列出配置文件
		-	`--type=<TYPE>`：列出指定类型 `TYPE` *Unit* 配置文件
	-	`cat <UNIT>`：查看配置文件内容
	-	`edit <UNIT>`：编辑配置文件

-	选项参数
	-	`-H <USER>@<ADDR>`：指定远程主机

####	`systemd-analyze`

-	`systemd-analyze`：查看启动耗时

-	命令参数
	-	缺省：查看启动耗时
	-	`blame`：查看每个服务的启动耗时
	-	`critical-chain <SERVICE-NAME>`：显示瀑布状启动过程流

####	`hostnamectl`

-	`hostnamectl`：查看、设置当前主机信息

-	命令参数
	-	缺省：查看当前主机信息
	-	`set-hostname <HOSTNAME>`：设置主机名

####	`localectl`

-	`localectl`：查看、设置本地化设置

-	命令参数
	-	`set-locale <OPTION=VALUE>`：设置本地化
	-	`set-keymap <OPTION>`

####	`timedatectl`

-	`timedatectl`：查看、设置时区

-	命令参数
	-	`list-timezones`：列出可用时区
	-	`set-timezone <ZONE>`：设置时区
	-	`set-time <FORMAT>`：设置日期、时间格式

####	`loginctl`

-	`loginctl`：查看登录用户信息

-	命令参数
	-	`list-sessions`：列出当前会话
	-	`list-users`：列出当前登录用户
	-	`show-user <USER-NAME>`：列出指定用户信息

####	`jounalctl`

-	`journalctl`：日志查看、管理
	-	`$ journalctl <OP> <EXE>`：查看指定脚本的日志，缺省所有日志

-	参数选项
	-	`-k`：查看内核日志
	-	`-b`：查看系统本次启动的日志
	-	`-l`
	-	`--since=<TEIMSTAMP>`：查看指定时间的日志
	-	`-n <NUM>`：查看最新 `NUM` 行日志，缺省 10 行
	-	`-f`：实时滚动最新日志
	-	`_PID=<PID>`：查看 `PID` 进程日志
	-	`_UID=<UID>`：查看 `UID` 用户日志
	-	`-u <UNIT>`：查看指定 `UNIT` 日志
	-	`-p <LEVEL>`：查看指定优先级及以上日志：`emerg`、`alert`、`crit`、`err`、`warning`、`notice`、`info`、`debug`
	-	`--no-pager`：标准正常输出，而非分页输出
	-	`-o <FORMAT>`：输出格式：`json`、`json-pretty`
	-	`--disk-usage`：显示日志占据的空间
	-	`--vacuum-size=<SIZE>`：指定日志文件占据的最大空间
	-	`--vacuum-time=<TIME>`：指定日志文件保存时限


