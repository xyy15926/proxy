#	系统级查看、设置

##	进程管理

###	`ps`

查看当前进程瞬时快照

###	`top`

显示当前正在运行进程（动态更新）

-	按照**使用内存**大小排序，可以用于查找内存使用情况

###	`pgrep`

按名称、属性查找进程

###	`pidof`

根据进程名查找正在运行的进程进程号

###	`kill`

终止进程

###	`killall`

按名称终止进程

###	`pkill`

按名称、属性终止进程

###	`timeout`

在指定时间后仍然运行则终止进程

###	`wait`

等待指定进程

###	`fuser`

显示使用指定文件、socket的进程

###	`pmap`

报告进程的内存映射

###	`lsof`

列出打开的文件

###	`chkconfig`

为系统服务更新、查询运行级别信息

##	作业

###	`&`

放在命令之后，命令后台执行

```shell
$ ./pso > pso.file 2>&1 &
	# 将`pso`放在后台运行，把终端输出（包括标准错误）
		# 重定向的到文件中
```

###	`nohup`

不挂起job，即使shell退出

```shell
$ nohup ./pso > pso.file 2>&1 &
	# 不挂起任务，输出重定向到文件
$ nohup -p PID
	# 不挂起某个进程
```

###	`jobs`

列出活动的作业

`-l`：返回任务编号、进程号

###	`bg`

恢复在后台暂停工作的作业

```shell
$ bg %n
	# 将编号为`n`的任务转后台运行
```

###	`fg`

将程序、命令放在前台执行

```shell
$ fg %n
	# 将编号为`n`的任务转前台运行
```

###	`setsid`

在一个新的会话中运行程序

```shell
$ setsid ./test.sh &`
	# 新会话中非中断执行程序，此时当前shell退出不会终止job
$ (./test.sh &)
	# 同`setsid`，用`()`括起，进程在subshell中执行
```

###	`disown

```shell
$ disown -h %job_id
	# *放逐*已经在后台运行的job，
	# 则即使当前shell退出，job也不会结束
```

###	`screen`

创建断开模式的虚拟终端

```bash
$ screen -dmS screen_test
	# 创建断开（守护进程）模式的虚拟终端screen_test
$ screen -list
	# 列出虚拟终端
$ screen -r screen_test
	# 重新连接screen_test，此时执行的任何命令都能达到nohup
	```

###	快捷键

-	`<c-z>`：挂起当前任务
-	`<c-c>`：结束当前任务

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

##	系统监控

###	`scar`

收集、报告、保存系统活动信息

###	`iostat`

报告CUP统计数据，设备、分区输入/输出信息

###	`iotop`

I/O监控

###	`mpstat`

报告CPU相关统计数据

###	`vmstat`

报告虚拟内存统计

###	`tload`

加载显示系统平均负载、指定tty终端平均负载

###	`time`

显示资源资源使用时间

###	`uptime`

显示系统已运行时间

###	`ipcs`

提供IPC设施信息

###	`ipcrm`

删除消息队列、信号量集、共享内存ID

###	`lslk`

列出本地锁

##	包管理

###	`apt`

-	`install`
-	`update`
-	`remove`
-	`autoremove`
-	`clean`

###	`rpm`

###	`yum`

####	说明

-	`yum`在不使用`yum makecache`手动检查源配置文件时，可能
	很长时间都不会更新cache，也就无法发现软件仓库的更新
	（当然，可能和仓库类型有关，使用ISO镜像作为仓库时如此，
	即使不挂载镜像，yum在执行其他操作也不会发现有个仓库无法
	连接）

###	`dpkg`

###	`zypper`

###	`pacman`

##	库、依赖

###	`ldconfig`

创建、查看动态链接库缓存

-	根据`/etc/ld.so.conf`文件中包含路径，搜索动态链接库文件
	，创建缓存文件`/etc/ld.so.cache`

-	默认包含`/lib`、`/lib64`、`/usr/lib`、`/usr/lib64`，
	优先级逐渐降低，且低于`/etc/ld.so.conf`中路径

####	参数

-	生成动态链接库缓存，并打印至标准输出
-	`-v/--verbose`：详细版
-	`-n`：仅扫描命令行指定目录
-	`-N`：不更新缓存
-	`-X`：不更新文件链接
-	`-p`：查看当前缓存中库文件
-	`-f CONF`：指定配置文件，默认`/etc/ld.so.conf`
-	`-C CACHE`：指定生成缓存文件
-	`-r ROOT`：指定执行根目录，默认`/`（调用`chroot`实现）
-	`-l`：专家模式，手动设置
-	`-f Format/--format=Format`：缓存文件格式
	-	`ld`：老格式
	-	`new`：新格式
	-	`compat`：兼容格式

###	`ldd`

查看程序所需共享库的bash脚本

-	通过设置一系列环境变量，如`LD_TRACE_LOADED_OBJECTS`、
	`LD_WARN`、`LD_BIND_NOW`、`LD_LIBRARY_VERSION`、
	`LD_VERBOSE`等

-	当`LD_TRACE_LOAD_OBJECTS`环境变量不空时，任何可执行程序
	运行时只显示模块的依赖，且程序不真正执行

-	实质上是通过`ld-linux.so`实现

	```c
	$ /lib/ld-linux.so* --list exe
	$ ldd exe
		// 二者等价
	```

	> - `ld-linux.so*`参见`cppc/func_lib.md`

####	参数

-	`-v`：详细模式
-	`-u`：打印未使用的直接依赖
-	`-d`：执行重定位，报告任何丢失对象
-	`-r`：执行数据、函数重定位，报告任何丢失的对象、函数

####	打印

-	第1列：程序动态链接库依赖
-	第2列：系统提供的与程序需要的库所对应的库
-	第3列：库加载的开始地址

> - 首行、尾行可能是两个由kernel向所有进程都注入的库

###	`strings`

查看系统*glibc*支持的版本

###	`objdump`

查看目标文件的动态符号引用表

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

##	环境

###	`export`

显示、设置环境变量，使其可在shell子系统中使用

-	设置环境变量直接`$ ENV=value`即可，但是此环境变量
	不能在子shell中使用，只有`$ export ENV`导出后才可
-	`-f`：变量名称为函数名称
-	`-n`：取消导出变量，原shell仍可用
-	`-p`：列出所有shell赋予的环境变量

##	字体

###	`fc-list`

```shell
$ fc-list :lang=zh
	# 查找系统已安装字体，参数表示中文字体
```
###	`fc-cache`

创建字体信息缓存文件

###	`mkfontdir/mkfontscale`

创建字体文件index

-	还不清楚这个字体index有啥用
-	`man`这两个命令作用差不多，只是后者是说创建scalable
	字体文件index
-	网上信息是说这两个命令配合`fc-cache`用于安装字体，
	但是好像没啥用，把字体文件复制到`/usr/share/fonts`
	（系统）或`~/.local/share/fonts`（用户），不用执行
	这两个命令，直接`fc-cache`就行

##	常用服务

-	`mysqld`/`mariadb`

-	`sshd`

-	`firewalld`
