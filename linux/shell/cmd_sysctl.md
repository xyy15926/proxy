#	系统级查看、设置

##	进程管理

###	`ps`

查看当前进程瞬时快照

###	`top`

显示当前正在运行进程

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

###	`dpkg`

###	`zypper`

###	`pacman`

##	服务、环境

###	`systemctl`

-	替代service更加强大

-	`systemctl`通过`d-bus`和systemd交流，在docker和wsl中可能
	没有systemd-daemon，此命令可能不能使用，使用`service`
	代替

####	动作

-	`start`：启动服务

-	`stop`：关闭服务

-	`enable`：开机自启动服务

-	`disable`：关闭开机自启动服务
	-	`enable`启动后的服务仅仅`disable`不会立刻停止服务

###	`service`

控制系统服务

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
