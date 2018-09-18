#	Shell命令

##	系统级查看、设置

###	进程、服务管理

-	`ps`：查看当前进程瞬时快照
-	`top`：显示当前正在运行进程
-	`pgrep`：按名称、属性查找进程
-	`pidof`：根据进程名查找正在运行的进程进程号
-	`kill`：终止进程
-	`killall`：按名称终止进程
-	`pkill`：按名称、属性终止进程
-	`timeout`：在指定时间后仍然运行则终止进程
-	`wait`：等待指定进程
-	`fuser`：显示使用指定文件、socket的进程
-	`nohup`：运行指定的命令不受挂起
-	`pmap`：报告进程的内存映射
-	`lsof`：列出打开的文件
-	`chkconfig`：为系统服务更新、查询运行级别信息
-	`bg`：恢复在后台暂停工作的作业
-	`fg`：将程序、命令放在前台执行
-	`jobs`：列出活动的作业

-	`systemctl`：替代service更加强大

	-	`start`：启动服务
	-	`stop`：关闭服务
	-	`enable`：开机自启动服务
	-	`disable`：关闭开机自启动服务

	`systemctl`通过`d-bus`和systemd交流，在docker和wsl中可能
	没有systemd-daemon，此命令可能不能使用，使用`service`
	代替

-	`service`：控制系统服务

###	性能监控

-	`scar`：收集、报告、保存系统活动信息
-	`iostat`：报告CUP统计数据，设备、分区输入/输出信息
-	`iotop`：I/O监控
-	`mpstat`：报告CPU相关统计数据
-	`vmstat`：报告虚拟内存统计
-	`tload`：加载显示系统平均负载、指定tty终端平均负载
-	`time`：显示资源资源使用时间
-	`uptime`：显示系统已运行时间
-	`ipcs`：提供IPC设施信息
-	`ipcrm`：删除消息队列、信号量集、共享内存ID
-	`lslk`：列出本地锁

###	任务计划

-	`crontab`：针对用户维护的crontab文件
-	`at`：在指定时间执行命令
-	`atq`：列出用户等待执行的作业
-	`atrm`：删除作业
-	`watch`：定期执行程序

###	字体

-	`fc-list :lang=zh`：查找系统已安装字体，参数表示中文字体
-	`fc-cache`：创建字体信息缓存文件
-	`mkfontdir/mkfontscale`：创建字体文件index
	-	还不清楚这个字体index有啥用
	-	`man`这两个命令作用差不多，只是后者是说创建scalable
		字体文件index
	-	网上信息是说这两个命令配合`fc-cache`用于安装字体，
		但是好像没啥用，把字体文件复制到`/usr/share/fonts`
		（系统）或`~/.local/share/fonts`（用户），不用执行
		这两个命令，直接`fc-cache`就行

###	环境变量

-	`export`：显示、设置环境变量，使其可在shell子系统中使用
	-	设置环境变量直接`$ ENV=value`即可，但是此环境变量
		不能在子shell中使用，只有`$ export ENV`导出后才可
	-	`-f`：变量名称为函数名称
	-	`-n`：取消导出变量，原shell仍可用
	-	`-p`：列出所有shell赋予的环境变量

##	文件、目录

###	目录、文件操作

-	`pwd`：显示当前工作目录绝对路径
-	`cd`：更改工作目录路径
-	`ls`：列出当前工作目录目录、文件信息
-	`dirs`：显示目录列表
-	`touch`：创建空文件或更改文件时间
-	`mkdir`：创建目录
-	`rmdir`：删除空目录
-	`cp`：复制文件和目录
-	`mv`：移动、重命名文件、目录
-	`rm`：删除文件、目录
-	`file`：查询文件的文件类型
-	`du`：显示目录、文件磁盘占用量（文件系统数据库情况）
-	`wc`：统计文件行数、单词数、字节数、字符数
-	`tree`：树状图逐级列出目录内容
-	`cksum`：显示文件CRC校验值、字节统计
-	`mk5sum`：显示、检查MD5（128bit）校验和
-	`sum`：为文件输出校验和及块计数
-	`dirname`：输出给出参数字符串中的目录名（不包括尾部`/`）
	，如果参数中不带`/`输出`.`表示当前目录
-	`basename`：输出给出参数字符串中的文件、**目录**名
-	`ln`：创建链接文件
-	`stat`：显示文件、文件系统状态

###	文件、目录权限、属性

-	`chmod`：更改文件、目录模式
-	`chown`：更改文件、目录的用户所有者、组群所有者
-	`chgrp`：更改文件、目录所属组
-	`umask`：显示、设置文件、目录创建默认权限掩码
-	`getfacl`：显示文件、目录ACL
-	`setfacl`：设置文件、目录ACL
-	`chacl`：更改文件、目录ACL
-	`lsattr`：查看文件、目录属性
-	`chattr`：更改文件、目录属性

###	显示文本文件

-	`cat`：显示文本文件内容
-	`more`：分页显示文本文件
-	`less`：回卷显示文本文件内容
-	`head`：显示文件指定前若干行
-	`tail`：实现文件指定后若干行
-	`nl`：显示文件行号、内容

###	文件处理

-	`sort`：对文件中数据排序
-	`uniq`：删除文件中重复行
-	`cut`：从文件的每行中输出之一的字节、字符、字段
-	`diff`：逐行比较两个文本文件
-	`diff3`：逐行比较三个文件
-	`cmp`：按字节比较两个文件
-	`tr`：从标准输入中替换、缩减、删除字符
-	`split`：将输入文件分割成固定大小的块
-	`tee`：将标准输入复制到指定温婉
-	`awk`：模式扫描和处理语言
	#todo
-	`sed`：过滤、转换文本的流编辑器

###	查找字符串、文件

-	`grep`：查找符合条件的字符串
-	`egrep`：在每个文件或标准输入中查找模式
-	`find`：列出文件系统内符合条件的文件
-	`whereis`：插卡指定文件、命令、手册页位置
-	`whatis`：在whatis数据库中搜索特定命令
-	`which`：显示可执行命令路径

###	文本编辑器

-	`vi`
-	`nano`：系统自带编辑器，有时只能编辑少部分配置文件

##	常用工具

###	获取命令系统帮助

-	`help`：重看内部shell命令帮助信息（常用）
-	`man`：显示在线帮助手册（常用）
-	`info`：info格式的帮助文档
5.	`ldd`：列出二进制文件共享库依赖

###	日期、时间

-	`cal`：显示日历信息
-	`date`：显示、设置系统日期时间
-	`hwclock`：查看、设置硬件时钟
-	`clockdiff`：主机直接测量时钟差
-	`rdate`：通过网络获取时间
-	`sleep`：暂停指定时间

###	数值计算

-	`bc`：任意精度计算器
-	`expr`：将表达式值打印到标准输出，注意转义

###	归档、压缩

-	`tar`：多个文件保存进行归档、压缩
-	`gzip`：压缩、解压缩gzip文件
-	`gunzip`：解压缩gzip文件
-	`zcmp`：调用diff比较gzip压缩文件
-	`unzip`：解压缩zip文件
-	`zip`：压缩zip文件
-	`zcat`：查看zip压缩文件
-	`zless`：查看zip压缩文件
-	`zipinfo`：列出zip文件相关详细信息
-	`zipsplit`：拆分zip文件
-	`zipgrep`：在zip压缩文件中搜索指定字符串、模式
-	`zmore`：查看gzip/zip/compress压缩文件

-	`rpm2cpio`：将rpm包转变为cpio格式文件，然后可以用cpio解压
	`rpm2cpio rpm_pkg | cpio -div`

###	远程连接服务器

-	`ssh`：ssh登陆服务器
	-	`ssh user_name@host`
	-	配置文件格式
		```conf
		Host link_name
			HostName host
			User user_name
			Port 22
			IdentityFile private_key
		```

-	`scp`：secure cp，安全传输（cp）文件
	-	本机到远程
		`scp /path/to/file user_name@host:/path/to/dest`
	-	远程到本机
		`scp user_name@host:/path/to/file /path/to/dest`
	-	这个命令应该在本机上使用，不是ssh环境下
		-	ssh环境下使用命令表示在远程主机上操作 
		-	而本机host一般是未知的（不是localhost）

-	`ssh-keygen`：生成ssh需要的rsa密钥

-	`ssh-add`：添加密钥给`ssh-agent`（避免每次输入密码？）


##	用户、登陆

###	显示登陆用户

-	`w`：详细查询已登录当前计算机用户
-	`who`：显示已登录当前计算机用户简单信息
-	`logname`：显示当前用户登陆名称
-	`users`：用单独一行显示当前登陆用户
-	`last`：显示近期用户登陆情况
-	`lasttb`：列出登陆系统失败用户信息
-	`lastlog`：查看用户上次登陆信息

###	用户、用户组

-	`useradd`：创建用户账户
-	`adduser`：`useradd`命令的符号链接
-	`newusers`：更新、批量创建新用户
-	`lnewusers`：从标准输入中读取数据创建用户
-	`usermod`：修改用户账户树形
-	`userdel`：删除用户账户
-	`groupadd`：创建用户组
-	`groupmod`：修改用户组
-	`groupdel`：删除用户组
-	`passwd`：设置、修改用户密码
-	`chpassws`：成批修改用户口令
-	`change`：更改用户密码到期信息
-	`chsh`：更改用户账户shell类型
-	`pwck`：校验`/etc/passwd`和`/etc/shadow`文件是否合法、
	完整
-	`grpck`：验证用户组文件`/etc/grous/`和`/etc/gshadow`
	完整性
-	`newgrp`：将用户账户以另一个组群身份进行登陆
-	`finger`：用户信息查找
-	`groups`：显示指定用户的组群成员身份
-	`id`：显示用户uid及用户所属组群gid
-	`su`：切换值其他用户账户登陆
-	`sudo`：以superuser用户执行命令

###	登陆、退出、关机、重启

-	`login`：登陆系统
-	`logout`：退出shell
-	`exit`：退出shell（常用）
-	`rlogin`：远程登陆服务器
-	`poweroff`：关闭系统，并将关闭记录写入`/var/log/wtmp`
	日志文件
-	`ctrlaltdel`：强制或安全重启服务器
-	`shutdown`：关闭系统
-	`halt`：关闭系统
-	`reboot`：重启系统
-	`init 0/6`：关机/重启

