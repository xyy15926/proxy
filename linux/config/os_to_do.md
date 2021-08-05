---
title: Linux 安装后常用配置
categories:
  - Linux
  - Configuration
tags:
  - Linux
  - CentOS
  - Repository
  - Configuration
date: 2019-07-31 21:10:51
updated: 2021-07-21 16:08:47
toc: true
mathjax: true
comments: true
description: Linux 安装后常用配置
---

##	用户设置

###	设置 `root` 密码

-	*Linux* 安装之初，在设置 `root` 密码之前无法使用 `$ su` 切换到 `root` 用户，需要先设置root用户密码

	```shell
	$ sudo passwd root
	```

##	应用设置

###	*Debian*

####	配置文件

> - *Debian* 源配置文件：`/etc/apt/source.list`
> - 修改完成后运行 `$ sudo apt update` 更新索引

-	163 源：<https://mirrors.163.com/.help/debian.html>

	```cnf
	deb http://mirrors.163.com/debian/ <VERSION> main non-free contrib
	deb http://mirrors.163.com/debian/ <VERSION>-updates main non-free contrib
	deb http://mirrors.163.com/debian/ <VERSION>-backports main non-free contrib
	deb-src http://mirrors.163.com/debian/ <VERSION> main non-free contrib
	deb-src http://mirrors.163.com/debian/ <VERSION>-updates main non-free contrib
	deb-src http://mirrors.163.com/debian/ <VERSION>-backports main non-free contrib
	deb http://mirrors.163.com/debian-security/ <VERSION>/updates main non-free contrib
	deb-src http://mirrors.163.com/debian-security/ <VERSION>/updates main non-free contrib
	```

-	*USTC* 源：<https://mirrors.ustc.edu.cn/help/debian.html>

	```cnf
	deb http://mirrors.ustc.edu.cn/debian/ <VERSION> main contrib non-free
	deb-src http://mirrors.ustc.edu.cn/debian/ <VERSION> main contrib non-free
	deb http://mirrors.ustc.edu.cn/debian/ <VERSION>-updates main contrib non-free
	deb-src http://mirrors.ustc.edu.cn/debian/ <VERSION>-updates main contrib non-free
	deb http://mirrors.ustc.edu.cn/debian/ <VERSION>-backports main contrib non-free
	deb-src http://mirrors.ustc.edu.cn/debian/ <VERSION>-backports main contrib non-free
	deb http://mirrors.ustc.edu.cn/debian-security/ <VERSION>/updates main contrib non-free
	deb-src http://mirrors.ustc.edu.cn/debian-security/ <VERSION>/updates main contrib non-free
	```

> - *<VERSION>* 为 *debian* 的版本名，根据版本改变
> - 一般的，直接将默认配置文件中 `http://deb.debian.org` 修改为相应源地址即可：`$ sudo sed -i 's/deb.debian.org/<mirror_addr>/g' /etc/apt/sources.list`

###	*openSUSE*

> - *openSUSE* 使用 *MirrorBrain* 技术，[中央服务器](http:/download.opensuse.org)会按照 *IP* 中转下载请求到附近的镜像，所以更改软件源通常只会加快刷新软件元的速度，对下载速度影响不大

####	命令行

-	*USTC* 源：<https://mirrors.ustc.edu.cn/help/opensuse.html>

	```shell
	# 禁用原有软件源
	$ sudo zypper mr -da
	$ sudo zypper ar -fcg https://mirrors.ustc.edu.cn/opensuse/distribution/leap/\$releasever/repo/oss USTC:OSS
	$ sudo zypper ar -fcg https://mirrors.ustc.edu.cn/opensuse/distribution/leap/\$releasever/repo/non-oss USTC:NON-OSS
	$ sudo zypper ar -fcg https://mirrors.ustc.edu.cn/opensuse/update/leap/\$releasever/oss USTC:UPDATE-OSS
	$ sudo zypper ar -fcg https://mirrors.ustc.edu.cn/opensuse/update/leap/\$releasever/non-oss USTC:UPDATE-NON-OSS
	# 15.3 或更高版本需要
	$ sudo zypper ar -fgc https://mirrors.ustc.edu.cn/opensuse/update/leap/\$releasever/sle USTC:UPDATE-SLE
	```

> - `$releasever`：*OpenSuSe leap* 版本，若知晓可以自行替换

####	配置文件

> - *openSUSE* 源配置文件夹：`/etc/zypp/repo.d`

-	配置文件格式

	```cnf
	[<ALIAS>]			# 源别名
	enabled=1			# 默认是否启用
	autorefresh=0
	baseurl=url			# 源地址
	type=rpm-md
	```

###	*CentOS*

> - 发行版中 `yum` 一般自带 `fast-mirrors` 插件，一般无需更新官方源

####	三方源配置

-	*Extra Packages for Enterprise Linux*：由 *Fedora* 社区创建、维护的 *RPM* 仓库，通常不会与官方源发生冲突或相互替换文件
	-	安装 *EPEL*：`$ sudo yum install epel-release`
	-	包括应用有
		-	*Chromium*

-	*RPMFusion*：提供 *Fedora* 和 *RedHat* 由于开源协议或者是禁止商业用途而无法提供 *RPM* 安装包
	-	包括两个仓库`free`、`nofree`
		-	`free`：开源软件但是由于其他原因无法提供
		-	`non-free`：闭源软件，包括不能用于商业用途
	-	包含应用有
		-	`mplayer`
		-	`gstreamer-pluginsXXXX`

-	*ELRepo*：包含和硬件相关的驱动程序
	-	安装
		```shell
		$ rpm --import https://www.elrepo.org/RPM-GPG-KEY-elrepo.org
		$ rpm -Uvh http://www.elrepo.org/elrepo.org/elrepo-release-7.0-3.el7.elrepo.noarch.rpm
		```

-	*NuxDextop*：包含与多媒体相关应用的 *RPM* 仓库
	-	安装
		```shell
		$ rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm
		```
	-	说明
		-	有的依赖在 *EPEL* 中，因此可能需要先安装 *EPEL*
		-	和其他源有（*RPMFusion*）冲突时
			-	可以设置默认情况下不启用，即修改 `/etc/yum.repos.d/nux.dextop.repo` 文件，设置 `enable=0`
			-	需要时手动启用：`$ yum --enablerepo=nux-dextop install <PACKAGENAME>`

###	应用安装方式

-	包管理器
	-	安装应用比较方便
	-	但某些发行版中应用源更新缓慢

-	自行下载二进制版本安装
	-	*Linux* 大部分应用是 *noarch*，即与架构无关，无需考虑兼容问题
	
-	下载源码编译安装
	-	安装流程
		-	查询文档安装编译依赖
		-	`./configure`配置编译选择，如：安装路径等
		-	`make & make install`

-	注意事项
	-	自行安装应用可以若设置安装路径不是推荐路径，记得检查环境变量 `XXXX_HOME`
	-	应用文件夹通常带有版本号，建议
		-	保留文件夹版本号
		-	另行创建无版本号符号链接指向所需版本文件夹

##	本地化

###	字体

-	终端中字体需要为 *monospace*
	-	在多语言环境下，非 *monospace* 字体字符宽度不同，导致字符重叠
	-	字体名称不是字体文件名，其定义在字体文件内部定义
		-	指定未安装字体只能通过文件名
		-	指定已安装字体可直接使用字体名称

###	`Locale`

`Locale`：特定于某个国家、地区的编码设定

-	代码页
-	数字、货币、时间与日期格式


