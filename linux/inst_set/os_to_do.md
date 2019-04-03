#	Linux安装后常用配置

##	用户设置

###	设置root密码

linux安装之初，在设置root密码之前无法使用`$ su`切换到root
用户，需要先设置root用户密码
```shell
$ sudo passwd root
```

##	应用设置

###	配置源

这个根据系统不同需要不同的操作，但是一般是修改
`/etc/pkg_manager/`中的文件，注意修改为国内源之后一般还需要
手动刷新

####	Debian

-	修改`/etc/apt/source.list`，添加163、ustc的国内源
-	`$ sudo apt update`更新源本地缓存

其中`stretch`是debian的版本名，需要根据版本改变。其中ustc源
不知道为啥没有速度，有可能是https的原因。

#####	163源

```cnf
deb http://mirrors.163.com/debian/ stretch main non-free contrib
deb http://mirrors.163.com/debian/ stretch-updates main non-free contrib
deb http://mirrors.163.com/debian/ stretch-backports main non-free contrib
deb-src http://mirrors.163.com/debian/ stretch main non-free contrib
deb-src http://mirrors.163.com/debian/ stretch-updates main non-free contrib
deb-src http://mirrors.163.com/debian/ stretch-backports main non-free contrib
deb http://mirrors.163.com/debian-security/ stretch/updates main non-free contrib
deb-src http://mirrors.163.com/debian-security/ stretch/updates main non-free contrib
```

#####	ustc源
```cnf
deb https://mirrors.ustc.edu.cn/debian/ stretch main contrib non-free
deb-src https://mirrors.ustc.edu.cn/debian/ stretch main contrib non-free
deb https://mirrors.ustc.edu.cn/debian/ stretch-updates main contrib non-free
deb-src https://mirrors.ustc.edu.cn/debian/ stretch-updates main contrib non-free
deb https://mirrors.ustc.edu.cn/debian/ stretch-backports main contrib non-free
deb-src https://mirrors.ustc.edu.cn/debian/ stretch-backports main contrib non-free
deb https://mirrors.ustc.edu.cn/debian-security/ stretch/updates main contrib non-free
deb-src https://mirrors.ustc.edu.cn/debian-security/ stretch/updates main contrib non-free
```

####	OpenSuSe

使用MirrorBrain技术，[中央服务器](http:/download.opensuse.org)
会按照IP中转下载请求到附近的镜像，所以更改软件源通常只会加快
刷新软件元的速度，对下载速度影响不大

`$ sudo zypper ref`刷新本地的源

#####	终端配置

```shell
$ sudo zypper mr -da
	# 禁用原有软件源
$ sudo zypper ar -fc -n oss-dist https://mirrors.ustc.edu.cn/opensuse/distribution/leap/XXXX/repo/oss oss-dist
$ sudo zypper ar -fc -n non-oss-dist https://mirrors.ustc.edu.cn/opensuse/distribution/leap/XXXX/repo/non-oss non-oss-dist
$ sudo zypper ar -fc -n oss-update https://mirrors.ustc.edu.cn/opensuse/update/leap/XXXX/oss oss-update
$ sudo zypper ar -fc -n non-oss-update https://mirrors.ustc.edu.cn/opensuse/update/leap/XXXX/non-oss non-update-oss
$ sudo zypper ar -fc -n debug-dist https://download.opensuse.org/debug/distribution/leap/XXXX/repo/oss/ debug-dist
$ sudo zypper ar -fc -n debug-update http://download.opensuse.org/debug/update/leap/XXXX/oss/ debug-update
```

> - `/leap/XXXX`：leap XXXX版本，需要根据具体版本更改

-	`debug-dist`/`debug-update`：调试相关包，包括`glibc-debuginfo`

#####	手动配置

以上4条命令分添加了4个源，即在`/etc/zypp/repos.d`添加4个文件
，名称就是`alias.repo`，因此可以自行在其中添加文件配置源

```cnf
[alias]
enabled=1
autorefresh=0
baseurl=url
type=rpm-md
```

类似的，将默认的`oss.repo`中改为`enabled=0`即可禁用默认源

####	CentOS7

发行版中`yum`一般自带`fast-mirrors`插件，对于CentOS更多的是
需要安装三方应用源

-	注意这里的源地址对应CentOS7，其他版本需要更改

#####	EPEL

Extra Packages for Enterprise Linux由Fedora社区创建、维护的
RPM仓库，通常不会与官方源发生冲突或相互替换文件

-	包括应用有：chromium

-	安装
	```shell
	$ sudo yum install epel-release
	```

#####	RPMFusion

提供Fedora和RedHat由于开源协议或者是禁止商业用途而无法提供
RPM安装包

-	包括两个仓库`free`、`nofree`
	-	`free`：开源软件但是由于其他原因无法提供
	-	`non-free`：闭源软件，包括不能用于商业用途
	-	和NuxDextop源有冲突，如：`gstreamer`，感觉上比
		NuxDextop更加权威

-	包含应用：`mplayer`、`gstreamer-pluginsXXXX`

-	安装
	```md
	$ sudo yum localinstall --nogpgcheck \
		https://download1.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm
		# free
	$ sudo rpm -ivh \
		https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-7.noarch.rpm
		# non-free
	```

###	ELRepo

包含和硬件相关的驱动程序

-	安装
	```shell
	$ rpm --import https://www.elrepo.org/RPM-GPG-KEY-elrepo.org
	$ rpm -Uvh http://www.elrepo.org/elrepo.org/elrepo-release-7.0-3.el7.elrepo.noarch.rpm
	```

###	NuxDextop

包含与多媒体相关应用的RPM仓库，好像是个人作者维护

-	有的依赖可能在EPEL源中，因此可能需要先安装EPEL

-	可能和其他源（RPMFusion）有冲突，可以设置默认情况下
	不启用，即修改`/etc/yum.repos.d/nux.dextop.repo`文件，
	设置`enable=0`，开启时手动启用
	```md
	$ yum --enablerepo=nux-dextop install PACKAGENAME
	```

-	安装
	```shell
	$ rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm
	```

###	应用

-	使用包管理器安装应用比较方便，只是有限发行版源更新缓慢

-	自行下载二进制版本应用安装，linux大部分应用是*noarch*，
	即与架构无关，无需考虑兼容问题
	
-	下载源码编译安装
	-	查询文档安装全编译依赖
	-	`./configure`配置编译选择，如：安装路径等
	-	`make & make install`

-	自行安装应用可以若设置安装路径不是linux系统推荐路径，
	一般需要自行配置

	```shell
	export XXXX_HOME=/path/to/app`
	export PATH=$PATH:$XXXX_HOME/bin
	```

	-	一般二进制包解压文件夹通常带有版本号，可以考虑保留
		版本号，但是为了方便和升级起见，可以考虑创建无版本号
		**符号链接**指向原文件夹

####	essential

-	vim >= 8.0
-	git
-	ctags
-	python3
-	pyenv
	-	anaconda3
		-	pymysql
		-	pyecharts

####	optional

-	mariadb/mysql
-	postgresql
-	ssh
	```shell
	$ ssh-keygen
	$ eval `ssh-agent`
	$ ssh-add /path/to/rsa_private_key
	```

####	配置

参见[脚本文件](../../home_config/setting.sh)

##	本地化

###	字体

linux系统本身不带有中文字体，用于terminal的字需要是monospace
字体，否则可能在terminal中字符直接会重叠。在官方源中一般会有
一些开源字体（wqy-microhei），但是这些字体一般是ttc文件，而
有些应用不支持ttc文件安装的字体，所以需要手动安装ttf

```shell
$ sudo cp /path/to/fonts.ttf /usr/share/fonts/
$ cd /usr/share/fonts
$ fc-cache
$ fc-list :lang=zh
	#输出已安装中文字体
```

###	`Locale`

`Locale`：特定于某个国家、地区的编码设定

-	代码页
-	数字、货币、时间与日期格式


