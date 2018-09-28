#	Linux安装后常用配置

##	用户设置

###	设置root密码

linux安装之初，在设置root密码之前无法使用`$ su`切换到root
用户，需要先设置root用户密码
```shell
$ sudo passwd root
```

###	配置国内源

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

使用MirrorBrain技术的，[中央服务器](http:/download.opensuse.org)
会按照IP中转下载请求到附近的镜像，所以更改软件源通常只会加快
刷新软件元的速度，对下载速度影响不大

`$ sudo zypper ref`刷新本地的源

#####	终端配置

```shell
$ sudo zypper mr -da
	# 禁用原有软件源
$ sudo zypper ar -fc hhtps://mirrors.ustc.edu.cn/opensuse/distribution/leap/42.1/repo/oss alias_1
$ sudo zypper ar -fc https://mirrors.ustc.edu.cn/opensuse/distribution/leap/42.1/repo/non-oss alias_2
$ sudo zypper ar -fc https://mirrors.ustc.edu.cn/opensuse/update/leap/42.1/oss alias_3
$ sudo zypper ar -fc https://mirrors.ustc.edu.cn/opensuse/update/leap/42.1/non-oss alias_4
```

其中`/leap/42.1`表示opensuse-leap 42.1版本，需要根据具体版本
更改

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

###	应用

如果是下载、安装应用，可以（最好）下载**binary**版本，无需
**src**版本。binary版本是可以直接运行的，无需考虑不同linux
发行版的兼容性（如果binary版本不区分）

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

[脚本文件](../../home_config/setting.sh)

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

