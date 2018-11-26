#	CentOS7常用配置

##	网络配置

编辑`/etc/sysconfig/network-scripts/ifcfg-ens33`

```cnf
TYPE = Ethernet				# 网卡类型：以太网
PROXY_METHOD=none			# 代理方式：无
BROWSER_ONLY=no				# 仅浏览器：否
BOOTPROTO=dhcp				# 网卡引导协议
DEFROUTE=yes				# 默认路由：是
IPV4_FAILURE_FATAL=no		# 开启IPV4致命错误检测：否
IPV6INIT=yes				# IPV6自动初始化：是
IPV6_AUTOCONF=yes			# IPV6自动配置：是
IPV6_DEFROUTE=yes			# IPV6是否可为默认路由：是
IPV6_FAILURE_FATAL=no		# 开启IPV6致命错误检测：否
IPV6_ADDR_GEN_MODE=stable-privacy
							# IPV6地地址生成模型：stable-privacy
NAME=ens33					# 网卡物理设备名称
UUID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
							# 通用唯一识别码
DEVICE=ens33				# 网卡设备名称
ONBOOT=yes					# 开启启动：是
DNS1=xxx.xxx.xxx.xxx		# DNS地址
IPADDR=xxx.xxx.xxx.xxx		# IP地址
PREFIX=24					# 子网掩码
GATEWAY=xxx.xxx.xxx.xxx		# 网关
```

-	UUID不能相同相同
-	`ifcfg-ens33`这个文件感觉像是个模板，但是不知道真正应用
	配置文件在哪

##	常用应用源

###	EPEL

Extra Packages for Enterprise Linux
由Fedora社区创建、维护的RPM仓库，通常不会与官方源发生冲突
或相互替换文件，包括应用有：chromium

直接使用yum安装：`$ sudo yum install epel-release`

###	RPMFusion

提供Fedora和RedHat由于开源协议或者是禁止商业用途而无法提供
RPM安装包，包括两个仓库
和NuxDextop源有冲突，如：gstreamer，感觉上比NuxDextop更加权威
包含应用：mplayer、gstreamer-pluginsXXXX、

-	free：开源软件但是由于其他原因无法提供，安装方式
	`$>sudo yum localinstall --nogpgcheck https://download1.rpmfusion.org/free/el/rpmfusion-free-release-7.noarch.rpm`

-	nonfree：闭源软件，包括不能用于商业用途，安装方式
	`$>sudo rpm -ivh https://download1.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-7.noarch.rpm`

###	ELRepo

包含和硬件相关的驱动程序，通过以下命令安装

	$>rpm --import https://www.elrepo.org/RPM-GPG-KEY-elrepo.org
	$>rpm -Uvh http://www.elrepo.org/elrepo.org/elrepo-release-7.0-3.el7.elrepo.noarch.rpm

###	NuxDextop

包含与多媒体相关应用的RPM仓库，好像是个人作者维护，有的依赖
可能在EPEL源中，因此可能需要先安装EPEL，可能和其他源
（RPMFusion）有冲突，可以设置默认情况下不启用，即修改
`/etc/yum.repos.d/nux.dextop.repo`文件，设置`enable=0`，
开启时手动启用
`$>yum --enablerepo=nux-dextop install PACKAGENAME`

	$>rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-5.el7.nux.noarch.rpm

		
			
## 装机必备

###	rhytmbox-mp3

centos7的gnome默认安装rhythmbox，但无法解码mp3，需要安装 
rpmfusion-free源中的

-	gstreamer-plugins-ugly.x86_64
-	gstreamer1-plugins-ugly.x86_64


### chromium
		
-	安装EPEL源之后直接安装

-	flash插件

	-	adobe官网下载flash-player-ppapi：
		<http://get.adobe.com/flashplayer>
	-	<pkgs.org>下载chromium-pepper-flash

	ppapi好像就是pepperapi的简称，但是两个flash插件不一样，
	安装的是pkgs上下载的,fedora社区维护的

-	html5视频播放支持：ffmpeg-libs
	google准备不再支持h.264格式（绝大部分）的视频，所以装
	了这个还需要其他设置，但firefox可播放大部分html5视频

###	wqy中文字体

yum源里的字体文件都是`*.ttc`文件，需要`*ttf`字体文件，有
在线解压网站可以解压

##	安装包常识

-	app和app-devel/app-dev：后者包括头文件、链接库，在编译
	使用了app的源代码才需要

##	系统配置

###	文件目录常识

-	`/usr/share/applications`里*.desktop是“桌面图标”文件，
	centos会 菜单中的会展示的“应用”就是这些

