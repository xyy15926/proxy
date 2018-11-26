#	系统安装常识

##	Win10/8系统

在已经安装win10/8系统的机器上安装linux系统，需要注意

-	电源设置中关闭快速启动，其有可能影响grub开机引导
-	boot中关闭secureboot

##	U盘启动盘

Win下usbwriter和ultraiso都可以制作，但是

-	usbwriter用于archlinux的制作
	-	打开
	-	启动->写入磁盘映像
	-	写入方式--USB-HDD+
-	ultraiso用于ubuntu和centos的制作

archlinux使用ultraiso和usbwriter制作u盘不同，用usbwriter的
u盘好像有隐藏分区，只能看到一个很小容量的盘，使用ultraiso
制作的启动盘好像没用

##	bootloader（启动引导器）

双系统根据需求选择

###	Windows使用EasyBCD引导Linux

####	EasyBCD

Windows下的一个引导器

NeoGrub是EasyBCD自带的`grub`（不是已经更加常用的`grub2`），
可以配置EasyBCD在无法找到其他Linux系统的`grub`的情况下，
将控制权转移给NeoGrub，使用其引导系统，这个NeoGrub就和
普通的`grub`类似，可以引导多个系统

-	有很多文件系统格式无法识别，能够确认可识别的只有`ext2`，
	不能的有`xfs`、`ext4`，其中`ext4`会被错认为`ex2fs`，
	不能正确读取分区文件，
-	据说只能识别标准格式的分区，无法识别`lvm`格式的分区

因此，如果需要使用NeoGrub引导系统，需要注意分区格式问题
		
####	分区

-	Ubuntu：”安装启动引导器的设备”设置为sdXY，即磁盘X的
	Y分区，这样不会更改默认引导程序，此时会安装`grub`，但是
	没有覆盖磁盘中默认的win引导，重启后会自动进入win，可以
	使用easybsd自动检测引导分区，直接就能添加引导条目
-	Centos：选择不安装启动引导器`grub`(centos无法选择引导器
	安装在某个分区），EasyBCD无法自动检测Linux的引导分区，
	需要手动配置EasyBCD自带的NeoGrub，并添加此条目


####	引导文件编写

添加NeoGrub条目之后，其配置文件仍然为空，因此选择此条目之后
会直接进入grub控制台，在此尝试boot其他系统。

-	`root (hdX,`：`X`表示磁盘代号，`<tab>`给出的候选分区，
	确定boot分区`Y`

-	`root (hdX,Y)`：指定根目录（中间有空格）

-	`kernel /vmlinuz`：`<tab>`给出候选的内核文件，确定内核
	文件（一般会有一个rescue的文件肯定不是）

-	`kernel /vmlinuz---------- ro root=/dev/sdXY ro quite vga=791`：
	其中`X`不再是`hd`后面的数字而是字母，`Y`是`root`中`Y+1`

-	`initrd /initramfs`：`tab`给出候选initrd镜像文件

-	`initrd /initramfs---------`

-	`boot`

如果成功进入系统，说明以上的命令可行，记录以上真正有效的
命令，据此修改NeoGrub配置文件

	title name
		root (hdX,Y)
		kernel /vmlinuz------------ ro root=/dev/sdXY ro quite vga=791
		initrd /initramfs----------
		boot

可以在EasyBCD中打开配置文件，也可以直接修改`C:/NST/menu.ls`
文件

###	Linux使用grub引导Windows

`grub`可以引导包括win、linux在内的大部分系统，而且大部分教程
都是默认这种方式

-	Ubuntu：”安装启动引导器的设备”设置为sdX，即直接安装在
	磁盘的最开始，修改默认引导
-	Centos：选择安装启动引导器
-	Archlinux：要手动安装os-prober（检测已安装系统）、grub，
	配置grub启动文件，具体方法参见grub使用或是archlwiki

##	分区

-	`/`根分区：唯一必须分区

-	`boot`分区：一般建议单独给boot分区

	-	根分区位于lvm、RAID，或者是文件系统不能被引导程序
		识别，单独的boot分区可以设为其他格式
	-	可以以只读方式挂载boot分区，防止内核文件被破坏
	-	多系统可以共享内核

-	`swap`分区

	-	如果设置swap分区，一般设置为内存的1～2倍
	-	建议使用交换文件代替，这样的比较灵活，而且如果内存
		够用的话可以不启用swap文件，这样提升效率、保护硬盘

		-	`fallocate -l SIZE /SWAPFILENAME`：创建交换文件
			，其中SIZE后面要跟单位（M、G）
		-	`chmod 600 /SWAPFILENAME`：更改交换文件权限
		-	`mkswap /SWAPFILENAME`：设置交换文件
		-	`swapon /SWAPFILENAME`：启用**一次**交换文件

		或修改`/etc/fstab`文件，每次开机默认启用交换文件

			/SWAPFILENAME	none	swap	defaults	0	0
			/SWAPFILENAME	swap	swap	sw	0	0

		前者是Arch教程，后者是Centos7教程，这两种写法应该
		是一样的
