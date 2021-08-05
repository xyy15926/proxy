---
title: Tmux
categories:
  - Linux
  - Tool
tags:
  - Linux
  - Tool
  - Terminal
  - Tmux
date: 2019-06-30 20:16:34
updated: 2020-07-12 18:10:43
toc: true
mathjax: true
comments: true
description: Tmux
---

##	Tmux

-	Session：用户与计算计算机的交互
	-	一般而言，temrinal窗口、Session进程绑定
		-	打开窗口，Session开始
		-	关闭窗口，Session结束

-	Tmux：终端复用软件，将terminal窗口、Session解绑
	-	允许多个窗口接入、断开多个Session
		-	新增多个Session
		-	接入已有Session，共享Session
	-	支持窗口竖直、水平拆分

<https://man7.org/linux/man-pages/man1/tmux.1.html>

##	命令行参数

-	`-S <socket-file>`：指定tmux使用socket文件（位置）

###	Session管理

-	`tmux new -s <session-name>`：创建指定名称session
	-	缺省数字命名
-	`tmux detach`：detach当前接入会话
-	`tmux ls/list-session`：列出session列表
-	`tmux rename -t <s1> <s2>`：重命名session
-	`tmux a [-t <session-name>]`：attach指定session
	-	缺省连接上个session
-	`tmux switch -t <session-name>`：切换至指定session
-	`tmux kill-session [[-a] -t s1]`：关闭session
	-	缺省关闭上次session
	-	`-a`：关闭除指定session外其他session
-	`tmux kill-server`：关闭所有session

###	Tab（Windows）管理

-	`tmux new-window [-n <win-name>]`：创建新窗口
-	`tmux switch-window -t <win-name>`：切换到指定窗口
-	`tmux rename-window <win-name>`：重命名当前窗口


###	Pane管理

-	`tmux split-window [-h]`：竖直/水平划分窗口
-	`tmux select-pane -U/-D/-L/-R`：激活上、下、左、右侧Pane
-	`tmux swap-pange -U/-D/-L/-R`：当前Pane上、下、左、右
	移动

###	帮助

-	`tmux list-key`：列出所有绑定键
-	`tmux list-command`：列出所有命令
-	`tmux info`：列出当前所有Tmux会话信息
-	`tmux source-file <tmux-conf>`：重新加载Tmux配置文件

##	配置

###	`set`

> - 默认配置文件为`~/.tmux.conf`

-	`set[-option] [-g] [-a]`：session选项
	-	全局、追加标志
		-	`-g`：全局设置
		-	`-a`：追加设置，适合`option`需要字符串、样式值
	-	`default-terminal`
	-	`display-time`
	-	`escape-time`
	-	`history-limit`
	-	`base-index`
	-	`pane-base-index`

-	`setw/set-window-option [-g] [-a]`：window选项
	-	全局、追加标志同`set[-option]`
	-	`allow-rename`
	-	`mode-keys`：快捷键模式，可以设置为`vi`
	-	`synchronize-panes`

-	`set[-option] -s`：server选项

####	StatusBar设置

-	StatusBar主要由5部分组成
	-	windows列表
		-	`windows-status-*`：默认windows
		-	`windows-status-current-*`：当前windows
		-	`windows-status-bell-*`：后台有活动windows
			（需开启支持）
	-	左侧显示区
	-	右侧显示区
	-	message显示条：占据整个status bar
	-	command输入条：占据整个status bar

-	`*-style bg=<color>,fg=<color>,<ATTR>`指定样式
	-	颜色可以用名称、`colour[0-255]`、`#RGB`方式指定
	-	属性包括（前加`no`表关闭）
		-	`bright`
		-	`dim`
		-	`underscore`
		-	`blink`
		-	`reverse`
		-	`hidden`
		-	`italics`
		-	`strikethrough`

-	`*-format`：设置格式
	-	`#{}`中变量名称会被替换为相应值，支持alias缩写的变量
		可以省略`{}`
		-	`host`：`#H`
		-	`host_short`：`#h`
		-	`pane_id`：`#D`
		-	`pane_index`：`#P`
		-	`pane_title`：`#T`
		-	`session_name`：`#S`
		-	`window_flags`：`#F`
			-	`*`：当前窗口
			-	`-`：最后打开的窗口
			-	`z`：Zoom窗口
		-	`window_index`：`#I`
		-	`window_name`：`#W`
	-	`#()`会被作为shell命令执行并被替换
		-	命令执行不会阻塞tmux，而是展示最近一次命令执行
			结果
		-	刷新频率由`status-interval`控制

> - 这里介绍2.9之后配置风格

###	`bind`

-	`bind[-key] [-n] [-r] <key>`：key mapping
	-	`-n`：无需prefix
	-	`-r`：此键可能重复
-	`unbind <key>`：解绑捕获

##	默认KeyMappings

-	快捷键前缀缺省为`C-b`
-	`<prefix>:`：进入命令行

> - 以下快捷键都是缺省值，可以解绑

###	Session

-	`s`：列出session，可用于切换
-	`$`：重命名session
-	`d`：detach当前session
-	`D`：detach指定session

###	Tab/Windows

-	`c`：创建新tab
-	`&`：关闭当前tab
-	`,`：重命名当前tab
-	`.`：修改当前tab索引编号
-	`w`：列出所有tab
-	`n`/`p`/`l`：进入下个/上个/之前操作tab
-	`[tab-n]`：切换到指定**编号**窗口
-	`f`：根据显示内容搜索tab

> - tmux中window相当于tab

###	Panes

-	`%`：水平方向创建窗口
-	`"`：竖直方向创建窗口
-	`Up/Down/Left/Right`：根据箭头访问切换窗口
-	`q`：显示窗口编号
-	`o`：顺时针切换窗口
-	`}`/`{`：与下个/上个窗口交换位置
-	`space`：在预置面板布局中循环切换
	-	even-horizontal
	-	even-vertical
	-	main-horizontal
	-	main-vertical
	-	tiled
-	`!`：将当前窗口置于新tab
-	`C-o`/`M-o`：顺/逆时针旋转当前窗口，即所有窗口循环前/后
	移动一个位次
-	`t`：在当前窗口显示时间
-	`z`：最大/恢复当前窗口
-	`i`：显示当前窗口信息
-	`x`：关闭当前窗口
-	`q`：显示当前窗口编号
-	`[`：进入自由复制模式，*VI* 模式下快捷键同 *VI*，且
	-	`<space>`：从光标处开始选择（支持 `V` 选择行）
	-	`<enter>`：复制选择部分
-	`]`：粘贴

> - tmux中pane相当于vim中windows

###	信息

-	`?`：列出所有绑定键

