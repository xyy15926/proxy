---
title: Tmux
tags:
  - 工具
categories:
  - 工具
date: 2019-06-30 20:16:34
updated: 2019-06-30 20:16:34
toc: true
mathjax: true
comments: true
description: Tmux
---

##	命令行参数

###	Session管理

-	`tmux new -s <session-name>`：创建指定名称session
	-	缺省数字命名
-	`tmux ls`：列出session列表
-	`tmux rename -t <s1> <s2>`：重命名session
-	`tmux a [-t <session-name>]`：attach指定session
	-	缺省连接上个session
-	`tmux kill-session [[-a] -t s1]`：关闭session
	-	缺省关闭上次session
	-	`-a`：关闭除指定session外其他session
-	`tmux kill-server`：关闭所有session

###	帮助

-	`tmux list-key`：列出所有绑定键
-	`tmux list-command`：列出所有命令

##	配置

###	`set`

-	配置文件为`~/.tmux.conf`
-	`set[-option] -s`：server选项
-	`set[-option] [-g] [-a]`：session选项
-	`setw/set-window-option [-g] [-a]`：window选项
	-	`-g`：全局设置
	-	`-a`：追加

###	`bind`

-	`bind[-key] [-n] [-r] <key>`：key mapping
	-	`-n`：无需prefix
	-	`-r`：此键可能重复
-	`unbind <key>`：解除捕获

##	默认KeyMappings

-	快捷键前缀缺省为`C-b`
-	`<prefix>:`：进入命令行

###	Session

-	`s`：列出session，可用于切换
-	`$`：重命名session
-	`d`：detach当前session
-	`D`：detach指定session

###	Windows

-	`c`：创建新tab
-	`&`：关闭当前tab
-	`,`：重命名当前tab
-	`.`：修改当前tab索引编号
-	`w`：列出所有tab
-	`n`/`p`/`l`：进入下个/上个/之前操作tab
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
-	`C-o`/`M-o`：顺/逆时针旋转当前窗口
-	`t`：在当前窗口显示时间
-	`z`：最大/恢复当前窗口
-	`i`：显示当前窗口信息
-	`x`：kill当前窗口

> - tmux中pane相当于windows

###	信息

-	`?`：列出所有绑定键

