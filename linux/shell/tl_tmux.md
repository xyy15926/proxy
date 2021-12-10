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
updated: 2021-12-10 20:56:37
toc: true
mathjax: true
comments: true
description: Tmux
---

##	*Tmux*

-	*Tmux*：终端复用软件
	-	将 *terminal* 、*Session* 解绑，避免
	-	允许多个窗口、多个 *Session* 之间的连接、断开
		-	新增多个 *Session*
		-	接入已有 *Session*，共享 *Session*
	-	支持窗口竖直、水平拆分

-	默认配置文件为 `~/.tmux.conf`

> - *(Shell) Session*：*Shell* 工作状态、环境，一般指用户与 *Shell* 的一次交互
> - <https://man7.org/linux/man-pages/man1/tmux.1.html>

###	服务器、客户端管理

-	相关命令
	-	`start-server`/`start`
	-	`suspend-client`/`suspendc`
	-	`refresh-client`/`refresh`
	-	`show-messages`/`showmsgs`
	-	`list-command`：列出所有命令
	-	`info`：列出当前所有会话信息
	-	`source-file <TMUX_CONF>`/`source`：重新加载配置文件
	-	`send-prefix`：发送快捷键映射
	-	`if-shell <SH_CMD> <TMUX_CMD>`

	通用选项
	-	`-t <TARGET_CLIENT>`

###	Sesssion 管理

-	`tmux` 会话子命令
	-	`new-session`/`new`：新建会话
	-	`detach-client`/`detach`：断开当前连接会话（缺省上次）
	-	`attach-sesssion`/`attach`：连接已有会话（缺省上次）
	-	`rename`：重命名会话
	-	`switch`：切换至指定会话
		`kill-session`：关闭指定会话
	-	`kill-server`：关闭所有会话
	-	`list-session`/`ls`：列出会话列表

-	`tmux` 通用选项
	-	`-S <SOCKET_FILE>`：指定 *socket* 文件（位置）
	-	`-s <SESS_NAME>`：会话名（新建会话时）
	-	`-t <TARGET_SESS>`：目标会话名
	-	`-c <WORKING_DIR>`：工作目录
	-	`-a`：关闭除指定会话外其他会话

###	Window（Tab） 管理

-	`tmux` Tab 子命令
	-	`new-window`/`neww`：创建新窗口
	-	`switch-window`：切换到指定窗口
	-	`rename-window`：重命名窗口
	-	`select-window`/`selectw`：激活窗口
	-	`swap-window`/`swapw`：交换窗口
	-	`move-window`/`movew`：移动窗口
	-	`next-window`/`next`：下个窗口
	-	`resize-window`/`resizew`
	-	`respawn-window`/`respawnw`
	-	`rotate-window`/`rotatew`
	-	`unlink-window`/`unlinkw`

-	`tmux` 通用选项
	-	`-n <WIN_NAME>`：窗口名（新建窗口时）
	-	`-t <TARGET_WIN>`：目标窗口名

###	Pane 管理

-	`tmux` Pane 管理子命令
	-	`split-window`：竖直、水平划分窗口
	-	`select-pane`/`selectp`：激活 Pane
	-	`swap-pane`/`swapp`：交换 Pane
	-	`move-pane`/`movep`：移动窗口
	-	`resize-pane`/`resizep`：调整 Pane 尺寸
	-	`run-shell <SHELL_COMMAND>`
	-	`respawn-pane`/`respawnp`
	-	`send-keys <KEY>`/`send`

-	`tmux` 通用选项
	-	`-h`：水平划分窗口
	-	`-U`、`-D`、`-L`、`-R`：上、下、左、右
	-	`-s <SRC_PANE>`
	-	`-t <DST_PANE>`
	-	`-x <WIDTH>`
	-	`-y <HEIGHT>`

###	缓冲

-	`tmux` 缓冲命令
	-	`set-buffer <DATA>`/`setb`
	-	`paste-buffer`/`pasteb`
	-	`save-buffer <FILENAME>`/`saveb`
	-	`show-buffer`/`showb`
	-	`delete-buffer`/`deleteb`

-	通用选项
	-	`-b <BUF_NAME>`
	-	`-n <NEW_BUF_NAME>`
	-	`-t <TARGET_PANE>`

##	*TMUX* 选项设置

-	`tmux` 选项设置子命令
	-	`set-environment <NAME> [VALUE]`/`setenv`：会话环境变量`
	-	`set-hook <HOOK> [COMMAND]`：会话钩子
	-	`set-option <OPTION> [VALUE]`/`set`：Pane 选项值
	-	`set-window-option <OPTION> [VALUE]`/`setw`：窗口选项值

-	`tmux` 选项查看子命令
	-	`show-environment`/`showenv`
	-	`show-hooks`
	-	`show-options`/`show`
	-	`show-window-options`/`showw`

-	`tmux` 通用选项
	-	`-t <TARGET>`：目标会话、窗口、Pane

###	`set-option`

-	`set-option` 标志位
	-	`-g`：全局设置
	-	`-a`：追加设置，适合选项值为字符串

-	`set-option` 常用设置选项
	-	`prefix`：快捷键映射前缀
	-	`default-terminal`
	-	`display-time`
	-	`escape-time`
	-	`history-limit`
	-	`base-index`
	-	`pane-base-index`

###	`set-window-option`

-	`set-window-option` 标志位
	-	`-g`：全局设置
	-	`-a`：追加设置，适合选项值为字符串

-	`set-window-option` 常用设置选项
	-	`allow-rename`
	-	`mode-keys`：快捷键模式，可以设置为`vi`
	-	`synchronize-panes`
	-	`monitor-activity`
	-	`visual-activity`
	-	`bell-action`
	-	`status-interval`

####	*StatusBar* 选项

-	*StatusBar* 主要由 5 部分组成
	-	Window 列表：不同选项前缀为不同状态窗口设置
		-	`windows-status-*`：默认windows
		-	`windows-status-current-*`：当前windows
		-	`windows-status-bell-*`：后台有活动windows
			（需开启支持）
	-	左侧显示区
	-	右侧显示区
	-	Message 显示条：占据整个 StatusBar
	-	Command 输入条：占据整个 StatusBar

-	Window 列表选项
	-	`*-style bg=<color>,fg=<color>,<ATTR>`：窗口展示样式
		-	颜色可以用名称、`colour[0-255]`、`#RGB`方式指定
		-	属性包括（前加 `no` 表关闭）
			-	`bright`
			-	`dim`
			-	`underscore`
			-	`blink`
			-	`reverse`
			-	`hidden`
			-	`italics`
			-	`strikethrough`
	-	`*-format`：窗口展示内容
		-	`#{}` 中变量名称会被替换为相应值（支持 `alias` 缩写的变量可以省略 `{}`）
			-	`host`：`#H`
			-	`host_short`：`#h`
			-	`pane_id`：`#D`
			-	`pane_index`：`#P`
			-	`pane_title`：`#T`
			-	`session_name`：`#S`
			-	`window_flags`：`#F`
				-	`*`：当前窗口
				-	`-`：最后打开的窗口
				-	`z`：Zoom 窗口
			-	`window_index`：`#I`
			-	`window_name`：`#W`
		-	`#()`会被作为 Shell 命令执行被替换为结果
			-	命令执行不会阻塞 `tmux`，而是展示最近一次命令执行结果
			-	刷新频率由 `status-interval` 控制

> - 这里介绍2.9之后配置风格

##	*TMUX* 键映射

-	`tmux` 键映射相关命令
	-	`bind-key <KEY> <COMMAND>`/`bind`
	-	`unbind-key <KEY>`/`unbind`
	-	`list-key`：列出所有绑定键

-	通用选项
	-	`-T <KEY_TABLE>`

###	`bind`

-	`bind` 命令标志位
	-	`-n`：无需键映射前缀
	-	`-r`：允许递归映射

-	`bind` 部分特殊默认键映射
	-	`<PREFIX>:`：进入 *Tmux* 命令行
	-	`?`：列出所有绑定键

> - 键映射前缀缺省为 `C-b`

#### `bind` 默认键映射

-	会话相关
	-	`s`：列出会话，可用于切换
	-	`$`：重命名会话
	-	`d`：断开当前会话
	-	`D`：断开指定会话

-	窗口相关
	-	`c`：创建新 Tab
	-	`&`：关闭当前 Tab
	-	`,`：重命名当前 Tab
	-	`.`：修改当前 Tab 索引编号
	-	`w`：列出所有 Tab
	-	`n`、`p`、`l`：进入下个、上个、之前操作 Tab
	-	`<N>`：切换到编号 `N` Tab
	-	`f`：切入 `Tmux` 命令行，根据内容搜索 Tab

-	Pane 相关
	-	`%`：水平方向创建窗口
	-	`"`：竖直方向创建窗口
	-	`<Up>`、`<Down>`、`<Left>`、`<Right>`：根据箭头访问切换窗口
	-	`q`：显示窗口编号
	-	`o`：顺时针切换窗口
	-	`}`/`{`：与下个/上个窗口交换位置
	-	`space`：在预置面板布局中循环切换
		-	*even-horizontal*
		-	*even-vertical*
		-	*main-horizontal*
		-	*main-vertical*
		-	*tiled*
	-	`!`：将当前窗口置于新tab
	-	`C-o`、`M-o`：顺、逆时针旋转当前窗口（即所有窗口循环前、后移动一个位次）
	-	`t`：在当前窗口显示时间
	-	`z`：最大、恢复当前窗口
	-	`i`：显示当前窗口信息
	-	`x`：关闭当前窗口
	-	`q`：显示当前窗口编号

-	Pane 内容操作想过
	-	`[`：进入自由复制模式，*VI* 模式下快捷键同 *VI*，且
		-	`<Space>`：从光标处开始选择（支持 `V` 选择行）
		-	`<CR>`：复制选择部分
	-	`]`：粘贴



