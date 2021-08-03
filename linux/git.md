---
title: Git 基础
categories:
  - Linux
tags:
  - Tool
  - Git
  - Version Control
date: 2019-07-10 00:48:32
updated: 2021-08-02 10:56:53
toc: true
mathjax: true
comments: true
description: Git常识
---

##	.gitingore

###	.gitignore忽略规则

-	`!`开头：排除**应被忽略**

-	`/`**结尾**：忽略文件夹

-	`/`**开头**：git仓库根目录路径开始（可省略）

-	遵循一定的正则表达式
	-	`*`：多个字符，不包括`/`，因此多层级目录需要一一指定
	-	`?`：单个字符
	-	`[]`：匹配其中候选字符

###	.gitignore配置方式

-	仓库根目录创建`.gitignore`文件，添加忽略规则
	-	忽略规则针对当前项目
	-	`.gitignore`文件*默认*随仓库分发，所有人共用忽略规则
		（当然可以在`.gitignore`中配置忽略`.gitignore`）

-	设置全局忽略文件，对所有git项目适用
	```shell
	git config --global core.excludesfile /path/to/.gitignore
	```

-	修改`.git/info/exclude`文件，添加忽略规则
	-	对单一项目有效
	-	非所有人共用忽略规则

##	Git配置/`config`

-	配置文件、配置类型
	-	`--system`：系统全局配置，配置文件`/etc/gitconfig`
	-	`--global`：用户全局配置，配置文件
		`$HOME/.gitconfig`、`$HOME/.config/git/config`
	-	`--local`：局部repo配置，配置文件`repo/.git/config`

	```sh
	# 修改`section.key`值（缺省`--local`）
	$ git config [--system|--global|--local] <section>.<key> <value>
	# 删除配置项
	$ git config [--system|--global|--local] --unset <section>.<key>
	```

###	配置文件常用

```md
 # 核心修改
[core]
	# 忽略文件权限修改
	filemode = false
	editor = vim
	# 提交、检出时换行符设置
		# `input`：提交时转换为`<LF>`、检出时不转换
		# `false`：提交、检出均不转换
		# `true`：提交时转换为`<LF>`、检出时转换为`<CRLF>`
	autocrlf = input
	# 是否允许文件混用换行符
		# `true`：拒绝提交包含混合换行符文件
		# `false`：允许提交包含混合换行符文件
		# `warn`：提交包含混合换行符文件时警告
	safecrlf = true
[user]
	name = xyy15926
	email = xyy15926@163.com
 # 设置别名
[alias]
	st = status
	ci = commit
	br = branch
	lg = "log --color --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit"
	d  =  difftool
 # 输出着色
[color]
	# 打开所有默认终端着色
	ui = true
[diff]
	tool = vimdiff
[difftool]
	prompt = false
```

-	`autocrlf`在linux若设置为`input`，在`add`包含`<CRLF>`
	文件会报`fatal`错
	-	因为`input`在提交时会将`<CRLF>`转换为`<LF>`，但在
		检出时无操作
	-	所以导致即使`add`也不能保证repo当前状态和提交状态
		一致

###	`remote`

```sh
 # 查看远程仓库
$ git remote -v
 # 设置远程仓库地址
$ git remote set-url <repo_name> <new_url>
```

###	指定ssh key

> - git依赖ssh进行存储库认证，无法直接告诉git使用哪个私钥

-	`~/.ssh/config`中配置ssh host：git每次使用host代替原
	服务器地址

	```conf
	host <host>
	HostName github.com
	IdentityFile $HOME/.ssh/id_rsa_github_private
	User git
	```

	> - ssh host详见*linux/shell/config_files"

-	`GIT_SSH_COMMAND`环境变量：指定ssh命令`-i`中参数

	```shell
	GIT_SSH_COMMAND="ssh -i ~/.ssh/id_rsa_github_private" git clone ...
	// 可以`export`持久化
	export GIT_SSH_COMMAND = ...
	// 写入git配置
	git config core.sshCommand "..."
	```

##	展示命令

###	`log`

```shell
$ git log [<file_name>]
	# 查看文件提交历史
```

###	`show`

```c
$ git show <tag_no|commit_hash>
	# 查看改动内容
```

