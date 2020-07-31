---
title: Git常识
tags:
  - 工具
  - Git
categories:
  - 工具
  - Git
date: 2019-07-10 00:48:32
updated: 2019-07-10 00:48:32
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

##	`config`

-	修改配置类型：缺省查看配置
	-	`--global`：全局配置
	-	`--local`：局部repo配置

###	`user`

-	`user.name`：用户名
-	`user.email`：用户邮箱

```shell
$ git config user.name
$ git config --global user.name "username"
$ git config --local user.email "email"
```

###	`core`

```shell
$ git config --global core.autocrlf input
$ git config --global core.safecrlf true
```

-	`core.autocrlf`：提交、检出时换行符设置
	-	`input`：提交时转换为`<LF>`、检出时不转换
	-	`false`：提交、检出均不转换
	-	`true`：提交时转换为`<LF>`、检出时转换为`<CRLF>`

-	`core.safecrlf`：是否允许文件混用换行符
	-	`true`：拒绝提交包含混合换行符文件
	-	`false`：允许提交包含混合换行符文件
	-	`warn`：提交包含混合换行符文件时警告

##	`log`

```shell
$ git log [<file_name>]
	# 查看文件提交历史
```

##	`show`

```c
$ git show <tag_no|commit_hash>
	# 查看改动内容
```

##	其他配置

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


