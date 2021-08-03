---
title: Hexo 建站
categories:
  - Web
  - NPM
tags:
  - Web
  - NPM
  - Hexo
  - Blog
date: 2021-08-03 15:46:51
updated: 2021-08-03 18:28:54
toc: true
mathjax: true
description: 
---

##	Hexo 安装、配置

-	安装 Hexo、并建站

	```sh
	 # 安装 Hexo
	npm install -g hexo-cli
	 # 初始化 Hexo
	hexo init <folder>
	cd <folder>
	 # 安装 Hexo 依赖
	npm install
	```

-	启动站点：<https://hexo.io/zh-cn/docs/server>

	```sh
	# 安装独立的服务器模块（也可以不注册依赖）
	npm install hexo-server --save
	# 启动服务器
	hexo server [-p <port>]
	```

###	Hexo 站点结构

-	`_config.yml`：配置信息
-	`package.json`：*Hexo* 模块 `npm` 配置文件
-	`scaffolds`：模板文件
	-	新建文章时，尝试根据布局参数寻找相应模板建立文章
-	`source`：资源
	-	`_posts`：文章
	-	`_drafts`：草稿文章，默认会被忽略 <https://hexo.io/zh-cn/docs/writing#%E8%8D%89%E7%A8%BF>
	-	`_data`：其中 *YAML*、*JSON* 文件可以作为全站数据引用 <https://hexo.io/zh-cn/docs/data-files>
	-	其余 `_` 开头文件、文件夹和隐藏文件将被忽略
	-	其中 *Markdown*、*HTML* 文件被解析放到 `public` 文件夹下
-	`public`：
-	`themes`：主题文件夹，一个文件夹即一个主题

> - <https://hexo.io/zh-cn/docs/configuration>

###	Hexo 主题结构

-	`_config.yml`：主题配置文件
	-	修改后会自动更新
-	`languages`：语言文件夹
-	`layout`：布局文件夹，存放主题模板文件
	-	Hexo 内建 *Swig* 模板引擎，可另外安装 *EJS*、*Haml*、*Jade*、*Pug* 插件支持
	-	Hexo 根据模板文件扩展名决定模板引擎
-	`scripts`：脚本文件夹
	-	启动时，Hexo 会载入其中 *JS* 文件
-	`source`：资源文件夹
	-	除模板外的资源，如：*CSS*、*JS* 均位于此处
	-	`_` 开头文件、文件夹和隐藏文件被忽略
	-	若文件可被渲染，则会被解析存储到 `public` 文件夹，否则直接拷贝

###	Hexo 命令

```sh
hexo init [folder]
hexo new [layout] <title>			# 新建
hexo generate						# 生成静态文件
hexo publish [layout] <filename>	# 发布草稿
hexo deploy [-g]					# 部署
hexo render <file1> [file2] [-o <output>]			# 渲染文件
hexo migrate <type>					# 从其他博客系统迁移
hexo clean							# 清楚缓存、静态文件
hexo list <type>					# 列出站点资料
hexo version
```

> - <https://hexo.io/zh-cn/docs/commands>





