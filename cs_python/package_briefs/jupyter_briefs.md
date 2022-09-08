---
title: Jupyter 常用基础
categories:
  - Python
  - Jupyter
tags:
  - Python
  - Jupyter
  - Briefs
date: 2019-03-21 17:27:37
updated: 2022-07-11 17:21:12
toc: true
mathjax: true
comments: true
description: Jupyter常用基础
---

#	Jupyter

##	*Jupyter* 相关项目

-	*Jupyter* 社区由大量子社区、项目组成
	-	*Jupyter* 用户界面
		-	[Jupyter Notebook][JupyterNotebook]
		-	[Jupyter Console][JupyterConsole]
		-	[Jupyter QtConole][JupyterQtConsole]
	-	*Jupyter* 内核：与 *Jupyter* 交互、独立执行特定语言的进程
		-	`ipykernel`：建立在 *IPython* 项目上的默认 Kernel
			-	IPython 项目作为由 Jupyter 团队维护
		-	Kernel 支持详细情况：<https://github.com/jupyter/jupyter/wiki/Jupyter-kernels>
	-	*Jupyter* 部署、基础设施：支持多用户、云部署等
		-	[JupyterHub][JupyterHub]：多用户支持
		-	[NBViewer][NBViewer]：通过静态 HTML 分享 notebook
		-	[Binder][Binder]：将 Git 仓库转换为交互式 notebook
		-	[DockerSpawner][DockerSpawner]：在 *Docker* 容器中为 *JupyterHub* 部署 notebook
		-	[Docker-Stacks][DockerStacks]

> - <https://docs.jupyter.org/en/latest/projects/content-projects.html>

[JupyterNotebook]: https://jupyter-notebook.readthedocs.io/en/latest/
[JupyterConsole]: https://jupyter-console.readthedocs.io/en/latest/
[JupyterQtConsole]: https://jupyter.org/qtconsole/stable/
[JupyterHub]: https://jupyterhub.readthedocs.io/en/latest/
[NBViewer]: https://github.com/jupyter/nbviewer
[Binder]: https://mybinder.readthedocs.io/en/latest/index.html
[DockerSpawner]: https://github.com/jupyterhub/dockerspawner
[DockerStacks]: https://github.com/jupyter/docker-stacks

##	*Jupyter* 文件目录

-	Jupyter 文件分为 3 类
	-	数据文件：插件、[kernelspecs][kernelspecs]
		-	按以下优先级搜索：不同类型文件位于搜索路径目录中不同子目录下
			-	`JUPYTER_PATH`：数据文件额外搜索路径，优先级最高
			-	`JUPYTER_DATA_DIR`：缺省 `~/.local/share/jupyter`
			-	`{sys.prefix}/share/jupyter`
			-	`/usr/local/share/jupyter`
			-	`/usr/share/jupyter`
	-	运行时文件：日志、进程文件、连接文件
		-	运行时文件存储路径
			-	`JUPYTER_RUNTIME_DIR`：缺省 `$XDG_RUNTIME_DIR/jupyter`
	-	配置文件
		-	按以下优先级搜索
			-	`JUPYTER_CONFIG_DIR`：缺省 `~/.jupyter`
			-	`JUPYTER_CONFIG_PATH`：额外搜索路径
			-	`{sys.prefix}/etc/jupyter`
			-	`/usr/local/etc/jupyter`
			-	`/etc/jupyter`

```sh
$ jupyter --paths					# 列出全部文件目录
$ jupyter --data-dir				# 列出数据文件目录
$ jupyter --runtime-dir
$ jupyter --config-dir
```

[kernelspecs]: https://jupyter-client.readthedocs.io/en/latest/kernels.html#kernelspecs

###	配置文件

-	说明事项
	-	特定内核的配置文件由内核确定，不由 *Jupyter* 直接管理

####	Python 配置文件

```sh
$ jupyter <APP> --generate-config			# 生成默认 Python 配置文件，默认为 `jupyter_<APP>_config.py`
$ jupyter <APP> --<ITEM>=<VALUE>			# 命令行传参配置
```

-	可通过配置文件、命令行传参方式配置选项
	-	命令行传参会覆盖配置文件选项
		-	值参数名称即配置文件中属性
		-	部分常用选项有简称、标志参数
	-	可利用命令行生成默认配置文件
		-	配置文件为 Python 脚本，遵循 Python 语法

```python
c.NotebookApp.password = u'sha1:xxxxxxxx'			# 配置sha1格式密文，需要自己手动生成
c.NotebookApp.ip = "*"								# 配置运行访问ip地址
c.NotebookApp.open_brower = False					# 默认启动时不开启浏览器
c.NotebookApp.port = 8888							# 监听端口，默认
c.NotebookAPp.notebook_dir = u"/path/to/dir"		# Jupyter 默认显示羡慕路径
c.NotebookApp.certfile = u"/path/to/ssl_cert_file"	# SSL 证书
c.NotebookApp.keyfile = u"/path/to/ssl_keyfile"
```

-	说明事项
	-	Jupyter 远程访问被要求必须用 *https* 登陆、访问，即必须配置 *SSL* 证书
		-	可使用 `openssl` 工具生成所需的 *SSL* 证书
	-	`notebook.auth.passwd()` 函数可创建符合 *Jupyter* 格式要求的加密密钥
		-	也可以利用其他工具如 `hashlib` 生成，但需手动添加格式信息：算法名、参数

```shell
$ openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout key_name.key -out cert_name.pem
```

