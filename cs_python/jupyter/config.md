---
title: Jupyter 常用基础
categories:
  - Python
  - Jupyter
tags:
  - Python
  - Jupyter
  - Configuration
date: 2019-03-21 17:27:37
updated: 2021-08-02 11:11:29
toc: true
mathjax: true
comments: true
description: Jupyter常用基础
---

##	配置文件

###	生成配置文件

```
$ jupyter notebook --generate-config
	# 生成配置文件，默认为`~/jupyter/jupyter-notebook-config.py`
```

###	修改配置文件

```python
c.NotebookApp.password = u'sha1:xxxxxxxx'
	# 配置sha1格式密文，需要自己手动生成
c.NotebookApp.ip = "*"
	# 配置运行访问ip地址
c.NotebookApp.open_brower = False
	# 默认启动时不开启浏览器
c.NotebookApp.port = 8888
	# 监听端口，默认
c.NotebookAPp.notebook_dir = u"/path/to/dir"
	# jupyter默认显示羡慕路径
c.NotebookApp.certfile = u"/path/to/ssl_cert_file"
c.NotebookAppp.keyfile = u"/path/to/ssl_keyfile"
	# jupyter使用ssl证书文件
	# 一般只有使用自己生成证书
```

####	生成密文

```python
from notebook.auth import passwd
passwd()
	# 输入密码两次，然后就会返回对应密文，写入配置文件
```
###	远程访问

jupyter必须使用https登陆、访问

-	因此服务端必须有ssl证书，
	-	自己生成的证书默认浏览器不认可，浏览器访问会高危，
		高级进入即可
	```shell
	openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout key_name.key -out cert_name.pem
	```

-	访问时也必须加上`https://`前缀，一般默认`http://`前缀
	会报服务器无响应（没配置重定向，应该是）
