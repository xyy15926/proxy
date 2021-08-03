---
title: Nginx 使用、配置
categories:
  - Linux
tags:
  - Linux
  - Tool
  - Web Server
  - Nginx
date: 2020-10-28 18:46:33
updated: 2021-07-16 17:39:05
toc: true
mathjax: true
description: 
---

##	配置

> - 配置文件夹为`/etc/nginx`

###	基本配置

####	`nginx.conf`

`nginx.conf`：Nginx主配置文件，包含“全部配置”

-	默认`include`
	-	`modules-enabled/*.conf`：已启用模块
	-	`mime.types`：代理文件类型
	-	`conf.d/*.conf`：服务器设置
	-	`sites-enabled/*`：已启用站点

-	`user`表示启动nginx进程的用户
	-	键值默认为`www-data`
	-	可修改为其他用户，使nginx具有访问某些文件的权限

	> - 若是在nginx启动之后修改，需修改`/var/log/nginx`中
		log文件的属主，否则无法正常log

####	http服务器设置

```conf
server{
	listen 8080;
	server_name localhost;
	location / {
		root /home/xyy15926/Code;
		autoindex on;
		autoindex_exact_size off;
		autoindex_localtime on;
	}
}
```

-	`root`：需要nginx进程可访问，否则*403 Forbidden*
-	`autoindex`：自动为文件生成目录
	-	若目录设置`index index.XXX`，不设置`autoindex`访问
		目录则会*403 Forbidden*

####	https服务器

```conf
server{
	listen 443 ssl;
	server_name localhost;
	ssl_certificate /home/xyy15926/Code/home_config/nginx/localhost.crt;
	ssl_certificate_key /home/xyy15926/Code/home_config/nginx/localhost.key;
	ssl_session_timeout 5m;
	ssl_protocols TLSv1 TLSv1.1 TLSv1.2;
	ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:HIGH:!aNULL:!MD5:!RC4:!DHE;
	ssl_prefer_server_ciphers on;
	location / {
		root /home/xyy15926/Code/statics;
		autoindex on;
		autoindex_exact_size off;
		autoindex_localtime on;
	}
}
```

-	`ssl_certificate`、`ssl_certificate_key`：SSL证书、私钥

	```sh
	# 个人签发https证书，个人一般不会是受信任机构，所以还需要
	# 将证书添加进受信任机构（但是windows下有些有问题）
	openssl req -x509 -out localhost.crt -keyout localhost.key \
		-newkey rsa:2048 -nodes -sha256 \
		-subj '/CN=localhost' -extensions EXT -config <( \
		printf "[dn]\nCN=localhost\n[req]\ndistinguished_name = dn\n[EXT]\nsubjectAltName=DNS:localhost\nkeyUsage=digitalSignature\nextendedKeyUsage=serverAuth")
	```

###	运行

```sh
 # Nginx启动、重启动
$ /etc/init.d/nginx start
$ /etc/init.d/nginx restart
```


