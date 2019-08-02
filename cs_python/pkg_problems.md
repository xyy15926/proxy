---
title: Python包问题
tags:
  - Python
categories:
  - Python
date: 2019-03-21 17:27:15
updated: 2019-02-28 11:09:27
toc: true
mathjax: true
comments: true
description: Python包问题
---

##	Shadowsocks

### Openssl

Openssl升级到1.1.0以上版本后，shadowsocks报错

> - undefined symbol: EVP_CIPHER_CTX_cleanup

-	原因：Openssl 1.1.0版本中，废弃了
	`EVP_CIPHER_CTX_cleanup`函数，使用
	`EVP_CIPHER_CTX_reset`代替

	-	两个函数均是用于释放内存，只是释放时机稍有不同

-	解决方法：修改`/path/to/python/site-packages/shadowsocks/crypto/openssl.py`，
	将其中`cleanup`替换为`reset`函数
