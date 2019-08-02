---
title: Python注意事项
tags:
  - Python
categories:
  - Python
date: 2019-03-21 17:27:15
updated: 2019-02-17 11:57:07
toc: true
mathjax: true
comments: true
description: Python注意事项
---

##	原生库

###	`json`

-	注意对象是否*serializable*
	-	一般除了原生`dict`、`list`没有其他的
	-	包括NDA也不能序列化
