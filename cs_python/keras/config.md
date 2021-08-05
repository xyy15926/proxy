---
title: Keras 安装配置
categories:
  - Python
  - Keras
tags:
  - Python
  - Keras
  - Configuration
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:07
toc: true
mathjax: true
comments: true
description: Keras按照配置
---

##	Keras配置文件

###	`$HOME/.keras/keras.json`

```json
{
	"image_data_format": "channel_last",
		# 指定Keras将要使用数据维度顺序
	"epsilon": 1e-07,
		# 防止除0错误数字
	"flaotx": "float32",
		# 浮点数精度
	"backend": "tensorflow"
		# 指定Keras所使用后端
}
```

