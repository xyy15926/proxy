---
title: 
categories:
  - Tools
  -	Windows Tools
tags:
  - Tools
  -	Windows
  -	Office
date: 2020-10-29 10:59:50
updated: 2020-10-29 10:59:50
toc: true
mathjax: true
description: 
---

##	插入元素

###	插入网页

-	WebView加载项：可以在ppt应用市场中获取
	-	只支持`https`页面
		-	本地页面都不支持
		-	尝试自建https服务器（自签发证书）失败
	-	可以在编辑状态查看页面效果

	> - 在OFFICE2010及以前不可用

-	`Microsoft Web Browser`控件
	-	调用IE渲染页面，因此网页对IE的兼容性很重要
	-	控件不会自动加载网页，需要通过VB通过触发事件调用其
		`Navigate2`方法加载网页，所以只有在ppt播放页面才能
		看到实际效果
		```vb
		// 页面切换事件
		// 注意不要`Private Sub`，否则事件不会被触发
		// 若想手动触发可以使用button控件的`CommandButton<X>_Click`事件
		Sub OnSlideShowPageChange()
			Dim FileName As String
			FileName = "<FILENAME>"
			// `WebBrowser1`：控件名称，唯一（单个slide内）标识控件
			// `ActivePresentation.PATH`：当前工作目录（未保存文件返回空），
			//		浏览器默认`http://`协议
			// `Navigate`方法可能会无法加载
			WebBrowser1.Navigate2(ActivePresentation.PATH + "/" + "<FILENAME>")
		End Sub
		```





