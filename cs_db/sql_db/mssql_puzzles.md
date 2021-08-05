---
title: MSSQL Puzzles
categories:
  - Database
  - SQL DB
tags:
  - Database
  - SQL DB
  - Data Migration
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:08
toc: true
mathjax: true
comments: true
description: MSSQL Puzzles
---

##	访问其他数据库服务器

SQL默认阻止对组件`Ad Hoc Distributed Queries`的*STATEMENT*
`OpenRowSet/OpenDatasource`的访问，需要使用`sp_configure`
启用`Ad Hoc Distributed Queries`

-	开启`Ad Hoc Distributed Queries`

	```sql
	exec sp_configure 'show advanced options',1
	reconfigure
	exec sp_configure 'Ad Hoc Distributed Queries',1
	reconfigure
	```

-	关闭

	```sql
	exec sp_configure 'Ad Hoc Distributed Queries',0
	reconfigure
	exec sp_configure 'show advanced options',0
	reconfigure
	```

##	特殊语法

###	数据导入

-	mssql中换行符设置为`\n`表示的是`\r\n`，即永远无法单独
	指定`\n`或者`\r`，尽量使用ASCII码`0xXX`表示

	```sql
	> bulk insert tbl_name from /path/to/file with (FILEDTERMINATOR="|", ROWTERMINATOR="0x0a");
	```


