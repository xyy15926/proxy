#	MSSQL Puzzles

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
