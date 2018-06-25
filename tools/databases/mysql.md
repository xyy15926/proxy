mysql安装配置

1.	安装

	a.	安装mysql：建议访问https://dev.mysql.com/downloads/获取更多信息 

		1.	在mysql官网下载对应系统版本、mysql版本的安装包

		2.	对于centos7下载的是RPM源rpm包，yum localintall安装后即可直接使用yum直接安装mysql，需要注意的是默认情况下只有最新版本的mysql源是enbabled，其他版本的需要--enablerepo指定或者是在/etc/yum.repo.d下修改文件

		3.	根据某些网站上信息，可以通过wget直接从：
			http://dev.mysql.com/get/mysql57-community-release-el7-8.noarch.rpm
			下载RPM源安装包

	b.	安装mariadb：centos7官方源就包含mariadb

		1.	mariadb和mysql大部分兼容，添加了一些新的特性，是mysql的一个开源分支，由mysql的作者维护，避免mysql在被Oracle收购后闭源

		2.	注意安装时需要安装mariadb和mariadb-server

			a.	有可能安装时没有启动服务，需要systemctl手动启动

			c.	mariadb进入控制台命令是mysql

		3.	mariadb安装完成后

			a.	第一个用户是：{安装用户}@'localhost'，密码为空

			b.	mysql默认情况下就是：{当前用户}@'localhost'登陆，所以安装后直接执行mysql命令就可以以当前用户名登陆

	c.	安装mysql-workbench

		1.	mysql源上的mysql-workbench是最新的版本6.3.10，直接安装可能会强退

		2.	从mysql官网下载的6.2.5使用yum localinstall安装只会安装部分依赖，无法打开

		3.	在命令行启动mysql-workbench就会显示缺少

			a.	libzip.so.2：libzip

			b.	libtinyxml：tinyxml-devel（依赖tinyxml）

2.	配置

	1.	mariadb-server（mysql-server）配置

	2.	mysql-client配置

		a.	mysql --defaults-file=file_name：按照指定文件连接数据库

			1.	默认情况下这个参数应该是有默认值，应该是/etc/my.conf，即连接localhost数据库

			2.	目前不是很清楚这个配置文件的全部功能

				a.	设置[client]连接某个数据库

					[client]
					host=
					user=
					password=
					database=（可选）

3.	mysql（mariadb启动命令也是mysql）控制台常用命令

	a.	show信息类

		1.	SHOW DATABASES：列出MySQLServer数据库。

		2.	SHOW TABLES [FROM db_name]：列出数据库数据表。

		3.	SHOW TABLE STATUS [FROM db_name]：列出数据表及表状态信息。

		4.	SHOW COLUMNS FROM tbl_name [FROM db_name]：列出资料表字段

		5.	SHOW FIELDS FROM tbl_name [FROM db_name]，DESCRIBE tbl_name [col_name]。

		6.	SHOW FULL COLUMNS FROM tbl_name [FROM db_name]：列出字段及详情

		7.	SHOW FULL FIELDS FROM tbl_name [FROM db_name]：列出字段完整属性

		8.	SHOW INDEX FROM tbl_name [FROM db_name]：列出表索引。

		9.	SHOW STATUS：列出 DB Server 状态。

		10.	SHOW VARIABLES：列出 MySQL 系统环境变量。

		11.	SHOW PROCESSLIST：列出执行命令。

		12.	SHOW GRANTS FOR user：列出某用户权限

	b.	user

		1.	create user 'user_name'@'address' identified by 'password'

		2.	set password for 'user_name'@'address' = password('new_pwd')
			set password = password('new_pwd')：修改当前用户密码

		3.	grant privileges on db_name.tbl_name to 'user_name'@'address'
				(with grant option)

		4.	revoke privilieges on db_name.tbl_name to 'user_name'@'addres%'

		5.	drop 'user_name'@'address'

		6.	说明

			a.	with grant option：允许用户授权

			b.	revoke和grant中的权限需要一致才能取消相应授权

				1.	grant select不能通过revoke all取消select

				2.	grant all也不能通过revoke select取消select

			c.	特殊符号

				1.	%：所有address

				2.	*：所有数据库、表

			d.	priviledges,

				1.	alter				alter table
				2.	alter routine		alter or drop routines
				3.	create				create table
				4.	create routine		create routine
				5.	create temporary table		create temporary table
				6.	create user			create, drop, rename users and revoke all privilieges
				7.	create view			create view
				8.	delete				delete
				9.	drop				drop table
				10.	execute				run stored routines
				11.	file				select info outfile and load data infile
				12.	index				create index and drop index
				13.	insert				insert
				14.	lock tables			lock tables on tables for which select is granted
				15.	process				show full processlist
				16.	reload				use flush
				17.	replicati on client	ask where slave or master server are
				18.	replicati on slave	
				19.	select				select
				20.	show databases		show databases
				21.	show view			show view
				22.	shutdown			use mysqladmin shutdown
				23.	super				change master, kill, purge master logs,
										set global sql statements,	use mysqladmin
										debug command, creat an extra connection 
										even reach the maximum amount
				24.	update				update
				25.	usage				connect without any specific priviliege
