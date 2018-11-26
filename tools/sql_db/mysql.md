#	Mysql/Mariadb安装配置

##	安装

###	Mysql

大部分系统常用源中包含mysql，直接使用自带的包管理工具安装
即可，对于源中无法找到mysql的系统可以访问[官网][mysql]获取
安装方法

```shell
$ sudo zypper install mysql-server mysql-client
```

####	CentOS7

CentOS7的常用源中即不含mysql，安装mysql则需要添加mysql源，
同样在[官网][mysql]中找到添加方式：

1.	[下载][mysql_rpm]的是RPM源rpm包
2.	`$ sudo yum localintall`安装后即添加源
3.	使用yum直接安装mysql，需要注意的是默认情况下只有最新版本
	mysql源是enabled，其他版本的需要`--enablerepo`指定或者
	`/etc/yum.repo.d`下修改文件

[mysql]: https://dev.mysql.com/downloads/ (mysql)
[mysql_rpm]: http://dev.mysql.com/get/mysql57-community-release-el7-8.noarch.rpm (mysql_rpm)

###	Mariadb

mariadb和mysql大部分兼容，添加了一些新的特性，是mysql的一个
开源分支，由mysql的作者维护，避免mysql在被Oracle收购后闭源。

-	大部分情况下，mariadb可以完全看作是mysql
-	甚至在某些系统中，mariadb服务有别名mysql
-	mariadb控制台命令也是`mysql`

##	配置

###	配置文件

-	`/etc/mysql/my.cnf`:mysql主配置文件
	-	mysql的此配置文件内包含有具体的配置
	-	mariadb此文件中则不包含具体配置，而是导入配置文件
	```toml
	!includedir /etc/mysql/conf.d/
	!includedir /etc/mysql/mariadb.cond.d/
	```

-	`~/.my.cnf`
	-	一般不存在，需要自行建文件，注意需要设置文件权限，
		只能root用户可写，否则mysql会忽略此配置文件
	-	mysqld服务启动需要root权限，因此`~`目录下的配置文件
		基本不可能影响mysql-server状态，即`[server]`下的配置
		基本是无效的

###	数据位置

-	数据库存放数据位置：`/var/lib/mysql/db_name/`

##	Mysql-Client

###	登陆

@todo
一个问题，我安装的mariadb，默认的root用户无法在一般用户账户
登陆，必须sudo才能正常登陆

####	参数登陆
```shell
$ mysql -h host -P port -u user -p
```
mysql不带参数启动则是默认`cur_user_name@localhost:3306`，
表示使用当前用户名作为用户名登陆，如果该用户设置密码，`-p`
参数不能省略

####	文件

```shell
$ mysql --defaults-file=file_name
```

文件内容格式类似于配置文件

```toml
[client]
host=
user=
password=
database=（可选）
```

####	注意

-	mysql中默认存在一个用户名为空的账户，只要在本地，可以
	不用账户、密码登陆mysql，因为这个账号存在，使用新建用户
	无法通过密码登陆，需要在
	```sql
	$ use mysql;
	$ delete from user where User="";
	$ flush privileges;
	```

###	Mysql交互命令

####	Show信息类

-	`SHOW DATABASES`：列出MySQLServer数据库。
-	`SHOW TABLES [FROM db_name]`：列出数据库数据表。
-	`SHOW TABLE STATUS [FROM db_name]`：列出数据表及表状态信息。
-	`SHOW COLUMNS FROM tbl_name [FROM db_name]`：列出资料表字段
	-	`DESC tbl_name`：同
-	`SHOW FIELDS FROM tbl_name [FROM db_name]`
-	`DESCRIBE tbl_name [col_name]`
-	`SHOW FULL COLUMNS FROM tbl_name [FROM db_name]`：列出字段及详情
-	`SHOW FULL FIELDS FROM tbl_name [FROM db_name]`：列出字段完整属性
-	`SHOW INDEX FROM tbl_name [FROM db_name]`：列出表索引
-	`SHOW STATUS`：列出 DB Server 状态
-	`SHOW VARIABLES [like pattern]`：列出 MySQL 系统环境变量
-	`SHOW PROCESSLIST`：列出执行命令。
-	`SHOW GRANTS FOR user`：列出某用户权限

####	User

-	`CREATE USER 'USER_NAME'@'ADDRESS' IDENTIFIED BY 'PASSWORD'`
	-	`IDENTIFIED BY PASSWORD`这个语法好像已经被丢弃了

-	`SET PASSWORD FOR 'USER_NAME'@'ADDRESS' = PASSWORD('NEW_PWD')`

-	`SET PASSWORD = PASSWORD('NEW_PWD')`：修改当前用户密码

-	`GRANT PRIVILEGES ON DB_NAME.TBL_NAME TO 'USER_NAME'@'ADDRESS'
	[WITH GRANT OPTION]`
	-	`WITH GRANT OPTION`：允许用户授权

-	`REVOKE PRIVILIEGES ON DB_NAME.TBL_NAME TO 'USER_NAME'@'ADDRES%'`

-	`DROP 'USER_NAME'@'ADDRESS'`

说明

-	revoke和grant中的权限需要一致才能取消相应授权
	-	`grant select`不能通过`revoke all`取消`select`
	-	`grant all`也不能通过`revoke select`取消`select`

-	特殊符号
	-	`%`：所有address，也可以是ip部分，如：111.111.111.%
		-	这个其实在sql语句中可以表示任意长度字符串
	-	`*`：所有数据库、表

####	Priviledges

|-----|-----|
|alter|alter table|
|alter routine|alter or drop routines|
|create|create table|
|create routine|create routine|
|create temporary table|create temporary table|
|create user|create, drop, rename users and revoke all privilieges|
|create view|create view|
|delete|delete|
|drop|drop table|
|execute|run stored routines|
|file|select info outfile and load data infile|
|index|create index and drop index|
|insert|insert|
|lock tables|lock tables on tables for which select is granted|
|process|show full processlist|
|reload|use flush|
|replicati on client|ask where slave or master server are|
|replicati on slave||
|select|select|
|show databases|show databases|
|show view|show view|
|shutdown|use mysqladmin shutdown|
|super|change master, kill, purge master logs,
	set global sql statements,	use mysqladmin
	debug command, create an extra connection
	even reach the maximum amount|
|update|update|
|usage|connect without any specific priviliege|

###	执行Sql脚本

-	shell内执行
	```sh
	mysql -h host -D db_name -u user -p < file_name.sql;
	```

-	mysql命令行执行
	```sql
	source file_name.sql;
	```

###	导入、导出数据

####	导入数据

-	shell内

-	mysql命令行内

	```sql
	load data [local] infile '/path/to/file' into table tbl_name
	fields terminated by 'sep_char'
	optionally enclosed by 'closure_char'
	escaped by 'esc_char'
	lines terminated by `\r\n`;
	```

	-	若`/path/to/file`不是绝对路径，则被认为是相对于当前
		数据库存储数据目录的相对路径，而不是当前目录
	-	关键字`local`表示从客户端主机导入数据，否则从服务器
		导入数据

####	导出数据

-	shell内

-	mysql命令行内

**注意**：远程登陆mysql时，两种方式导出数据不同，shell导出
数据可以导出至client，而mysql命令行内导出至server

##	Mysql-Server

###	数据库字符编码方式

####	查看

-	只查看**数据库编码**方式
	```sql
	show variables like "character_set_database;
	```

-	查看数据库**相关的编码**方式
	```sql
	show variables like "character%";
	```

	|-----|-----|
	|variable_name|value|
	|character_set_client|latin1|
	|character_set_connection|latin1|
	|character_set_database|latin1|
	|character_set_filesystem|binary|
	|character_set_results|latin1|
	|character_set_server|latin1|
	|character_set_system|utf8|
	|character_sets_dir|/usr/share/mysql/charsets/|

-	另一种查询数据库编码方式
	```sql
	show variables like "collation%";
	```

	|-----|-----|
	|Variable_name|Value|
	|collation_connection|utf8mb4_general_ci|
	|collation_database|utf8mb4_general_ci|
	|collation_server|utf8mb4_general_ci|

####	相关变量说明

-	`character_set_client`：客户端编码方式
-	`character_set_connection`：建立连接是所用编码
-	`character_set_database`：数据库编码
-	`character_set_results`：结果集编码
-	`character_set_server`：服务器编码

保证以上编码方式相同则不会出现乱码问题，还需要注意其他连接
数据库的方式不一定同此编码方式，可能需要额外指定

####	修改编码方式

#####	修改数据库默认编码方式

修改mysql配置文件（`/etc/mysql/my.cnf`）
	#todo
	mysql配置文件的优先级
	utf8mb4意义
```toml
[client]
default-character-set=utf8mb4
[mysqld]
default-character-set=utf8mb4
init_connect="SET NAMES utf8mb4"
```
重启mysql即可

#####	修改单个数据库

-	创建时
	```sql
	create database db_name character set utf8 collate utf8_general_ci;
	```
	```sql
	create database if not exists db_name defualt charater set utf8;
	```

#####	脚本、窗口

```sql
set names gbk;
```
只修改`character_set_client`、`character_set_connection`、
`character_set_results`编码方式，且只对当前窗口、脚本有效，
不影响数据库底层编码方式

