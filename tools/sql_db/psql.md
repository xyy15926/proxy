#	Postgresql笔记

##	安装

###	包管理器安装

####	OpenSuSe

```sh
$ sudo zypper in postgresql postgresql-server
```

-	交互式客户端：`psql`
-	服务器：`postgres`

##	配置

###	初始化

1.	postgres安装完成后，会创建一个postgres用户，密码为空，
	为其设置密码
	```sh
	$ sudo passwd postgres
	```
	其用户目录默认是`/var/lib/pgsql`

2.	切换至postgres用户，创建数据库簇（cluster）（存放数据
	位置）
	```sh
	$ su postgres
	$ initdb -D /var/lib/pgsql/data
	```

###	启动服务

1.	切换至postgres用户
	```sh
	$ su postgres
	```

2.	通过`pg_ctl`工具启动服务
	```sh
	$ pg_ctl start -D /var/lib/pgsql/data
	```

###	Pg_ctl

`pg_ctl`是用于控制PostgreSQL服务器的工具，此工具基本需要在
postgres用户下才能使用

-	查看状态：`$ pg_ctl status -D /var/lib/pgsql/data`
-	

####	PGDATA

可以添加环境变量`PGDATA`避免使用`-D`参数指定数据库簇目录，
在postgres用户`.bashrc`中添加（注意是postgres用户目录下）

```sh
 # postgres database cluster directory
export PGDATA=/var/lib/pgsql/data
```

##	Psql

###	登陆

####	参数登陆

```shell
$ psql [-h host] [-p port] [-U user_name] [db_name]
```

-	`-h host`：localhost
-	`-p port`：5432
-	`-U user_name`：当前linux用户名
-	`[-d] db_name`：当前linux用户名

###	Psql交互命令

####	User

-	`CREATE ROLE user_name WITH priviledges PASSWORD 'pwd';`
-	`DROP user_name;`

####	元命令

`\`开头，也称反斜杠命令，由psql自己处理

#####	格式

`\`后跟命令动词、参数，其间使用空白字符分割

-	`\? [commands]`：元命令帮助
-	`\? options`：psql命令行选项帮助
-	`\? variables`：特殊变量帮助
-	`\h [clauses]`：SQL命令语法帮助（`*`表示全部）

#####	参数

-	单引号`'`
	-	包含空白时需用单引号包裹
	-	单引号中包含的参数的内容会进行类C的替换
		-	`\n`（换行）、`\digits`（八进制）
		-	包含单引号需要使用反斜杠转义

-	冒号`:`：不在引号中的冒号开头的参数会被当作psql变量

-	反勾号<code>`<\code>：参数内容被当作命令传给shell，
	输出（除去换行符）作为参数值

-	双引号<code>"<\code>
	-	遵循SQL语法规则，双引号保护字符不被强制转换为
		小写，且允许其中使用空白
	-	双引号内的双引号使用双引号转义

###	高级特性

####	变量

-	`=> \set foo bar`：设置变量（可以像php设置“变量 变量”
	`=> \set :foo bar`）
-	`=> \unset foo`：重置变量

#####	特殊变量

特殊变量是一些选项设置，在运行时可以通过改变变量的值、应用的
表现状态改变，不推荐改变这些变量的用途

-	`AUTOCOMMIT`：缺省为`on`，每个SQL命令完成后自行提交，此时
	需要输出`BEGIN`、`START TRANSACTION`命令推迟提交
-	`DBNAMW`：当前所连接数据库
-	`ENCODING`：客户端字符集编码

还有其他很多，详情查询手册

###	其他shell命令

postgres不仅仅提供psql交互环境作为shell命令，还提供了一些
可以直接在shell中运行的数据库命令，当然前提是当前linux登陆
用户在数据库中存在、有权限

-	`$ createdb db_name`
-	`$ dropdb db_name`
