---
title: Postgresql笔记
tags:
  - 工具
  - SQLDB
categories:
  - 工具
  - SQLDB
date: 2019-05-10 01:01:26
updated: 2019-05-10 01:01:26
toc: true
mathjax: true
comments: true
description: Postgresql笔记
---

##	安装

-	交互式客户端：`postgresql`
-	服务器：`postgres-server`
-	额外功能：`postgresql-contrib`
-	开发工具：`postgresql-devel`

###	OpenSuSe

```shell
$ sudo zypper in postgresql postgresql-server \
	postgresql-contrib postgresql-devel
```

###	CentOS

```shell
$ sudo yum install postgresql postgresql-server \
	postgresql-contrib postgresql-devel
```

####	其他版本

-	从中选择合适版本下载：[Postgres Yum repositories](https://yum.postgresql.org/repopackages.php)

	```shell
	$ wget https://download.postgresql.org/pub/repos/yum/9.6/redhat/rhel-7-x86_64/pgdg-centos96-9.6-3.noarch.rpm
	```

-	安装下载的RPM（依赖EPEL repo）

	```shell
	$ sudo yum install pgdg-centos96-9.6-3.noarch.rpm
	```

-	更新Yum、安装指定PG版本

	```shell
	$ sudo yum update
	$ sudo yum install postgresql96-sever postgresql96-contrib
	```

> - 安装的PG带有版本后缀，初始化、启动时注意

##	配置

-	postgres安装完成后，默认创建Linux用户

	-	用户密码为空，要为其设置密码以切换到其
		```shell
		$ sudo passwd postgres
		$ su - postgres
		```
	-	用户目录默认是`/var/lib/pgsql`
	-	很多命令可以切换到用户`postgres`直接执行

> - 初始化数据库簇后，默认创建数据库角色`postgres`、数据库
	`postgres`

###	初始化

-	创建新PostgreSQL数据库簇

	```shell
	$ sudo postgresql-setup initdb
		# 或
	$ sudo inidb -D /var/lib/pgsql/data
	```

	-	默认数据库存储路径为`/var/lib/pgsql/data`

-	开启PG密码认证：修改*host-based authentication*设置

	```shell
	# /var/lib/pgsql/data/pg_hba.conf
	# TYPE		DATABASE	USER	ADDRESS			MEHTOD
	host		all			all		127.0.0.1/32	md5
	host		all			all		::1/128			md5
	```

	-	替换默认`ident`为`md5`开启密码认证
	-	修改之后需要重启PG

-	修改`postgres`用户密码，以可以通过密码作为`postgres`连接
	数据库

	```shell
	$ su - postgres
	$ psql -d template1 -c "ALTER USER postgres with password '<passwd>'"
		# 也可以在数据库prompt中执行
	```

###	启动数据库

-	作为服务：`start`、`enable`PG

	```shell
	$ sudo systemctl start postgresql
	$ sudo systemctl enable postgresql
	```

-	作为普通程序启动：`pg_ctl`

	```shell
	$ su - postgres
	$ pg_ctl start -D /var/lib/pgsql/data
	```

##	*Roles*

PG使用概念*roles*处理认证、权限问题

-	角色相较于区分明显的用户、组概念更加灵活

-	`create user`和`create role`几乎完全相同
	-	`create user`：创建角色默认带`LOGIN`属性
	-	`create role`：创建角色默认不带`LOGIN`属性

###	权限

-	`SUPERUSER`/`NOSUPERUSER`：数据库超级用户
-	`CREATEDB`/`NOCREATEDB`：创建数据库
-	`CREATEUSER`/`NOCREATEUSER`
-	`CREATEROLE`/`NOCREATEROLE`：创建、删除普通用户角色
-	`INHERIT`/`INHERIT`：角色可以继承所属用户组权限
-	`LOGIN`/`NONLOGIN`：作连接数据库初始角色名
-	`REPLICATION`/`NOREPLICATION`：流复制时用到
-	`CONNECTION LIMIT connlimit`
-	[`ENCRYPTED`/`UNENCRYPTED`]`PASSWORD '<passwd>'`
-	`VALID UNTIL '<timestamp>'`
-	`IN ROLE <role_name>[, ...]`：角色所属用户组
-	`IN GROUP <role_name>[, ...]`
-	`ROLE <role_name>[, ...]`
-	`ADMIN <role_name>[, ...]`
-	`USER <role_name>[, ...]`
-	`SYSID uid`

####	角色赋权

```shell
psql> create role/user <name> [[with] <option> [...]];
	# 创建角色时直接赋权
psql> alter role/user <name> [[with] <option> [...]];
psql> grant connect on database <db_name> to <name>;
	# 修改已创建角色权限
```

####	组赋权

-	把多个角色归为组，通过给组赋权、撤销权限实现权限管理
-	PG中角色赋权是通过`inherit`方式实现的

```shell
psql> create role father login createdb createrole;
	# 创建组角色、赋权
psql> create role son1 inherit;
psql> grant father to son1;
	# 创建有`inherit`权限的用户、赋权
psql> create role son2 inherit in role father;
	# 创建用户时直接赋组权
```

###	认证方式

####	Peer Authentication

`peer`：从内核中获取操作系统中用户名，作为数据库角色名连接

-	默认连接同名数据库
-	信任Linux用户身份（认证），不询问密码
	-	即使`-W`强制输入密码，也不会检查密码正确性
-	只有`local`连接支持此方法

####	Trust Authentication

`trust`：允许任何数据库角色名的连接

-	信任任何连接、不询问密码
	-	只应该在操作系统层面上能提供足够保护下情况下使用
		-	文件系统权限：限制对Linux域套接字文件的访问
	-	适合单用户、本地连接

-	数据库、用户权限限制仍然存在

#### Ident Authentication

`ident`：从ident服务器中获取操作系统中用户名，用于连接数据库

-	仅在`TCP/IP`连接情况下支持
	-	若被指定给`local`连接，将使用`peer`认证

-	数据库服务器向客户端ident服务器询问“连接数据库的”用户，
	据此判断连
	-	此流程依赖于客户端完整性，若客户端机器不可信，则
		攻击者可以在113端口执行任何程序返回任何用户名
	-	故此认证方法只适合封闭网络，所以客户端机器都被严格
		控制
	-	有些ident服务器开启非标准选项导致返回的加密用户名，
		此选项应该关闭

	> - 基本每个类Unix操作系统都带有ident服务器，用于监听
		113端口TCP

#####	涉及配置

-	`map`：运行系统、数据库用户名之间映射

####	Password Authentication

Password认证：基于密码的认证方式

> - `password`：明文传输密码验证
> - `md5`：MD5-hashed传输密码o

-	`md5`可以避免密码嗅探攻击

-	`password`总应该尽量避免使用
	-	启用`db_user_namespace`特性时无法使用`md5`
	-	SSL加密连接下`password`也能安全使用

-	每个数据库的密码存储在`pg_authid`系统表中
	-	若用户没有设置密码，则存储的密码为`null`，密码验证
		也总是失败
	-	使用`create user`、`alter role`等SQL语句修改密码

####	GSSAPI Authentication

GSSAPI：定义在*RFC 2743*中的安全认证产业界标准协议

-	GSSAPI为支持其的系统自动提供认证
	-	认证本身是安全的，但是通过数据库连接的数据默认没有
		加密，除非使用SSL

> - PG中GSSAPI需要编译时启用支持

####	SSPI Authentication

`negotiate`：windows的安全认证技术

-	PG将尽可能使用*Kerberos*，并自动回滚为*NTLM*
-	仅服务器、客户端均在windows下或GSSAPI可用的情况下才能
	工作

> - 使用Kerberos情况下，SSPI、GSSAPI工作方式相同

#####	涉及配置

-	`include_realm`
-	`map`
-	`krb_realm`

####	Kerberos Authentication

Kerberos：适合公共网络上分布式计算的产业界标准安全认证系统

-	Kerberos提供不加密的语句、数据安全认证，若需要加密则使用
	SSL
-	PG支持Kerberos第5版，需要在build时开启Kerberos支持

#####	涉及配置

-	`map`
-	`include_realm`
-	`krb_realm`
-	`krb_server_hostname`

####	LDAP Authentication

LDAP：类似`password`，只是使用*LDAP*作为密码认证方法

#####	涉及配置

-	`ldapserver`
-	`ldapport`
-	`ldaptls`
-	`ldapprefix`
-	`ldapsuffix`
-	`ldapbasedn`
-	`ldapbinddn`
-	`ldapbindpasswd`
-	`ldapsearchattribute`

####	RADIUS Authentication

RADIUS：类似`password`，只是使用*RADIUS*作为密码认证方法

#####	涉及配置

-	`radiusserver`
-	`radiussecret`
-	`radiusport`
-	`radiusidentifier`

####	Certificate Authentication

Certificate：使用SSL客户多证书进行认证

-	所以只在SSL连接中可用
-	服务器要求客户端提供有效证书，不会向客户端传递密码prompt
	-	`cn`属性（*common name*）将回和目标数据库的用户名
		比较
	-	可通过名称映射允许`cn`属性和数据库用户名不同

#####	涉及配置

-	`map`：允许系统、数据库用户名之间映射

####	PAM Authentication

PAM：类似`password`，只是使用
*PAM(Pluggable Anthentication Modules)*作为密码认证机制

#####	涉及配置

-	`parmservice`：PAM服务名
	-	默认`postgresql`

##	`pg_ctl`

`pg_ctl`：用于控制PostgreSQL服务器的工具，此工具基本需要在
postgres用户下才能使用

-	查看状态：`$ pg_ctl status -D /var/lib/pgsql/data`

##	`psql`

###	Shell

####	连接数据库

```shell
$ psql [-h <host>] [-p <port>] [-U <user_name>] [[-d] <db_name>]
```

> - `-h`：缺省为`local`类型连接本地数据库
> > -	`local`、`host`连接类型对应不同认证方式
> > -	`-h localhost`和缺省对应不同`hba.conf`条目
> - `-p`：缺省端口`5432`
> - `-U/--user_name=`：缺省linux用户名
> - `[-d]/--database=`：当前linux用户名
> - `-W`：密码，`peer`、`trust`模式下无价值

####	Shell命令

-	postgres不仅仅提供psql交互环境作为shell命令，还提供可以
	直接在shell中运行的数据库命令

	-	当然前提是当前linux登陆用户在数据库中存在、有权限

```c
$ createdb <db_name> [-O <user_name>]
	# 创建数据库，设置所有权为`user_name`
$ dropdb <db_name>
	# 删除数据库
```

###	元命令

元命令：反斜杠命令，`\`开头，由psql自己处理

-	`\`后：跟命令动词、参数，其间使用空白字符分割

-	冒号`:`：不在引号中的冒号开头的参数会被当作psql变量

-	反点号：参数内容被当作命令传给shell，
	输出（除去换行符）作为参数值

-	单引号`'`：参数包含空白时需用单引号包o，其中包含的参数
	的内容会进行类C的替换

	-	`\n`（换行）、`\digits`（八进制）
	-	包含单引号需要使用反斜杠转义

-	双引号<code>\"<\code>
	-	遵循SQL语法规则，双引号保护字符不被强制转换为
		小写，且允许其中使用空白
	-	双引号内的双引号使用双双引号`""`转义

####	帮助

-	`\?`
	-	`[<commands>]`：元命令帮助
	-	`<options>`：psql命令行选项帮助
	-	`<variables>`：特殊变量帮助

-	`\h [<clauses>]`：SQL命令语法帮助（`*`表示全部）

####	展示信息

-	`\du`：查看用户权限
-	`\c <db_name> <name>`：以身份`name`访问数据库`db_name`
-	`\l[ist]`：查看数据库
-	`\dt`：展示当前数据库中表

####	变量

-	`\set foo bar`：设置变量
	-	可以像php设置“变量 变量”：`\set :foo bar`
-	`\unset foo`：重置变量

##	数据库变量

###	内部变量

####	特殊变量

特殊变量：一些选项设置，在运行时可通过改变变量的值、应用的
表现状态改变其，不推荐改变这些变量的用途

-	`AUTOCOMMIT`：缺省为`on`，每个SQL命令完成后自行提交，此时
	需要输出`BEGIN`、`START TRANSACTION`命令推迟提交
-	`DBNAMW`：当前所连接数据库
-	`ENCODING`：客户端字符集编码

> - 详情查询手册

###	环境变量

-	`PGDATA`：指定数据库簇（存放数据）目录

	```shell
	$export PGDATA=/var/lib/pgsql/data
	```

	-	默认`/var/lib/pgsql/data`
	-	`-D`命令行参数指定

