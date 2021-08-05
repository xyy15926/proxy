---
title: SQL数据库Puzzles
categories:
  - Database
  - SQL DB
tags:
  - Database
  - SQL DB
  - Data Migration
date: 2019-03-21 17:27:37
updated: 2021-07-19 09:09:11
toc: true
mathjax: true
comments: true
description: SQL数据库Puzzles
---

##	数据迁移

###	直接查询、插入

####	同库

```sql
insert into dst_tb select * from src_tb;
insert into dst_tb(field1, field2, ...) select (field_a, field_b, ...) from src_tb;
```

####	异库、同服务器

```sql
insert into db1.dst_db select * from db2.src_db;
	# 插入已有表
create table db1.dst_tb as select * from db2.src_tb;
	# 创建表并插入数据
rename table src_db.src_tb to dst_db.dst_tb;
	# 重命名迁移完整表
```

####	异服务器

###	文件中介、跨实例

####	`.sql`

```shell
$ mysqldump [-u user] -p --single-transaction [--where=""] src_db src_tb > src_db.src_tb.sql
	# 导入数据
	# 加上`-d`仅导出表结构
$ mysql [-u user] -p dst_db < src_db.src_tb.sql
	# 导入数据
```

```sql
source src_db.src_tb.sql;
```

####	`.csv`

#####	`secure_file_priv`

`load data infile`和`into outfile`需要mysql开启
`secure_file_priv`选项，可以通过查看

```sql
show global variables like `%secure%`;
```

mysql默认值`NULL`不允许执行，需要更改配置文件

```cnf
[mysqld]
secure_file_priv=''
```

#####	本机Server

```sql
select * from src_tb into outfile file_name.csv
	fields terminated by ','
	optionally enclosed by '"'
	escaped by '"'
	lines terminated by '\r\n';
	# 导出至`.csv`

load data infile file_name.csv [replace] into table dst_tb(field1, field2, @dummy...)
	fields terminated by ','
	optionally enclosed by '"'
	escaped by '"'
	lines terminated by '\r\n';
	# 从`.csv`数据导入
	# 表结构不同时可以设置对应字段，多余字段`@dummy`表示丢弃
```

#####	异机Server

```shell
$ mysql -h host -u user -p src_db -N -e "select * from src_tb;" > file_name.csv
	# 只能通过*shell*查询并导出至文件
	# 需要`file`权限
	# `-N`：skip column names
```

```sql
load data local infile filename.csv;
	# 指定`local`则从*client*读取文件，否则从*server*读取
```

###	大表分块迁移

-	容易分块的字段
	-	自增id
	-	时间

###	注意




