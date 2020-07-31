---
title: SQL语法
tags:
  - 工具
  - SQLDB
categories:
  - 工具
  - SQLDB
date: 2019-07-10 00:48:33
updated: 2019-07-10 00:48:33
toc: true
mathjax: true
comments: true
description: SQL语法
---

##	基本操作

```sql
SELECT			<field>, DISTINCT <field>
INTO			<new_tbl> [IN <other_db>]
FROM			<tbl>
WHERE			<field> <OP> <value>/<field>
ORDER BY		<field> [ASC/DESC];

INSERT INTO		<tbl>[<field>]
VALUES			(<value>);

UPDATE			<tbl>
SET				<field> = <value>
WHERE			<field> <OP> <value>/<field>;

DELETE
FROM			<tbl>
WHERE			<field> <OP> <value>/<field>;
```

###	数据库、表、索引、视图

####	创建

```sql
!-- 创建数据库
CREATE DATABASE <db_name>;

!-- 创建表
CREATE TABLE <tbl>(
	<field>		<dtype>,
	<field>		<dtype>		<CSTRT>,					!-- MSSQL、ORACLE
	<CSTRT>		(<field>,),								!-- MySQL
	CONSTRAINT [<cstrt_name>] <cstrt> (<field>,)		!-- common
)

!-- 创建索引
CREATE INDEX <index_name>
ON <tbl> (<field>);

!-- 创建视图
CREATE VIEW <view_name> AS
<select_expr>;
```

#####	自增字段

```sql
!-- MSSQL
<field> <dtype> IDENTITY
!-- MySQL
<field> <dtype> AUTO_INCREMENT
!-- ORACLE：创建自增序列，调用`.nextval`方法获取下个自增值
CREATE SEQUENCE <seq>
MINVALUE <min>
START WITH <value>
INCREMENT BY <step>
CACHE <cache>				!-- 提高性能
```

####	丢弃

```sql
!-- 丢弃索引
!-- MSSQL
DROP INDEX <tbl>.<index_name>;
!-- ORACLE
DROP INDEX <index_name>;
!-- MySQL
ALTER TABLE <tbl>
DROP INDEX <index_name>;

!-- 丢弃表/数据
DROP TABLE <tbl>;
TRUNCATE TABLE <tbl>;

!-- 丢弃数据库
DROP DATABASE <db_name>;

!-- 丢弃视图
DROP VIEW <view>;
```

####	修改表

```sql
!-- 添加列
ALTER TABLE <tbl>
ADD <field> <dtype>;

!-- 删除列
ALTER TABLE <tbl>
DROP COLUMN <field>;

!-- 修改类型
ALTER TABLE <tbl>
ALTER COLUMN <field> <dtype>;
```

###	关键字

####	*TOP*

-	MSSQL：`SELECT TOP <num>/<num> PERCENT *`
-	MYSQL：`LIMIT <num>`
-	ORACLE：`WHERE ROWNUM <= <num>`

####	*Alias*

-	`AS`：指定行、列别名

####	*Join*

-	`[INNER] JOIN`
-	`LEFT JOIN`
-	`RIGHT JOIN`
-	`FULL JOIN`

####	*Union*

-	`UNION`：合并`SELECT`结果集
	-	要求结果集中列数量、类型必须相同

####	*NULL*

-	`IS [NOT] NULL`：比较是否为`NULL`值

> - 比较符无法测试`NULL`值

###	符号

####	运算符

-	`=`：有些方言可以使用`==`
-	`<>`：有些方言可以使用`!=`
-	`>`
-	`<`
-	`>=`
-	`<=`
-	`BETWEEN <value> AND <value>`
-	`[NOT] IN (<value>)`
-	`[NOT] LIKE <pattern>`
	-	`%`：匹配0个、多个字符
	-	`_`：匹配一个字符
	-	`[<char>]`：字符列中任意字符
	-	`^[<char>]/[!<char>]`：非字符列中任意字符

####	逻辑运算

-	`AND`
-	`OR`

####	符号

-	`'`：SQL中使用单引号包括文本值
	-	大部分方言也支持`"`双引号

###	数据类型

####	MySQL

|TEXT类型|描述|
|-----|-----|
|`CHAR([<size>])`||
|`VARCHAR([<size>])`||
|`TINYTEXT`||
|`LONGTEXT`||
|`MEDIUMITEXT`||
|`BLOB`||
|`MEDIUMBLOB`||
|`LONGBLOB`||
|`ENUM(<val_list>)`||
|`SET`||

|NUMBER类型|描述|
|-----|-----|
|`TINYINT([<size>])`||
|`SMALLINT([<size>])`||
|`MEDIUMINT([<size>])`||
|`INT([<size>])`||
|`BIGINT([<size>])`||
|`FLOAT([<size>])`||
|`DOUBLE([<size>])`||
|`DECIMAL([<size>])`||

|DATE类型|描述|
|-----|-----|
|`DATE()`||
|`DATETIME()`||
|`TIMSTAMP()`||
|`TIME()`||
|`YEAR()`||

####	MSSQL

|ASCII类型|描述|
|-----|-----|
|`CHAR([<size>])`||
|`VARCHAR([<size>])`||
|`TEXT`||

|UNICODE类型|描述|
|-----|-----|
|`CHAR([<size>])`||
|`VARCHAR([<size>])`||
|`text`||

|BINARY类型|描述|
|-----|-----|
|`bit`||
|`binary([<n>])`||
|`varbinary([<n>])`||
|`image`||

|NUMBER类型|描述|
|-----|-----|
|`TINYINT`||
|`SMALLINT`||
|`MEDIUMINT`||
|`INT`||
|`BIGINT`||
|`DECIMAL(p, s)`||
|`FLOAT([<n>])`||
|`REAL`||
|`SMALLMONEY`||
|`MONEY`||

|DATE类型|描述|
|-----|-----|
|`DATETIME`||
|`DATETIME2`||
|`SMALLDATETIME`||
|`DATE`||
|`TIME`||
|`DATETIMEOFFSET`||
|`TIMESTAMP`||

###	约束

-	建表时添加约束
	-	MSSQL、ORACLE：可直接在字段声明后添加约束
	-	MySQL：需独立指定约束

-	向已有表添加约束
	-	可以添加匿名、具名约束
	-	MSSQL、ORACLE：有`COLUMN`关键字

-	删除约束
	-	MySQL：使用约束关键字指定
	-	MSSQL、ORACLE：使用`CONSTRAINT`关键字指定

####	`NOT NULL`

```sql
<field> <dtype> NOT NULL
```

####	`DEFAULT`

`DEFAULT`

```sql
!-- 建表
<field> <dtype> DEFAULT <value>

!-- 已有表添加
!-- MySQL
ALTER TABLE <tbl>
ALTER <field> SET DEFAULT <value>;
!-- MSSQL、ORACLE
ALTER TABLE <tbl>
ALTER COLUMN <field> SET DEFAULT <value>;

!-- 删除
!-- MySQL
ALTER TABLE <tbl>
ALTER <field> DROP DEFAULT;
!-- MSSQL、ORACLE
ALTER TABLE <tbl>
ALTER COLUMN <field> DROP DEFAULT;
```

####	`UNIQUE`

`UNIQUE`

```sql
!-- 建表
!-- MySQL、MSSQL、ORACLE
CONSTRAINT [<cstrt_name>] UNIQUE (<field>,)
!-- MySQL
UNIQUE [KEY] [<cstrt_name>] (<field>)
!-- MSSQL、ORACLE
<field> <dtype> UNIQUE

!-- 已有表添加
!-- MySQL、MSSQL、ORACLE
ALTER TABLE <tbl>
ADD UNIQUE(<field>);
ALTER TABLE <tbl>
ADD CONSTRAINT <cstrt_name> UNIQUE(<field>,);

!-- 删除
!-- MySQL
ALTER TABLE <tbl>
DROP INDEX <cstrt_name>;
!-- MSSQL、ORACLE
ALTER TABlE <tbl>
DROP CONSTRAINT <cstrt_name>;
```

####	`PRIMARY KEY`

`PRIMARY KEY`

```sql
!-- 建表
!-- MySQL、MSSQL、ORACLE
CONSTRAINT [<cstrt_name>] PRIMARY KEY (<field>,)
!-- MYSQL
PRIMARY KEY (<field>,)
!-- MSSQL、ORACLE
<field> <dtype> PRIMARY KEY


!-- 已有表添加
!-- MySQL、MSSQL、ORACLE
ALTER TABLE <tbl>
ADD PRIMARY KEY (<field>,);
ALTER TABLE <tbl>
ADD CONSTRAINT <cstrt_name> PRIMARY KEY (<field>,);

!-- 删除
!-- MySQL
ALTER TABLE <tbl>
DROP PRIMARY KEY;
!-- MSSQL、ORACLE
ALTER TABLE <tbl>
DROP CONSTRAINT <cstrt_name>;
```

####	`FOREIGN KEY`

`FOREIGN KEY`

```sql
!-- 建表
!-- MySQL、MSSQL、ORACLE
CONSTRAINT [<cstrt_name>] FOREIGN KEY (<field>,)
REFERENCES <tbl>(<field>,)
!-- MYSQL
FOREIGN KEY (<field>,)
REFERENCES <tbl>(<field>,)
!-- MSSQL、ORACLE
<field> <dtype> FOREIGN KEY
REFERENCES <tbl>(<field>,)


!-- 已有表添加
!-- MySQL、MSSQL、ORACLE
ALTER TABLE <tbl>
ADD FOREIGN KEY (<field>,)
REFERENCES <tbl>(<field>,);
ALTER TABLE <tbl>
ADD CONSTRAINT <cstrt_name> FOREIGN KEY (<field>,)
REFERENCES <tbl>(<field>);

!-- 删除
!-  MySQL
ALTER TABLE <tbl>
DROP FOREIGN KEY <cstrt_name>;
!-- MSSQL、ORACLE
ALTER TABLE <tbl>
DROP CONSTRAINT <cstrt_name>;
```

####	`CHECK`

`CHECK`

```sql
!-- 建表
!-- MySQL、MSSQL、ORACLE
CONSTRAINT [<cstrt_name>] CHECK(<condition>)
!-- MySQL
CHECK (condition)
!-- MSSQL、ORACLE
<field> <dtype> CHECK(<condition>)

!-- 已有表添加
!-- MySQL、MSSQL、ORACLE
ALTER TABLE <tbl>
ADD CHECK (condition);
ALTER TABLE <tbl>
ADD CONSTRAINT <cstrt_name> CHECK (condition);

!-- 删除
!-- MySQL
ALTER TABLE <tbl>
DROP CHECK <cstrt_name>;
!-- MSSQL、ORACLE
ALTER TABLE <tbl>
DROP CONSTRAINT <cstrt_name>;
```

##	内建函数

###	*Date*

-	MySQL
	-	`NOW()`
	-	`CURDATE()`
	-	`CURTIME()`
	-	`DATE()`
	-	`EXTRACT()`
	-	`DATE_ADD()`
	-	`DATE_SUB()`
	-	`DATE_DIFF()`
	-	`DATE_FORMAT()`

-	MSSQL
	-	`GETDATE()`
	-	`DATEPART()`
	-	`DATEADD()`
	-	`DATEDIFF()`
	-	`CONVERT()`

###	*NULL*

-	MSSQL
	-	`ISNULL(<field>, <replacement>)`

-	ORACLE
	-	`NVL(<field>, <repalcement>)`

-	MySQL
	-	`IFNULL(<field>, <replacement>)`
	-	`COALESCE(<field>, <replacement>)`

###	*Aggregate*聚集函数

###	*Scalar*


