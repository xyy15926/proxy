---
title: Spark SQL
tags:
  - DataBase
  - Spark
categories:
  - DataBase
  - Spark
  - SQL
date: 2019-07-11 00:51:41
updated: 2021-08-02 17:32:31
toc: true
mathjax: true
comments: true
description: Spark SQL
---

##	Spark SQL

*Spark SQL*：结构化数据查询模块

-	内置JDBC服务器，通过JDBC API暴露Spark数据集，让客户程序
	可以在其上直接执行SQL查询

-	通过ETL工具从不同格式数据源装载数据，并运行一些
	Ad-Hoc Query

-	可以连接传统的BI、可视化工具至数据集

> - 前身*Shark*即为*Hive on Spark*，后出于维护、优化、
	性能考虑放弃
> - *Extraction Transformation Loading*：ETL

![sparksql_structure](imgs/spark_structure.png)

###	*sql.SQLContext*

```scala
import org.apache.spark.sql.{SQLContext, HiveContext}

class SQLContext{

	// 缓存使用柱状格式的表
	// Spark将仅仅浏览需要的列、自动压缩数据减少内存使用
	def cacheTable(tableName: String)

	// 将普通RDD转换为SchemaRDD
	def implicit createSchemaRDD(rdd: RDD): SchemaRDD

	// 载入parquet格式文件
	def parquetFile(fileName: String): SchemaRDD

	// 载入json格式文件
	def jsonFile(fileName: String): SchemaRDD
	def jsonRDD(rdd: RDD[String]): SchemaRDD

	// 执行SQL query
	def sql(query: String): SchemeRDD
}
```

> - `HiveContext`支持`SQLContext`支持功能的超集，增加在
	MetaStore发现表、利用HiveSQL写查询功能

##	`sql.SchemaRDD`

```c
class SchemaRDD{

	// 存储为parquet文件
	def saveAsParquetFile(fileName: String)

	// 注册为临时表，然后可以使用SQL语句查询
	def registerTempTable(tableName: String)

	// 打印表schema
	def printSchema()
}
```

在数据存储层面对数据进行结构化描述的schema

-	由SchemaRDD（上个版本）发展而来，在其上增加schema层
	，以便对各个数据列命名、数据类型描述

-	可以通过DF API把过程性处理、Relational Processing
	（对表格的选择、投影、连接等操作）集成

-	DF API操作是Lazy的，使得Spark可以对关系操作、数据处理
	工作流进行深入优化

-	结构化的DF可以通过调用DF API重新转换为无结构的RDD数据集

-	可以通过不同Data Source创建DF
	-	已经存在的RDD数据集
	-	结构化数据文件
	-	JSON数据集
	-	Hive表格
	-	外部数据库表

###	*Data Source*

数据源：通过DS API可以存取不同格式保存的结构化数据

-	Parquet
-	JSON
-	Apache Avro数据序列化格式
-	JDBC DS：可以通过JDBC读取关系型数据库

```scala
import org.apache.spark.sql.{SQLContext, StructType, StructField, Row}
import org.apache.spark.sql.HiveContext

val sqlContext = new SQLContext(sc)
import sqlContext.createSchemeRDD


case class Person(name: String, age: Int)

// 通过反射推断包含特定对象类型的RDD的模式
// 需要编写时已知模式
// 代码更简洁、工作更好
val people: RDD[Person] = sc.textFile("people.txt")
	.map(_.split(","))
	.map(p => Person(p(0), p(1).trim.toInt))


// 编程指定模式：构建模式，然后在已经存在的RDDs上使用
// 运行在运行期前不知道列、列类型情况下构造SchemaRDDs
val schemaString = "name age"
val people = sc.textFile("people.txt")
val schema = StructType(schemaString.split(" ")
	.map(fieldName => StructField(fieldName, StringType, true))
)
val rowRDD = people.map(_.split(","))
	.map(p => Row(p(0), p(1).trim))
val peopleSchemaRDD = sqlContext.applySchema(rowRDD, schema)
peopleSchemaRDD.registerTempTable("people")


// 查询语言集成
val teenagers = people.where("age >= 13").select("name")


people.registerTempTable("people")
val teenagers = sqlContext.sql("SELECT name FORM people WHERE age >= 13")


val apRDD = sc.parallelize(
	"""{"name": "Tom", "address": { "city": "Columbus", "state": "Ohio" }}""" :: Nil)
val anotherPeople = sqlContext.jsonRDD(apRDD)
```



