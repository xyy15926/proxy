#	Spark

##	`spark.streaming`

###	`StreamingContext`

```scala
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.Seconds
import org.apache.spark.streaming.StreamingContext._

def StreamingContext(?conf: SparkContext/SparkConf, ?slice: Int){

	// 开始接受、处理数据
	def start()
	// 结束处理过程
	def stop(?stop_spark_context=True)
	// 等待计算完成
	def awaitTermination()
}
```

```scala
val conf = new SparkConf().setAppName("app name").setMaster(master)
val ssc = new StreamingContext(conf, Seconds(1))
```

###	*Basic Sources*

```scala
// 套接字连接TCP源获取数据
def ssc.socketTextStream(?host: String, ?port: Int): DStream

// 文件流获取数据
def ssc.fileStream[keyClass, valueClass, inputFormatClass]
	(dataDirectory: String): DStream
def ssc.testFileStream(dataDirectory)

// 自定义actor流
def ssc.actorStream(actorProps: ?, actorName: String): DStream

// RDD队列流
def ssc.queueStream(queueOfRDDs: Seq[RDD])
```

###	*Checkpoint*

-	为保证流应用程序全天运行，需要checkpoint足够信息到容错
	存储系统，使得系统能从程序逻辑无关错误中恢复

	-	*metadata checkpointing*：流计算的定义信息，用于恢复
		worker节点故障
	-	*configuration*：streaming程序配置
	-	*DStream operation*：streaming程序操作集合
	-	*imcomplete batches*：操作队列中未完成批
	-	*data checkpointing*：中间生成的RDD，在有状态的转换
		操作中必须，避免RDD依赖链无限增长

-	需要开启checkpoint场合
	-	使用有状态转换操作：`updateStateByKey`、
		`reduceByKeyAndWindow`等
	-	从程序的driver故障中恢复

```scala
// 设置checkpoint存储信息目录
def ssc.checkpoint(?checkpointDirectory: String)
// 从checkpoint中恢复（若目录存在）、或创建新streaming上下文
def StreamingContext.getOrCreate(?checkPointDirectory: String, ?functionToCreateContext: () => StreamingContext)
```

```scala
def functionToCreateContext(): StreamingContext = {
	val conf = new SparkConf()
	val ssc = new StreamingContext(conf)
	// other streaming setting
	ssc.checkpoint("checkpointDirectory")
	ssc
}
```

##	`spark.sql`

```scala
// `HiveContext`支持`SQLContext`支持功能的超集
// 增加在MetaStore发现表、利用HiveSQL写查询功能
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

class SchemaRDD{

	// 存储为parquet文件
	def saveAsParquetFile(fileName: String)

	// 注册为临时表，然后可以使用SQL语句查询
	def registerTempTable(tableName: String)

	// 打印表schema
	def printSchema()
}
```

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


