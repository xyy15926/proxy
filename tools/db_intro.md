#	数据库背景

##	数据库术语

###	数据库结构

-	数据源：多源数据继承
-	数据仓库继承工具：FTL工具、MapReduce
-	数据仓库服务器：列存储数据库引擎
-	数据即使：数据仓库的数据子集、聚集数据继
-	OLAP服务器：提供多维数据视图
-	前台数据分析工具
	-	报表工具
	-	多维分析工具
	-	数据挖掘工具

###	数据库类型

-	传统关系型数据库
-	MPP大规模并行处理数据库
-	NoSQL数据库
-	图数据库
-	NewSQL数据库
-	GPU数据库

##	数据查询

-	Project Pushdown：投影下推
	-	只读取、查询需要的**列**
	-	减少每次查询的IO数据量

-	Predicate Pushdown：谓词下推
	-	将过滤条件尽快执行，跳过不满足条件**行**

##	数据压缩算法

-	Run Length Encoding：重复数据
-	Delta Encoding：有序数据集
-	Dictionary Encoding：小规模数据集合
-	Prefix Encoding：字符串版Delta Encoding

