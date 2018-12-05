#	HDFS

##	HDFS设计模式

-	数据读取、写入
	-	HDFS一般存储不可更新的文件，只能对文件进行数据追加
		，也不支持多个写入者的操作
	-	认为一次写入、多次读取是最高效的访问模式
	-	namenode将metadata存储在内存中，所以文件系统所能
		存储的文件总数受限于NameNode的内存

-	适合模式
	-	每次分析都涉及数据集大部分数据甚至全部，因此读取整个
		数据集的大部分数据、甚至全部，因此读取整个数据集的
		时间比读取第一条记录时间延迟更重要
	-	HDFS不适合要求地时间延迟的数据访问应用，HDFS是为
		高数据吞吐量应用优化的，可能会提高时间延迟

-	硬件：HDFS无需高可靠硬件，HDFS被设计为即使节点故障也能
	继续运行，且不让用户察觉

###	数据块

和普通文件系统一样，HDFS同样也有块的概念，默认为64MB

-	HDFS上的文件也被划分为块大小的多个chunk，作为独立的存储
	单元，但是HDFS中小于块大小的文件不会占据整个块空间

-	对分布式文件系统块进行抽象的好处
	-	文件大小可以大于网络中任意一个磁盘的容量
	-	使用抽象块而非整个文件作为存储单元，简化了存储子系统
		的设计
		-	简化存储管理，块大小固定，计算磁盘能存储块数目
			相对容易
		-	消除了对元素见的顾虑，文件的元信息（权限等），
			不需要和块一起存储，可由其他系统单独管理

-	块非常适合用于数据备份，进而提供数据容错能力、提高可用性


###	NameNode

HDFS系统中的管理者

-	集中存储了HDFS的元信息Metadata
	-	维护文件系统的文件树、全部的文件和文件夹的元数据
	-	管理文件系统的Namespace：创建、删除、修改、列出所有
		文件、目录

-	执行数据块的管理操作
	-	把文件映射到所有的数据块
	-	创建、删除数据块
	-	管理副本的Placement、Replication

-	负责DataNode的成员管理
	-	接受DataNode的Registration
	-	接受DataNode周期性的Heart Beat

Hadoop上层模块，根据NameNode上的元信息，就可以知道每个数据块
有多少副本、存放在哪些节点上，据此分配计算任务，通过
**Move Computation to Data**，而不是移动数据本身，减少数据
移动开销，加快计算过程

####	Metadata的保存

为了支持高效的存取操作，NameNode把所有的元信息保存在主内存，
包括文件和数据块的命名空间、文件到数据块的映射、数据块副本
的位置信息。文件命名空间、文件到数据块的映射信息也会持久化
到NameNode本地文件系统

-	FsImage：命名空间镜像文件，保存整个文件系统命名空间、
	文件到数据块的映射信息
-	EditLog：编辑日志文件，是一个Transaction Log文件，记录了
	对文件系统元信息的所有更新操作：创建文件、改变文件的
	Replication Factor

NameNode启动时，读取FsImage、EditLog文件，把EditLog的所有
事务日志应用到从FsImage文件中装载的旧版本元信息上，生成新的
FsImage并保存，然后截短EditLog

####	NameNode可恢复性

#####	多个文件系统备份

备份文件系统元信息的持久化版本

-	在NameNode写入元信息的持久化版本时，同步、atomic写入多个
	文件系统（一般是本地磁盘、mount为本地目录的NFS）

#####	Secondary NameNode

运行Secondary NameNode：负责周期性的使用EditLog更新
FsImage，保持EditLog在一定规模内

-	Seconadary NameNode保存FsImage、EditLog文件副本，
	每个一段时间从NameNode拷贝FsImage，和EditLog文件进行
	合并，然后把更新后的FsImage复制回NameNode

-	若NameNode宕机，可以启动其他机器，从Secondary
	NameNode获得FsImage、EditLog，恢复宕机之前的最新的
	元信息，当作新的NameNode，也可以直接作为主NameNode

-	Secondary NameNode保存的出状态总是滞后于主节点，需要
	从NFS获取NameNode部分丢失的metadata

-	Secondary NameNode需要运行在另一台机器，需要和主
	NameNode一样规模的CPU计算能力、内存，以便完成元信息
	管理

想要从失效的NameNode恢复，需要启动一个拥有文件系统数据副本的
新NameNode，并配置DataNode和客户端以便使用新的NameNode

-	将namespace的映像导入内存中
-	重做编辑日志
-	接收到足够多的来自DataNode的数据块报告，并退出安全模式

###	DataNode

HDFS中保存数据的节点

-	数据被切割为多个数据块，以冗余备份的形式存储在多个
	DataNode中，因此不需要再每个节点上安装RAID存储获得硬件上
	可靠存储支持。DataNode之间可以拷贝数据副本，从而可以重新
	平衡每个节点存储数据量、保证数据可靠性（保证副本数量）

-	DotaNode定期向NameNode报告其存储的数据块列表，以备使用者
	通过直接访问DataNode获得相应数据

-	所有NameNode和DataNode之间的通讯，包括DataNode的注册、
	心跳信息、报告数据块元信息，都是由DataNode发起请求，由
	NameNode被动应答和完成管理

###	HDFS高可用性

对于大型集群，NN冷启动需要30min甚至更长，因此Hadoop2.x中添加
对高可用性HA（high-availability）的支持

-	配置Active-Standby NameNode
	-	ANN失效后，SNN就会接管任务并开始服务，没有明显中断
	-	ANN、SNN应该具有相同的硬件配置

-	NN之间需要通过**高可用的共享存储（JounalNode）**实现
	Editlog共享
	-	JN进程轻量，可以和其他节点部署在同一台机器
	-	JN至少为3个，最好为奇数个，这样JN失效$(n-1)/2$个时
			仍然可以正常工作
	-	SNN接管工作后，将通读共享编辑日志直到末尾，实现与ANN
		状态同步

-	DN需要同时向两个NN发送数据块处理报告，因为数据块映射信息
	存储在NN内存中

-	客户端需要使用特定机制处理NN失效问题，且机制对用户透明

-	如果两个namenode同时失效，同样可以冷启动其他namenode，
	此时情况就和*no-HA*模式冷启动类似

注意：HA模式下，不应该再次配置Secondary NameNode
>	Note that, in an HA cluster, the Standby NameNode also 
	performs checkpoints of the namespace state, and thus it
	is not necessary to run a Secondary NameNode, 
	CheckpointNode, or BackupNode in an HA cluster. In fact,
	to do so would be an error. This also allows one who is
	reconfiguring a non-HA-enabled HDFS cluster to be
	HA-enabled to reuse the hardware which they had
	previously dedicated to the Secondary NameNode.

####	Failover Controller

故障转移控制器系统中有一个新实体管理者管理namenode之间切换，

-	Failover Controller最初实现基于Zookeeper，可插拔

-	每个namenode运行着一个Failover Controller，用于监视宿主
	namenode是否失效（heart beat机制）， 并在失效时进行故障
	切换
	-	管理员也可以手动发起故障切换，称为*平稳故障转移*

-	在非平稳故障切换时，无法确切知道失效namenode是否已经停止
	运行，如网速慢、网络切割均可能激发故障转移，引入fencing
	机制
	-	杀死namenode进程
	-	收回对共享存储目录权限
	-	屏蔽相应网络端口
	-	STONITH：shoot the other node in the head，断电

###	联邦HDFS

NameNode在内存中保存文件系统中每个文件、数据块的引用关系，
所以对于拥有大量文件的超大集群，内存将成为系统扩展的瓶颈，
2.x中引入的联邦HDFS可以添加NameNode实现扩展

-	每个NameNode维护一个namespace volume，包括命名空间的
	元数据、命令空间下的文件的所有数据块、数据块池
-	namespace volume之间相互独立、不通信，其中一个NameNode
	失效也不会影响其他NameNode维护的命名空间的可用性
-	数据块池不再切分，因此集群中的DataNode需要注册到每个
	NameNode，并且存储来自多个数据块池的数据块

##	Hadoop文件系统

Hadoop有一个抽象问的文件系统概念，HDFS只是其中的一个实现，
Java抽象类`org.apche.hadoop.fs.FileSystem`定义了Hadoop中的
一个文件系统接口，包括以下具体实现

|文件系统|URI方案|Java实现|描述|
|-----|-----|-----|-----|
|Local|`file`|`fs.LocalFileSystem`|使用客户端校验和本地磁盘文件系统，没有使用校验和文件系统`RawLocalFileSystem`|
|HDFS|`hdfs`|`hdfs.DistributedFileSystem`|HDFS设计为与MapReduce结合使用实现高性能|
|HFTP|`hftp`|`hdfs.HftpFileSystem`|在HTTP上提供对HDFS只读访问的文件系统，通常与distcp结合使用，以实现在运行不同版本HDFS集群之间复制数据|
|HSFTP|`hsftp`|`hdfs.HsftpFileSystem`|在HTTPS上同以上|
|WebHDFS|`Webhdfs`|`hdfs.web.WebHdfsFileSystem`|基于HTTP，对HDFS提供安全读写访问的文件系统，为了替代HFTP、HFSTP而构建|
|HAR|`har`|`fs.HarFileSystem`|构建于其他文件系统之上，用于文件存档的文件系统，通常用于需要将HDFS中的文件进行存档时，以减少对NN内存的使用|
|hfs|`kfs`|`fs.kfs.kosmosFileSystem`|CloudStore（前身为Kosmos文件系统）类似于HDFS（GFS），*C++*编写|
|FTP|`ftp`|`fs.ftp.FTPFileSystem`|由FTP服务器支持的文件系统|
|S3（原生）|`S3n`|`fs.s3native.NativeS3FileSystem`|由Amazon S3支持的文件系统|
|S3（基于块）|`S3`|`fs.sa.S3FileSystem`|由Amazon S3支持的文件系统，以块格式存储文件（类似于HDFS），以解决S3Native 5GB文件大小限制|
|分布式RAID|`hdfs`|`hdfs.DistributedRaidFileSystem`|RAID版本的HDFS是为了存档而设计的。针对HDFS中每个文件，创建一个更小的检验文件，并允许数据副本变为2，同时数据丢失概率保持不变。需要在集群中运行一个RaidNode后台进程|
|View|`viewfs`|`viewfs.ViewFileSystem`|针对其他Hadoop文件系统挂载的客户端表，通常用于联邦NN创建挂载点|

###	文件系统接口

Hadoop对文件系统提供了许多接口，一般使用**URI方案**选择合适的
文件系统实例进行交互

####	命令行接口

```bash
$ hadoop fs -copyFromLocal file hdfs://localhost/user/xyy15926/file
	# 调用Hadoop文件系统的shell命令`fs`
	# `-copyFromLocalFile`则是`fs`子命令
	# 事实上`hfds://localhost`可以省略，使用URI默认设置，即
		# 在`core-site.xml`中的默认设置
	# 类似的默认复制文件路径为HDFS中的`$HOME`

$ hadoop fs -copyToLocal file file
```

####	HTTP

-	直接访问：HDFS后台进程直接服务来自于客户端的请求
	-	由NN内嵌的web服务器提供目录服务（默认50070端口）
	-	DN的web服务器提供文件数据（默认50075端口）

-	代理访问：依靠独立代理服务器通过HTTP访问HDFS
	-	代理服务器可以使用更严格的防火墙策略、贷款限制策略

####	C

Hadoop提供`libhdfs`的C语言库，是Java `FileSystem`接口类的
镜像

-	被写成访问HDFS的C语言库，但其实可以访问全部的Hadoop文件
	系统
-	使用Java原生接口（JNI）调用Java文件系统客户端

####	FUSE

Filesystem in Userspace允许把按照用户空间实现的文件系统整合
成一个Unix文件系统

-	使用Hadoop Fuse-DFS功能模块，任何一个Hadoop文件系统可以
	作为一个标准文件系统进行挂载
	-	Fuse_DFS使用C语言实现，调用libhdfs作为访问HDFS的接口
-	然后可以使用Unix工具（`ls`、`cat`等）与文件系统交互
-	还可以使用任何编程语言调用POSIX库访问文件系统

###	读文件

1.	客户端程序使用要读取的文件名、Read Range的开始偏移量、
	读取范围的程度等信息，询问NameNode

2.	NameNode返回落在读取范围内的数据块的Location信息，根据
	与客户端的临近性（Proximity）进行排序，客户端一般选择
	最临近的DataNode发送读取请求

具体实现如下

1.	客户端调用`FileSystem`对象`open`方法，打开文件，获得
	`DistributedFileSystem`类的一个实例

2.	`DistributedFileSystem`返回`FSDataInputStream`类的实例，
	支持文件的定位、数据读取

	-	`DistributedFileSystem`通过**RPC**调用NameNode，获得
		文件首批若干数据块的位置信息（Locations of Blocks）
	-	对每个数据块，NameNode会返回拥有其副本的所有DataNode
		地址
	-	其包含一个`DFSInputStream`对象，负责管理客户端对HDFS
		中DataNode、NameNode存取

3.	客户端从输入流调用函数`read`，读取文件第一个数据块，不断
	调用`read`方法从DataNode获取数据

	-	`DFSInputStream`保存了文件首批若干数据块所在的
		DataNode地址，连接到closest DataNode
	-	当达到数据块末尾时，`DFSInputStream`关闭到DataNode
		的连接，创建到保存其他数据块DataNode的连接
	-	首批数据块读取完毕之后，`DFSInputStream`向NameNode
		询问、提取下一批数据块的DataNode的位置信息

4.	客户端完成文件的读取，调用`FSDataInputStream`实例`close`
	方法关闭文件

###	写文件

-	客户端询问NameNode，了解应该存取哪些DataNode，然后客户端
	直接和DataNode进行通讯，使用Data Transfer协议传输数据，
	这个流式数据传输协议可以提高数据传输效率

-	创建文件时，客户端把文件数据缓存在一个临时的本地文件上，
	当本地文件累计超过一个数据块大小时，客户端程序联系
	NameNode，NameNode更新文件系统的NameSpace，返回Newly
	Allocated数据块的位置信息，客户端根据此信息本文件数据块
	从临时文件Flush到DataNode进行保存

具体实现如下：

1.	客户端调用`DistributedFileSystem`的`create`方法

	-	`DistributedFileSystem`通过发起RPC告诉NameNode在
		其NameSpace创建一个新文件，此时新文件没有任何数据块
	-	NameNode检查：文件是否存在、客户端权限等，检查通过
		NameNode为新文件创建新记录、保存其信息，否则文件创建
		失败

2.	`DistributedFileSystem`返回`FSDataOutputStream`给客户端
	
	-	其包括一个`DFSOutputStream`对象，负责和NameNode、
		DataNode的通讯

3.	客户端调用`FSDataOutputStream`对象`write`方法写入数据

	-	`DFSOutputStream`把数据分解为数据包Packet，写入内部
		Data Queue
	-	`DataSteamer`消费这个队列，写入本地临时文件中
	-	当写入数据超过一个数据块大小时，`DataStreamer`请求
		NameNode为新的数据块分配空间，即选择一系列合适的
		DataNode存放各个数据块各个副本
	-	存放各个副本的DataNode形成一个Pipeline，流水线上的
		Replica Factor数量的DataNode接收到数据包之后转发给
		下个DataNode
	-	`DFSOutputStream`同时维护数据包内部Ack Queue，用于
		等待接受DataNode应答信息，只有某个数据包已经被流水线
		上所有DataNode应答后，才会从Ack Queue上删除

4.	客户端完成数据写入，调用`FSDataOutputStream`的`close`
	方法

	-	`DFSOutputStream`把所有的剩余的数据包发送到DataNode
		流水线上，等待应答信息
	-	最后通知NameNode文件结束
	-	NameNode自身知道文件由哪些数据块构成，其等待数据块
		复制完成，然后返回文件创建成功

##	Hadoop平台上的列存储

列存储的优势

-	更少的IO操作：读取数据的时候，支持Prject Pushdown，甚至
	是Predicate Pushdown，可大大减少IO操作

-	更大的压缩比：每列中数据类型相同，可以针对性的编码、压缩

-	更少缓存失效：每列数据相同，可以使用更适合的Cpu Pipline
	编码方式，减少CPU cache miss

###	RCFile

Record Columnar File Format：FB、Ohio州立、中科院计算所合作
研发的列存储文件格式，首次在Hadoop中引入列存储格式

-	*允许按行查询，同时提供列存储的压缩效率*的列存储格式
	-	具备相当于行存储的数据加载速度、负载适应能力
	-	读优化可以在扫描表格时，避免不必要的数据列读取
	-	使用列维度压缩，有效提升存储空间利用率

-	具体存储格式
	-	首先横向分割表格，生成多个Row Group，大小可以由用户
		指定
	-	在RowGroup内部，按照列存储一般做法，按列把数据分开，
		分别连续保存
		-	写盘时，RCFile针对每列数据，使用Zlib/LZO算法进行
			压缩，减少空间占用
		-	读盘时，RCFile采用Lazy Decompression策略，即用户
			查询只涉及表中部分列时，会跳过不需要列的解压缩、
			反序列化的过程

###	ORC存储格式

Optimized Row Columnar File：对RCFile优化的存储格式

-	支持更加丰富的数据类型

	-	包括Date Time、Decimal
	-	Hive的各种Complex Type，包括：Struct、List、Map、
		Union

-	Self Describing的列存储文件格式

	-	为Streaming Read操作进行了优化
	-	支持快速查找少数数据行

-	Type Aware的列存储文件格式

	-	文件写入时，针对不同的列的数据类型，使用不同的编码器
		进行编码，提高压缩比
		-	整数类型：Variable Length Compression
		-	字符串类型：Dictionary Encoding

-	引入轻量级索引、基本统计信息

	-	包括各数据列的最大/小值、总和、记录数等信息
	-	在查询过程中，通过谓词下推，可以忽略大量不符合查询
		条件的记录

####	文件结构

一个ORC文件由多个Stripe、一个包含辅助信息的FileFooter、以及
Postscript构成

#####	Stripe

每个stripe包含index data、row data、stripe footer

-	stripe就是ORC File中划分的row group
	-	默认大小为256MB，可扩展的长度只受HDFS约束
	-	大尺寸的strip、对串行IO的优化，能提高数据吞吐量、
		读取更少的文件，同时能把减轻NN负担

-	Index Data部分
	-	包含每个列的极值
	-	一系列Row Index Entry记录压缩模块的偏移量，用于跳转
		到正确的压缩块的位置，实现数据的快速读取，缺省可以
		跳过10000行


-	Row Data部分；包含每个列的数据，每列由若干Data Stream
	构成

-	Stripe Footer部分

	-	Data Stream位置信息
	-	每列数据的编码方式

#####	File Footer

包含该ORCFile文件中所有stripe的元信息

-	每个Stripe的位置
-	每个Stripe的行数
-	每列的数据类型
-	还有一些列级别的聚集结果，如：记录数、极值、总和

#####	Postscript

-	用来存储压缩参数
-	压缩过后的Footer的长度

###	Parquet

灵感来自于Google关于Drenel系统的论文，其介绍了一种支持嵌套
结构的列存储格式，以提升查询性能

####	支持

Parquet为hadoop生态系统中的所有项目，提供支持高压缩率的
列存储格式

-	兼容各种数据处理框架
	-	MapReduce
	-	Spark
	-	Cascading
	-	Crunch
	-	Scalding
	-	Kite

-	支持多种对象模型
	-	Avro
	-	Thrift
	-	Protocol Buffers

-	支持各种查询引擎
	-	Hive
	-	Impala
	-	Presto
	-	Drill
	-	Tajo
	-	HAWQ
	-	IBM Big SQL

####	Parquet组件

-	Storage Format：存储格式，定义了Parquet内部的数据类型、
	存储格式

-	Object Model Converter：对象转换器，由Parquet-mr实现，
	完成对象模型与Parquet数据类型的映射
	-	如Parquet-pig子项目，负责把内存中的Pig Tuple序列化
		并按存储格式保存为Parquet格式的文件，已经反过来，
		把Parquet格式文件的数据反序列化为Pig Tuple

-	Object Model：对象模型，可以理解为内存中的数据表示，包括
	Avro、Thrift、Protocal Buffer、Hive Serde、Pig Tuple、
	SparkSQL Internal Row等对象模型

####	Parquet数据schema

数据schema（结构）可以用一个棵树表达

-	有一个根节点，根节点包含多个Feild（子节点），子节点可以
	包含子节点

-	每个field包含三个属性

	-	repetition：field出现的次数
		-	`required`：必须出现1次
		-	`optional`：出现0次或1次
		-	`repeated`：出现0次或多次

	-	type：数据类型
		-	primitive：原生类惬
		-	group：衍生类型

	-	name：field名称

-	Parquet通过把多个schema结构按树结构组合，提供对复杂类型
	支持

	-	List、Set：repeated field
	-	Map：包含键值对的Repeated Group（key为Required）

-	schema中有多少**叶子节点**，Parquet格式实际存储多少列，
	父节点则是在表头组合成schema的结构

####	Parquet文件结构

-	HDFS文件：包含数据和元数据，数据存储在多个block中
-	HDFS Block：HDFS上最小的存储单位
-	Row Group：按照行将数据表格划分多个单元，每个行组包含
	一定行数，行组包含该行数据各列对应的列块
	-	一般建议采用更大的行组（512M-1G），此意味着更大的
		列块，有毅力在磁盘上串行IO
	-	由于可能依次需要读取整个行组，所以一般让一个行组刚好
		在一个HDFS数据块中，HDFS Block需要设置得大于等于行组
		大小
-	Column Chunk：每个行组中每列保存在一个列块中
	-	行组中所有列连续的存储在行组中
	-	不同列块使用不同压缩算法压缩
	-	列块存储时保存相应统计信息，极值、空值数量，用于加快
		查询处理
	-	列块由多个页组成
-	Page：每个列块划分为多个Page
	-	Page是压缩、编码的单元
	-	列块的不同页可以使用不同的编码方式

##	HDFS命令

###	用户

-	HDFS的用户就是当前linux登陆的用户

##	Hadoop组件

###	Hadoop Streaming







