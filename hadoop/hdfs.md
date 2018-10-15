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


##	NameNode

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

###	Metadata的保存

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

###	NameNode可恢复性

-	备份文件系统元信息的持久化版本：在NameNode写入元信息的
	持久化版本时，同步、atomic写入多个文件系统（一般是本地
	磁盘、mount为本地目录的NFS

-	运行Secondary NameNode：负责周期性的使用EditLog更新
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

##	DataNode

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

##	读文件

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

##	写文件

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

