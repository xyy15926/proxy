---
title: Hadoop安装配置
categories:
  - Database
  - Hadoop
tags:
  - Database
  - Hadoop
  - Installment
  - Setting
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:08
toc: true
mathjax: true
comments: true
description: Hadoop安装配置
---

##	Hadoop安装

###	依赖

-	Java
	-	具体版本<http://wiki.apache.org/hadoop/HadoopJavaVersions>
	-	需要配置好java环境（`~/.bashrc`）

-	ssh：必须安装且保证sshd一直运行，以便使用hadoop脚本管理
	远端hadoop守护进程
	-	pdsh：建议安装获得更好的ssh资源管理
	-	要设置免密登陆

###	机器环境配置

####	`~/.bashrc`

这里所有的设置都只是设置环境变量

-	所以这里所有环境变量都可以放在`hadoop-env.sh`中

-	放在`.bashrc`中不是基于用户隔离的考虑
	-	因为hadoop中配置信息大部分放在`.xml`，放在这里无法
		实现用户隔离
	-	更多的考虑是给hive等依赖hadoop的应用提供hadoop配置

```shell
export HADOOP_PREFIX=/opt/hadoop
	# 自定义部分
	# 此处是直接解压放在`/opt`目录下
export HADOOP_HOME=$HADOOP_PREFIX
export HADOOP_COMMON_HOME=$HADOOP_PREFIX
	# hadoop common
export HADOOP_HDFS_HOME=$HADOOP_PREFIX
	# hdfs
export HADOOP_MAPRED_HOME=$HADOOP_PREFIX
	# mapreduce
export HADOOP_YARN_HOME=$HADOOP_PREFIX
	# YARN
export HADOOP_CONF_DIR=$HADOOP_PREFIX/etc/hadoop

export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export HADOOP_OPTS="$HADOOP_OPTS -Djava.library.path=$HADOOP_COMMON_LIB_NATIVE_DIR"
	# 这里`-Djava`间不能有空格

export CLASSPATH=$CLASS_PATH:$HADOOP_PREFIX/lib/*
export PATH=$PATH:$HADOOP_PREFIX/sbin:$HADOOP_PREFIX/bin
```

####	`/etc/hosts`

```cnf
192.168.31.129 hd-master
192.168.31.130 hd-slave1
192.168.31.131 hd-slave2
127.0.0.1 localhost
```

-	这里配置的ip地址是各个主机的ip，需要自行配置
-	`hd-master`、`hd-slave1`等就是主机ip-主机名映射
-	#todo?一定需要在`/etc/hostname`中设置各个主机名称

####	`firewalld`

必须关闭所有节点的防火墙

```shell
$ sudo systemctl stop firewalld.service
$ sudo systemctl disable firewalld.service
```

####	文件夹建立

-	所有节点都需要建立

```shell
$ mkdir tmp
$ mkdir -p hdfs/data hdfs/name
```

###	Hadoop配置

Hadoop**全系列**（包括hive、tez等）配置取决于以下两类配置文件

-	只读默认配置文件
	-	`core-defualt.xml`
	-	`hdfs-default.xml`
	-	`mapred-default.xml`

-	随站点变化的配置文件

	-	`etc/hadoop/core-site.xml`
	-	`etc/hadoop/hdfs-site.xml`
	-	`etc/hadoop/mapred-site.xml`
	-	`etc/hadoop/yarn-env.xml`

-	环境设置文件：设置随站点变化的值，从而控制`bin/`中的
	hadoop脚本行为

	-	`etc/hadoop/hadoop-env.sh`、
	-	`etc/hadoop/yarn-env.sh`
	-	`etc/hadoop/mapred-env.sh`

	中一般是环境变量配置，**补充**在shell中未设置的环境变量

-	注意

	-	`.xml`配置信息可在不同应用的配置文件中**继承**使用，
		如在tez的配置中可以使用`core-site.xml`中
		`${fs.defaultFS}`变量

	-	应用会读取/执行相应的`*_CONF_DIR`目录下所有
		`.xml`/`.sh`文件，所以理论上可以在`etc/hadoop`中存放
		所以配置文件，因为hadoop是最底层应用，在其他所有应用
		启动前把环境均已设置完毕？？？

Hadoop集群有三种运行模式

-	Standalone Operation
-	Pseudo-Distributed Operation
-	Fully-Distributed Operation

针对不同的运行模式有，hadoop有三种不同的配置方式

####	Standalone Operation

hadoop被配置为以非分布模式运行的一个独立Java进程，对调试有
帮助

-	默认为单机模式，无需配置

#####	测试

```shell
$ cd /path/to/hadoop
$ mkdir input
$ cp etc/hadoop/*.xml input
$ bin/hadoop jar share/hadoop/mapreduce/hadoop-mapreduce-examples-3.1.1.jar grep input output 'dfs[a-z.]+'
$ cat output/*
```

####	Pseudo-Distributed Operation

在单节点（服务器）上以所谓的伪分布式模式运行，此时每个Hadoop
守护进程作为独立的Java进程运行

#####	`core-site.xml`

```xml
<configuration>
	<property>
		<name>fs.defaultFS</name>
		<value>hdfs://localhost:9000</value>
	</property>
</configuration>
```

#####	`hdfs-site.xml`

```xml
<configuration>
	<property>
		<name>dfs.replication</name>
		<value>1</value>
	</property>
</configuration>
```

#####	`mapred-site.xml`

```xml
<configuration>
	<property>
		<name>mapreduce.framework.name</name>
		<value>yarn</value>
	</property>
</configruration>

<configuration>
	<property>
		<name>mapreduce.application.classpath</name>
		<value>$HADOOP_HOME/share/hadoop/mapreduce/*:$HADOOP_MAPRED_HOME/share/hadoop/mapreduce/lib/*</value>
	</preperty>
</configruation>
```

#####	`yarn-site.xml`

```cnf
<configuration>
	<property>
		<name>yarn.nodemanager.aux-services</name>
		<value>mapreduce_shuffle</value>
	</property>
	<property>
		<name>yarn.nodemanager.env-whitelist</name>
		<value>JAVA_HOME,HADOOP_COMMON_HOME,HADOOP_HDFS_HOME,HADOOP_CONF_DIR,CLASSPATH_PREPEND_DISTCACHE,HADOOP_YARN_HOME,HADOOP_MAPRED_HOME</value>
	</property>
</configuration>
```

####	Fully-Distributed Operation

-	**单节点配置完hadoop之后，需要将其同步到其余节点**

#####	`core-site.xml`

模板：`core-site.xml`

```xml
<configuration>
	<property>
		<name>fs.defaultFS</name>
		<value>hdfs://hd-master:9000</value>
		<description>namenode address</description>
	</property>
	<property>
		<name>hadoop.tmp.dir</name>
		<value>file:///opt/hadoop/tmp</value>
	</property>
	<property>
		<name>io.file.buffer.size</name>
		<value>131702</value>
	</property>

	<property>
		<name>hadoop.proxyuser.root.hosts</name>
		<value>*</value>
	</property>
	<property>
		<name>hadoop.proxyuser.root.groups</name>
		<value>*</value>
	</property>
	<!-- 为将用户`root`设置为超级代理，代理所有用户，如果是其他用户需要相应的将root修改为其用户名 -->
	<!-- 是为hive的JDBCServer远程访问而设置，应该有其他情况也需要 -->
</configuration>
```

#####	`hdfs-site.xml`

模板：`hdfs-site.xml`

```xml
<configuration>
	<property>
		<name>dfs.namenode.secondary.http-address</name>
		<value>hd-master:9001</value>
	</property>
	<property>
		<name>dfs.namenode.name.dir</name>
		<value>file:///opt/hadoop/hdfs/name</value>
		<description>namenode data directory</description>
	</property>
	<property>
		<name>dfs.datanode.data.dir</name>
		<value>file:///opt/hadoop/hdfs/data</value>
		<description>datanode data directory</description>
	</property>
	<property>
		<name>dfs.replication</name>
		<value>2</value>
		<description>replication number</description>
	</property>
	<property>
		<name>dfs.webhdfs.enabled</name>
		<value>true</value>
	</property>

	<property>
		<name>dfs.datanode.directoryscan.throttle.limit.ms.per.sec</name>
		<value>1000</value>
	</property>
	<!--bug-->
</configuration>
```

#####	`yarn-site.xml`

-	模板：`yarn-site.xml`

```xml
<configuration>
	<property>
		<name>yarn.resourcemanager.hostname</name>
		<value>hd-master</value>
	</property>
	<property>
		<name>yarn.resourcemanager.address</name>
		<value>hd-master:9032</value>
	</property>
	<property>
		<name>yarn.resourcemanager.scheduler.address</name>
		<value>hd-master:9030</value>
	</property>
	<property>
		<name>yarn.resourcemanager.resource-tracker.address</name>
		<value>hd-master:9031</value>
	</property>
	<property>
		<name>yarn.resourcemanager.admin.address</name>
		<value>hd-master:9033</value>
	</property>
	<property>
		<name>yarn.resourcemanager.webapp.address</name>
		<value>hd-master:9099</value>
	</property>

	<!-- container -->
	<property>
		<name>yarn.scheduler.maximum-allocation-mb</name>
		<value>512</value>
		<description>maximum memory allocation per container</description>
	</property>
	<property>
		<name>yarn.scheduler.minimum-allocation-mb</name>
		<value>256</value>
		<description>minimum memory allocation per container</description>
	</property>
	<!-- container -->

	<!-- node -->
	<property>
		<name>yarn.nodemanager.resource.memory-mb</name>
		<value>1024</value>
		<description>maximium memory allocation per node</description>
	</property>
	<property>
		<name>yarn.nodemanager.vmem-pmem-ratio</name>
		<value>8</value>
		<description>virtual memmory ratio</description>
	</property>
	<!-- node -->

	<property>
		<name>yarn.app.mapreduce.am.resource.mb</name>
		<value>384</value>
	</property>
	<property>
		<name>yarn.app.mapreduce.am.command-opts</name>
		<value>-Xms128m -Xmx256m</value>
	</property>

	<property>
		<name>yarn.nodemanager.vmem-check-enabled</name>
		<value>false</value>
	</property>

	<property>
		<name>yarn.nodemanager.resource.cpu-vcores</name>
		<value>1</value>
	</property>

	<property>
		<name>yarn.nodemanager.aux-services</name>
		<value>mapreduce_shuffle</value>
	</property>
	<property>
		<name>yarn.nodemanager.aux-services.mapreduce.shuffle.class</name>
		<value>org.apache.hadoop.mapred.ShuffleHandler</value>
	</property>
</configuration>
```

#####	`mapred-site.xml`

-	模板：`mapred-site.xml.template`

```xml
<configuration>
	<property>
		<name>mapreduce.framework.name</name>
		<value>yarn</value>
		<!--
		<value>yarn-tez</value>
		设置整个hadoop运行在Tez上，需要配置好Tez
		-->
	</property>
	<property>
		<name>mapreduce.jobhistory.address</name>
		<value>hd-master:10020</value>
	</property>
	<property>
		<name>mapreduce.jobhistory.webapp.address</name>
		<value>hd-master:19888</value>
	</property>

	<!-- mapreduce -->
	<property>
		<name>mapreduce.map.memory.mb</name>
		<value>256</value>
		<description>memory allocation for map task, which should between minimum container and maximum</description>
	</property>
	<property>
		<name>mapreduce.reduce.memory.mb</name>
		<value>256</value>
		<description>memory allocation for reduce task, which should between minimum container and maximum</description>
	</property>
	<!-- mapreduce -->

	<!-- java heap size options -->
	<property>
		<name>mapreduce.map.java.opts</name>
		<value>-Xms128m -Xmx256m</value>
	</property>
	<property>
		<name>mapreduce.reduce.java.opts</name>
		<value>-Xms128m -Xmx256m</value>
	</property>
	<!-- java heap size options -->

</configuration>
```
#####	参数说明

-	`yarn.scheduler.minimum-allocation-mb`：container内存
	单位，也是container分配的内存最小值

-	`yarn.scheduler.maximum-allocation-mb`：container内存
	最大值，应该为最小值整数倍

-	`mapreduce.map.memeory.mb`：map task的内存分配
	-	hadoop2x中mapreduce构建于YARN之上，资源由YARN统一管理
	-	所以maptask任务的内存应设置container最小值、最大值间
	-	否则分配一个单位，即最小值container

-	`mapreduce.reduce.memeory.mb`：reduce task的内存分配
	-	设置一般为map task的两倍

-	`*.java.opts`：JVM进程参数设置
	-	每个container（其中执行task）中都会运行JVM进程
	-	`-Xmx...m`：heap size最大值设置，所以此参数应该小于
		task（map、reduce）对应的container分配内存的最大值，
		如果超出会出现physical memory溢出
	-	`-Xms...m`：heap size最小值？#todo

-	`yarn.nodemanager.vmem-pmem-ratio`：虚拟内存比例
	-	以上所有配置都按照此参数放缩
	-	所以在信息中会有physical memory、virtual memory区分

-	`yarn.nodemanager.resource.memory-mb`：节点内存设置
	-	整个节点被设置的最大内存，剩余内存共操作系统使用

-	`yarn.app.mapreduce.am.resource.mb`：每个Application
	Manager分配的内存大小

####	主从文件

#####	`masters`

-	设置主节点地址，根据需要设置

```cnf
hd-master
```

#####	`slaves`

-	设置从节点地址，根据需要设置

```cnf
hd-slave1
hd-slave2
```

####	环境设置文件

-	这里环境设置只是起补充作用，在`~/.bashrc`已经设置的
	环境变量可以不设置
-	但是在这里设置环境变量，然后把整个目录同步到其他节点，
	可以保证在其余节点也能同样的设置环境变量

#####	`hadoop-env.sh`

设置`JAVA_HOME`为Java安装根路径

```cnf
JAVA_HOME=/opt/java/jdk
```
#####	`hdfs-env.sh`

设置`JAVA_HOME`为Java安装根路径

```cnf
JAVA_HOME=/opt/java/jdk
```
#####	`yarn-env.sh`

设置`JAVA_HOME`为Java安装根路径

```cnf
JAVA_HOME=/opt/java/jdk
JAVA_HEAP_MAX=Xmx3072m
```

####	初始化、启动、测试

#####	HDFS

-	格式化、启动

	```shell
	$ hdfs namenode -format
		# 格式化文件系统
	$ start-dfs.sh
		# 启动NameNode和DataNode
		# 此时已可访问NameNode，默认http://localhost:9870/
	$ stop-dfs.sh
	```

-	测试

	```shell

	$ hdfs dfsadmin -report
		# 应该输出3个节点的情况

	$ hdfs dfs -mkdir /user
	$ hdfs dfs -mkdir /user/<username>
		# 创建执行MapReduce任务所需的HDFS文件夹
	$ hdfs dfs -mkdir input
	$ hdfs dfs -put etc/hadoop/*.xml input
		# 复制文件至分布式文件系统
	$ hadoop jar /opt/hadoop/share/hadoop/mapreduce/hadoop-mapreduce-examples-2.7.7.jar grep input output 'dfs[a-z]+'
		# 执行自带样例
		# 样例名称取决于版本

	$ hdfs dfs -get output outut
	$ cat output/*
		# 检查输出文件：将所有的输出文件从分布式文件系统复制
		# 至本地文件系统，并检查
	$ hdfs dfs -cat output/*
		# 或者之间查看分布式文件系统上的输出文件


	$ hadoop jar /opt/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.7.7.jar \
		-input /path/to/hdfs_file \
		-output /path/to/hdfs_dir \
		-mapper "/bin/cat" \
		-reducer "/user/bin/wc" \
		-file /path/to/local_file \
		-numReduceTasks 1
	```

#####	YARN

```shell
$ sbin/start-yarn.sh
	# 启动ResourceManger守护进程、NodeManager守护进程
	# 即可访问ResourceManager的web接口，默认：http://localhost:8088/
$ sbin/stop-yarn.sh
	# 关闭守护进程
```

###	其他

####	注意事项

-	`hdfs namenode -format`甚至可以在datanode节点没有java时
	成功格式化

-	没有关闭防火墙时，整个集群可以正常启动，甚至可以在hdfs里
	正常建立文件夹，但是**无法写入文件**，尝试写入文件时报错

####	可能错误

#####	节点启动不全

-	原因
	-	服务未正常关闭，节点状态不一致

-	关闭服务、删除存储数据的文件夹`dfs/data`、格式化namenode

#####	文件无法写入

>	could only be replicated to 0 nodes instead of minReplication (=1).  There are 2 datanode(s) running and 2 node(s) are excluded in this operation.

-	原因
	-	未关闭防火墙
	-	存储空间不够
	-	节点状态不一致、启动不全
	-	在log里面甚至可能会出现一个连接超时1000ms的ERROR

-	处理
	-	关闭服务、删除存储数据的文件夹`dfs/data`、格式化
		namenode
		-	这样处理会丢失数据，不能用于生产环境
	-	尝试修改节点状态信息文件`VERSION`一致
		-	`${hadoop.tmp.dir}`
		-	`${dfs.namenode.name.dir}`
		-	`${dfs.datanode.data.dir}`

#####	Unhealthy Node

>	1/1 local-dirs are bad: /opt/hadoop/tmp/nm-local-dir; 1/1 log-dirs are bad: /opt/hadoop/logs/userlogs

-	原因：磁盘占用超过90%

####	常用命令

```shell
scp -r /opt/hadoop/etc/hadoop centos2:/opt/hadoop/etc
scp -r /opt/hadoop/etc/hadoop centos3:/opt/hadoop/etc
	# 同步配置

scp /root/.bashrc centos2:/root
scp /root/.bashrc centos3:/root
	# 同步环境

rm -r /opt/hadoop/tmp /opt/hadoop/hdfs
mkdir -p /opt/hadoop/tmp /opt/hadoop/hdfs
ssh centos2 rm -r /opt/hadoop/tmp /opt/hadoop/hdfs
ssh centos2 mkdir -p /opt/hadoop/tmp /opt/hadoop/hdfs/name /opt/hadoop/hdfs/data
ssh centos3 rm -r /opt/hadoop/tmp /opt/hadoop/hdfs/name /opt/hadoop/data
ssh centos3 mkdir -p /opt/hadoop/tmp /opt/hadoop/hdfs/name /opt/hadoop/hdfs/data
	# 同步清除数据

rm -r /opt/hadoop/logs/*
ssh centos2 rm -r /opt/hadoop/logs/*
ssh centos3 rm -r /opt/hadoop/logs/*
	# 同步清除log
```

##	Hive

###	依赖

-	hadoop：配置完成hadoop，则相应java等也配置完成
-	关系型数据库：mysql、derby等

###	机器环境配置

####	`~/.bashrc`

```shell
export HIVE_HOME=/opt/hive
	# self designed
export HIVE_CONF_DIR=$HIVE_HOME/conf
export PATH=$PATH:$HIVE_HOME/bin
export CLASSPATH=$CLASS_PATH:$HIVE_HOME/lib/*
```

####	文件夹建立

#####	HDFS

```shell
$ hdfs dfs -rm -r /user/hive
$ hdfs dfs -mkdir -p /user/hive/warehouse /user/hive/tmp /user/hive/logs
	# 这三个目录与配置文件中对应
$ hdfs dfs -chmod 777 /user/hive/warehouse /user/hive/tmp /user/hive/logs
```

##### FS

```shell
$ mkdir data
$ chmod 777 data
	# hive数据存储文件夹
$ mkdir logs
$ chmod 777 logs
	# log目录
```

###	Hive配置

####	XML参数

#####	`conf/hive-site.xml`

-	模板：`conf/hive-default.xml.template`

```xml
<property>
	<name>javax.jdo.option.ConnectionURL</name>
	<value>jdbc:mysql://hd-master:3306/metastore_db?createDatabaseIfNotExist=true</value>
</property>
<property>
	<name>javax.jdo.option.ConnectionDriverName</name>
	<value>org.mariadb.jdbc.Driver</value>
</property>
<property>
	<name>javax.jdo.option.ConnectionUserName</name>
	</value>hive</value>
</property>
<property>
	<name>javax.jdo.option.ConnectionPassword</name>
	<value>1234</value>
</property>
<property>
	<name>hive.metastore.warehouse.dir</name>
	<value>/user/hive/warehouse</value>
</property>
<property>
	<name>hive.exec.scratchdir</name>
	<value>/user/hive/tmp</value>
</property>

<!--
<property>
	<name>hive.exec.local.scratchdir</name>
	<value>${system:java.io.tmpdir}/${system:user.name}</value>
</property>
<property>
	<name>hive.downloaded.resources.dir</name>
	<valeu>${system:java.io.tmpdir}/${hive.session.id}_resources</value>
</property>
<property>«
	<name>hive.server2.logging.operation.log.location</name>«
	<value>${system:java.io.tmpdir}/${system:user.name}/operation_logs</value>«
	<description>Top level directory where operation logs are stored if logging functionality is enabled</description>«
</property>«
所有`${system.java.io.tmpdir}`都要被替换为相应的`/opt/hive/tmp`，
可以通过设置这两个变量即可，基本是用于设置路径
-->

<property>
	<name>system:java.io.tmpdir</name>
	<value>/opt/hive/tmp</value>
</property>
<property>
	<name>system:user.name</name>
	<value>hive</value>
<property>

<!--
<property>
	<name>hive.querylog.location</name>
	<value>/user/hive/logs</value>
	<description>Location of Hive run time structured log file</description>
</property>
这里应该不用设置，log放在本地文件系统更合适吧
-->

<property>
	<name>hive.metastore.uris</name>
	<value>thrift://192.168.31.129:19083</value>
</property>
<!--这个是配置metastore，如果配置此选项，每次启动hive必须先启动metastore，否则hive实可直接启动-->

<property>
	<name>hive.server2.logging.operation.enabled</name>
	<value>true</value>
</property>
<!-- 使用JDBCServer时需要配置，否则无法自行建立log文件夹，然后报错，手动创建可行，但是每次查询都会删除文件夹，必须查一次建一次 -->
```

-	`/user`开头的路径一般表示hdfs中的路径，而`${}`变量开头
	的路径一般表示本地文件系统路径
	-	变量`system:java.io.tmpdir`、`system:user.name`在
		文件中需要自己设置，这样就避免需要手动更改出现这些
		变量的地方
	-	`hive.querylog.location`设置在本地更好，这个日志好像
		只在hive启动时存在，只是查询日志，不是hive运行日志，
		hive结束运行时会被删除，并不是没有生成日志、`${}`表示
		HDFS路径

-	配置中出现的目录（HDFS、locaL）有些手动建立
	-	HDFS的目录手动建立？
	-	local不用

-	`hive.metastore.uris`若配置，则hive会通过metastore服务
	访问元信息
	-	使用hive前需要启动metastore服务
	-	并且端口要和配置文件中一样，否则hive无法访问

####	环境设置文件

#####	`conf/hive-env.sh`

-	模板：`conf/hive-env.sh.template`

```shell
export JAVA_HOME=/opt/java/jdk
export HADOOP_HOME=/opt/hadoop
export HIVE_CONF_DIR=/opt/hive/conf
	# 以上3者若在`~/.bashrc`中设置，则无需再次设置
export HIVE_AUX_JARS_PATH=/opt/hive/lib
```

#####	`conf/hive-exec-log4j2.properties`

-	模板：`hive-exec-log4j2.properties.template`

	```xml
	property.hive.log.dir=/opt/hive/logs
		# 原为`${sys:java.io.tmpdir}/${sys:user.name}`
		# 即`/tmp/root`（root用户执行）
	```

#####	`conf/hive-log4j2.properties`

-	模板：`hive-log4j2.properties.template`

####	MetaStore

#####	MariaDB

-	安装MariaDB

-	修改MariaDB配置
	```shell
	$ cp /user/share/mysql/my-huge.cnf /etc/my.cnf
	```

-	创建用户，注意新创建用户可能无效，见mysql配置
	-	需要注意用户权限：创建数据库权限、修改表权限
	-	初始化时Hive要自己创建数据库（`hive-site`中配置），
		所以对权限比较严格的环境下，可能需要先行创建同名
		数据库、赋权、删库

-	下载`mariadb-java-client-x.x.x-jar`包，复制到`lib`中

#####	初始化数据库

```shell
$ schematool -initSchema -dbType mysql
```
这个命令要在所有配置完成之后执行

####	服务设置

```shell
$ hive --service metastore -p 19083 &
	# 启动metastore服务，端口要和hive中的配置相同
	# 否则hive无法连接metastore服务，无法使用
	# 终止metastore服务只能根据进程号`kill`
$ hive --service hiveserver2 --hiveconf hive.server2.thrift.port =10011 &
	# 启动JDBC Server
	# 此时可以通过JDBC Client（如beeline）连接JDBC Server对
		# Hive中数据进行操作
$ hive --service hiveserver2 --stop
	# 停止JDBC Server
	# 或者直接kill
```

####	测试

#####	Hive可用性

需要先启动hdfs、YARN、metastore database（mysql），如果有
设置独立metastore server，还需要在正确端口启动

```sql
hive>	create table if not exists words(id INT, word STRING)
		row format delimited fields terminated by " "
		lines terminated by "\n";
hive>	load data local inpath "/opt/hive-test.txt" overwrite into
		table words;
hive>	select * from words;
```

#####	JDBCServer可用性

-	命令行连接
	```shell
	$ beeline -u jdbc:hive2://localhost:10011 -n hive -p 1234
	```

-	beeline中连接
	```shell
	$ beeline
	beeline> !connect jdbc:hive2://localhost:10011
		# 然后输入用户名、密码（metastore数据库用户名密码）
	```

###	其他

####	可能错误

>	Failed with exception Unable to move source file

-	linux用户权限问题，无法操作原文件
-	hdfs用户权限问题，无法写入目标文件
-	hdfs配置问题，根本无法向hdfs写入：参见hdfs问题

>	org.apache.hive.service.cli.HiveSQLException: Couldn't find log associated with operation handle: 

-	原因：hiveserver2查询日志文件夹不存在

-	可以在hive中通过

	```sql
	$ set hive.server2.logging.operation.log.location;
	```
	查询日志文件夹，建立即可，默认为
	`${system:java.io.tmpdir}/${system:user.name}/operation_logs`
	，并设置权限为777

	-	好像如果不设置权限为777，每次查询文件夹被删除，每
		查询一次建立一次文件夹？#todo
	-	在`hive-sitex.xml`中配置允许自行创建？

>	User: root is not allowed to impersonate hive

-	原因：当前用户（不一定是root）不被允许通过代理操作
	hadoop用户、用户组、主机
	-	hadoop引入安全伪装机制，不允许上层系统直接将实际用户
		传递给超级代理，此代理在hadoop上执行操作，避免客户端
		随意操作hadoop

-	配置hadoop的`core-site.xml`，使得当前用户作为超级代理

##	Tez

###	依赖

-	hadoop

###	机器环境配置

####	`.bashrc`

```shell
export TEZ_HOME=/opt/tez
export TEZ_CONF_DIR=$TEZ_HOME/conf

for jar in `ls $TEZ_HOME | grep jar`; do
	export HADOOP_CLASSPATH=$HADOOP_CLASSPATH:$TEZ_HOME/$jar
done
for jar in `ls $TEZ_HOME/lib`; do
	export HADOOP_CLASSPATH=$HADOOP_CLASSPATH:$TEZ_HOME/lib/$jar
done
	# this part could be replaced with line bellow
export HADOOP_CLASSPATH=$HADOOP_CLASSPATH:$TEZ_HOME/*:$TEZ_HOME/lib/*
	# `hadoop-env.sh`中说`HADOOP_CLASSPATH`是Extra Java CLASSPATH
		# elements
	# 这意味着hadoop组件只需要把其jar包加到`HADOOP_CLASSPATH`中既可
```

####	HDFS

-	上传`$TEZ_HOME/share/tez.tar.gz`至HDFS中

	```md
	$ hdfs dfs -mkdir /apps
	$ hdfs dfs -copyFromLocal tez.tar.gz /apps
	```

###	HadoopOnTez

在hadoop中配置Tez

-	侵入性较强，对已有的hadoop集群全体均有影响

-	所有hadoop集群执行的MapReduce任务都通过tez执行

	-	这里所有的任务应该是指直接在hadoop上执行、能在
		webRM上看到的任务
	-	hive这样的独立组件需要独立配置

####	XML参数

#####	`tez-site.xml`

-	模板：`conf/tez-default-tmplate.xml`
-	好像还是需要复制到hadoop的配置文件夹中

```shell
<property>
	<name>tez.lib.uris</name>
	<value>${fs.defaultFS}/apps/tez.tar.gz</value>
	<!--设置tez安装包位置-->
</property>
<!--
<property>
	<name>tez.container.max.java.heap.fraction</name>
	<value>0.2</value>
<property>
内存不足时-->
```

#####	`mapred-site.xml`

-	修改`mapred-site.xml`文件：配置mapreduce基于`yarn-tez`，
	（配置修改在hadoop部分也有）

```xml
<property>
	<name>mapreduce.framework.name</name>
	<value>yarn-tez</value>
</property>
```

####	环境参数

###	HiveOnTez

-	此模式下Hive可以在mapreduce、tez计算模型下自由切换？

	```sql
	hive> set hive.execution.engine=tez;
		# 切换查询引擎为tez
	hive> set hive.execution.engine=mr;
		# 切换查询引擎为mapreduce
		# 这些命令好像没用，只能更改值，不能更改实际查询模型
	```

-	只有Hive会受到影响，其他基于hadoop平台的mapreduce作业
	仍然使用tez计算模型

####	Hive设置

-	若已经修改了`mapred-site.xml`设置全局基于tez，则无需复制
	jar包，直接修改`hive-site.xml`即可

#####	Jar包复制

复制`$TEZ_HOME`、`$TEZ_HOME/lib`下的jar包到`$HIVE_HOME/lib`
下即可

#####	`hive-site.xml`

```
<property>
	<name>hive.execution.engine</name>
	<value>tez</value>
</property>
```

###	其他

####	可能错误

>	SLF4J: Class path contains multiple SLF4J bindings.

-	原因：包冲突的
-	解决方案：根据提示冲突包删除即可

##	Spark

###	依赖

-	java
-	scala
-	python：一般安装anaconda，需要额外配置
	```shell
	export PYTHON_HOME=/opt/anaconda3
	export PATH=$PYTHON_HOME/bin:$PATH
	```
-	相应资源管理框架，如果不以standalone模式运行

###	机器环境配置

####	`~/.bashrc`

```shell
export SPARK_HOME=/opt/spark
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin
export PYTHON_PATH=$PYTHON_PATH:$SPARK_HOME/python:$SPARK_HOME/python/lib/*
	# 把`pyshark`、`py4j`模块对应的zip文件添加进路径
	# 这里用的是`*`通配符应该也可以，手动添加所有zip肯定可以
	# 否则无法在一般的python中对spark进行操作
	# 似乎只要master节点有设置`/lib/*`添加`pyspark`、`py4j`就行
```

###	Standalone

####	环境设置文件

#####	`conf/spark-env.sh`

-	模板：`conf/spark-env.sh.template`

这里应该有些配置可以省略、移除#todo

```shell
export JAVA_HOME=/opt/jdk
export HADOOP_HOME=/opt/hadoop
export hADOOP_CONF_DIR=/opt/hadoop/etc/hadoop
export HIVE_HOME=/opt/hive

export SCALA_HOME=/opt/scala
export SCALA_LIBRARY=$SPARK_HOME/lib
	# `~/.bashrc`设置完成之后，前面这段应该就这个需要设置

export SPARK_HOME=/opt/spark
export SPARK_DIST_CLASSPATH=$(hadoop classpath)
	# 这里是执行命令获取classpath
	# todo
	# 这里看文档的意思，应该也是类似于`$HADOOP_CLASSPATH`
	# 可以直接添加进`$CLASSPATH`而不必设置此变量
export SPARK_LIBRARY_PATH=$SPARK_HOME/lib

export SPARK_MASTER_HOST=hd-master
export SPARK_MASTER_PORT=7077
export SPARK_MASTER_WEBUI_PORT=8080
export SPARK_WORKER_WEBUI_PORT=8081
export SPARK_WORKER_MEMORY=1024m
	# spark能在一个container内执行多个task
export SPARK_LOCAL_DIRS=$SPARK_HOME/data
	# 需要手动创建

export SPARK_MASTER_OPTS=
export SPARK_WORKER_OPTS=
export SPARK_DAEMON_JAVA_OPTS=
export SPARK_DAEMON_MEMORY=
export SPARK_DAEMON_JAVA_OPTS=
```

#####	文件夹建立

```shell
$ mkdir /opt/spark/spark_data
	# for `$SPARK_LOCAL_DIRS`
```

####	Spark配置

#####	`conf/slaves`

文件不存在，则在当前主机单节点运行

-	模板：`conf/slaves.template`

```cnf
hd-slave1
hd-slave2
```

#####	`conf/hive-site.xml`

这里只是配置Spark，让Spark作为“thrift客户端”能正确连上
metastore server

-	模板：`/opt/hive/conf/hive-site.xml`

```xml
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
	<property>
		<name>hive.metastore.uris</name>
		<value>thrift://192.168.31.129:19083</value>
		<description>Thrift URI for the remote metastor. Used by metastore client to connect to remote metastore</description>
	</property>
	<property>
		<name>hive.server2.thrift.port</name>
		<value>10011</value>
	</property>
	<!--配置spark对外界thrift服务，以便可通过JDBC客户端存取spark-->
	<!--这里启动端口同hive的配置，所以两者不能默认同时启动-->
	<property>
		<name>hive.server2.thrift.bind.host</name>
		<value>hd-master</value>
	</property>
</configuration>
```

####	测试

#####	启动Spark服务

需要启动hdfs、正确端口启动的metastore server

```shell
$ start-master.sh
	# 在执行**此命令**机器上启动master实例
$ start-slaves.sh
	# 在`conf/slaves`中的机器上启动worker实例
$ start-slave.sh
	# 在执行**此命令**机器上启动worker实例

$ stop-master.sh
$ stop-slaves.sh
$ stop-slave.sh
```

#####	启动Spark Thrift Server

```shell
$ start-thriftserver.sh --master spark://hd-master:7077 \
	--hiveconf hive.server2.thrift.bind.host hd-master \
	--hiveconf hive.server2.thrift.port 10011
	# 这里在命令行启动thrift server时动态指定host、port
		# 如果在`conf/hive-site.xml`有配置，应该不需要

	# 然后使用beeline连接thrift server，同hive
```

#####	Spark-Sql测试

```sql
$ spark-sql --master spark://hd-master:7077
	# 在含有配置文件的节点上启动时，配置文件中已经指定`MASTER`
		# 因此不需要指定后面配置

spark-sql> set spark.sql.shuffle.partitions=20;
spark-sql> select id, count(*) from words group by id order by id;
```

#####	pyspark测试

```python
$ MASTER=spark://hd-master:7077 pyspark
	# 这里应该是调用`$PATH`中第一个python，如果未默认指定

from pyspark.sql import HiveContext
sql_ctxt = HiveContext(sc)
	# 此`sc`是pyspark启动时自带的，是`SparkContext`类型实例
	# 每个连接只能有一个此实例，不能再次创建此实例

ret = sql_ctxt.sql("show tables").collect()
	# 这里语句结尾不能加`;` 

file = sc.textFile("hdfs://hd-master:9000/user/root/input/capacity-scheduler.xml")
file.count()
file.first()
```

#####	Scala测试

```scala
$ MASTER=spark://hd-master:7077 spark-shell \
	executor-memory 1024m \
	--total-executor-cores 2 \
	--excutor-cores 1 \
	# 添加参数启动`spark-shell`

import org.apache.spark.sql.SQLContext
val sqlContext = new org.apache.spark.sql.hive.HiveContext(sc)
sqlContext.sql("select * from words").collect().foreach(println)
sqlContext.sql("select id, word from words order by id").collect().foreach(println)

sqlContext.sql("insert into words values(7, \"jd\")")
val df = sqlContext.sql("select * from words");
df.show()

var df = spark.read.json("file:///opt/spark/example/src/main/resources/people.json")
df.show()
```

###	Spark on YARN

###	其他

####	可能错误

>	Initial job has not accepted any resources;

-	原因：内存不足，spark提交application时内存超过分配给
	worker节点内存

-	说明

	-	根据结果来看，`pyspark`、`spark-sql`需要内存比
		`spark-shell`少？
		（设置worker内存512m，前两者可以正常运行）
	-	但是前两者的内存分配和scala不同，scala应该是提交任务
		、指定内存大小的方式，这也可以从web-ui中看出来，只有
		spark-shell开启时才算是*application*

-	解决方式
	-	修改`conf/spark-env.sh`中`SPARK_WORKER_MEMORY`更大，
		（spark默认提交application内存为1024m）
	-	添加启动参数`--executor-memory XXXm`不超过分配值

>	 ERROR KeyProviderCache:87 - Could not find uri with key [dfs.encryption.key.provider.uri] to create a keyProvider

-	无影响

##	HBase

###	依赖

-	java
-	hadoop
-	zookeeper：建议，否则日志不好管理

###	机器环境

####	`~/.bashrc`

```shell
export HBASE_HOME=/opt/hbase
export PATH=$PAHT:$HBASE_HOME/bin
export HADOOP_CLASSPATH=$HADOOP_CLASSPATH:$HBASE_HOME/lib/*
```

####	建立目录

```shell
$ mkdir /tmp/hbase/tmpdir
```
###	HBase配置

####	环境变量

#####	`conf/hbase-env.sh`

```shell
export HBASE_MANAGES_ZK=false
	# 不使用自带zookeeper
```

#####	`conf/zoo.cfg`

若设置使用独立zookeeper，需要复制zookeeper配置至HBase配置
文件夹中

```md
$ cp /opt/zookeeper/conf/zoo.cfg /opt/hbase/conf
```


####	Standalone模式

#####	`conf/hbase-site.xml`

```xml
<configuration>
	<property>
		<name>hbase.rootdir</name>
		<value>file://${HBASE_HOME}/data</value>
	</property>
	<property>
		<name>hbase.zookeeper.property.dataDir</name>
		<value>/tmp/zookeeper/zkdata</value>
	</property>
</configuration>
```

####	Pseudo-Distributed模式

#####	`conf/hbase-site.xml`

-	在Standalone配置上修改

```xml
<proeperty>
	<name>hbase.cluster.distributed</name>
	<value>true</value>
</property>
<property>
	<name>hbase.rootdir</name>
	<value>hdfs://hd-master:9000/hbase</value>
</property>
```

####	Fully-Distributed模式

#####	`conf/hbase-site.xml`

```xml
<property>
	<name>hbase.rootdir</name>
	<value>hdfs://hd-master:9000/hbase</value>
</property>
<property>
	<name>hbase.cluster.distributed</name>
	<value>true</name>
</property>
<property>
	<name>hbase.zookeeper.quorum</name>
	<value>hd-master,hd-slave1,hd-slave2</value>
</property>
<property>
	<name>hbase.zookeeper.property.dataDir</name>
	<value>/tmp/zookeeper/zkdata</value>
</property>
```

####	测试

-	需要首先启动HDFS、YARN
-	使用独立zookeeper还需要先行在每个节点启动zookeeper

```md
$ start-hbase.sh
	# 启动HBase服务
$ local-regionservers.sh start 2 3 4 5
	# 启动额外的4个RegionServer
$ hbase shell
hbase> create 'test', 'cf'
hbase> list 'test'
hbase> put 'test', 'row7', 'cf:a', 'value7a'
	put 'test', 'row7', 'cf:b', 'value7b'
	put 'test', 'row7', 'cf:c', 'value7c'
	put 'test', 'row8', 'cf:b', 'value8b',
	put 'test', 'row9', 'cf:c', 'value9c'
hbase> scan 'test'
hbase> get 'test', 'row7'
hbase> disable 'test'
hbase> enable 'test'
hbaee> drop 'test'
hbase> quit
```

##	Zookeeper

###	依赖

-	java

-	注意：zookeeper集群中工作超过半数才能对外提供服务，所以
	一般配置服务器数量为奇数

###	机器环境

####	`~/.bashrc`

```shell
export ZOOKEEPER_HOME=/opt/zookeeper
export PATH=$PATH:$ZOOKEEPER_HOME/bin
export HADOOP_CLASSPATH=$HADOOP_CLASSPATH:$ZOOKEEPER_HOME/lib
```
####	创建文件夹

-	在所有节点都需要创建相应文件夹、`myid`文件

```shell
mkdir -p /tmp/zookeeper/zkdata /tmp/zookeeper/zkdatalog
echo 0 > /tmp/zookeeper/zkdatalog/myid

ssh centos2 mkdir -p /tmp/zookeeper/zkdata /tmp/zookeeper/zkdatalog
ssh centos3 mkdir -p /tmp/zookeeper/zkdata /tmp/zookeeper/zkdatalog
ssh centos2 "echo 2 > /tmp/zookeeper/zkdata/myid"
ssh centos3 "echo 3 > /tmp/zookeeper/zkdata/myid"
```

###	Zookeeper配置

####	Conf

#####	`conf/zoo.cfg`

```cnf
tickTime=2000
	# The number of milliseconds of each tick
initLimit=10
	# The number of ticks that the initial
	# synchronization phase can take
syncLimit=5
	# The number of ticks that can pass between
	# sending a request and getting an acknowledgement
dataDir=/tmp/zookeeper/zkdata
dataLogDir=/tmp/zookeeper/zkdatalog
	# the directory where the snapshot is stored.
	# do not use /tmp for storage, /tmp here is just
	# example sakes.
clientPort=2181
	# the port at which the clients will connect

autopurge.snapRetainCount=3
	# Be sure to read the maintenance section of the
	# administrator guide before turning on autopurge.
	# http://zookeeper.apache.org/doc/current/zookeeperAdmin.html#sc_maintenance
	# The number of snapshots to retain in dataDir
autopurge.purgeInterval=1
	# Purge task interval in hours
	# Set to "0" to disable auto purge feature

server.0=hd-master:2888:3888
server.1=hd-slave1:2888:3888
server.2=hd-slave2:2888:3888
	# Determine the zookeeper servers
	# fromation: server.NO=HOST:PORT1:PORT2
		# PORT1: port used to communicate with leader
		# PORT2: port used to reelect leader when current leader fail
```

#####	`$dataDir/myid`

-	`$dataDir`是`conf/zoo.cfg`中指定目录
-	`myid`文件里就一个id，指明当前zookeeper server的id，服务
	启动时读取文件确定其id，需要自行创建

```cnf
0
```

####	启动、测试、清理

启动zookeeper

```shell
$ zkServer.sh start
	# 开启zookeeper服务
	# zookeeper服务要在各个节点分别手动启动

$ zkServer.sh status
	# 查看服务状态

$ zkCleanup.sh
	# 清理旧的快照、日志文件
```

##	Flume

###	依赖

-	java

###	机器环境配置

####	`~/.bashrc`

```shell
export PATH=$PATH:/opt/flume/bin
```

###	Flume配置

####	环境设置文件

#####	`conf/flume-env.sh`

-	模板：`conf/flume-env.sh.template`

```shell
JAVA_HOME=/opt/jdk
```

####	Conf文件

#####	`conf/flume.conf`

-	模板：`conf/flume-conf.properties.template`

```md
agent1.channels.ch1.type=memory
	# define a memory channel called `ch1` on `agent1`
agent1.sources.avro-source1.channels=ch1
agent1.sources.avro-source1.type=avro
agent1.sources.avro-source1.bind=0.0.0.0
agent1.sources.avro-source1.prot=41414
	# define an Avro source called `avro-source1` on `agent1` and tell it
agent1.sink.log-sink1.channels=ch1
agent1.sink.log-sink1.type=logger
	# define a logger sink that simply logs all events it receives
agent1.channels=ch1
agent1.sources=avro-source1
agent1.sinks=log-sink1
	# Finally, all components have been defined, tell `agent1` which one to activate
```

####	启动、测试

```md
$ flume-ng agent --conf /opt/flume/conf \
	-f /conf/flume.conf \
	-D flume.root.logger=DEBUG,console \
	-n agent1
	# the agent name specified by -n agent1` must match an agent name in `-f /conf/flume.conf`

$ flume-ng avro-client --conf /opt/flume/conf \
	-H localhost -p 41414 \
	-F /opt/hive-test.txt \
	-D flume.root.logger=DEBUG, Console
	# 测试flume
```

###	其他

##	Kafka

###	依赖

-	java
-	zookeeper

###	机器环境变量

####	`~/.bashrc`

```shell
export PATH=$PATH:/opt/kafka/bin
export KAFKA_HOME=/opt/kafka
```

###	多brokers配置

####	Conf

#####	`config/server-1.properties`

-	模板：`config/server.properties`
-	不同节点`broker.id`不能相同
-	可以多编写几个配置文件，在不同节点使用不同配置文件启动

```cnf
broker.id=0
listeners=PLAINTEXT://:9093
zookeeper.connect=hd-master:2181, hd-slave1:2181, hd-slave2:2181
```

####	测试

-	启动zookeeper

```shell
$ kafka-server-start.sh /opt/kafka/config/server.properties &
	# 开启kafka服务（broker）
	# 这里是指定使用单个默认配置文件启动broker
		# 启动多个broker需要分别使用多个配置启动多次
$ kafka-server-stop.sh /opt/kafka/config/server.properties

$ kafka-topics.sh --create --zookeeper localhost:2181 \
	--replication-factor 1 \
	--partitions 1 \
	--topic test1
	# 开启话题
$ kafka-topics.sh --list zookeeper localhost:2181
	# 
$ kafka-topics.shd --delete --zookeeper localhost:2181
	--topic test1
	# 关闭话题

$ kafka-console-producer.sh --broker-list localhost:9092 \
	--topic test1
	# 新终端开启producer，可以开始发送消息

$ kafka-console-consumer.sh --bootstrap-server localhost:9092 \
	--topic test1 \
	--from-beginning

$ kafka-console-consumer.sh --zookeeper localhost:2181 \
	--topic test1 \
	--from beginning
	# 新终端开启consumer，可以开始接收信息
	# 这个好像是错的
```

###	其他

##	Storm

###	依赖

-	java
-	zookeeper
-	python2.6+
-	ZeroMQ、JZMQ

###	机器环境配置

####	`~/.bashrc`

```shell
export STORM_HOME=/opt/storm
export PAT=$PATH:$STORM_HOME/bin
```

###	Storm配置

####	配置文件

#####	`conf/storm.yaml`

-	模板：`conf/storm.yarml`

```shell
storm.zookeeper.servers:
	-hd-master
	-hd-slave1
	-hd-slave2
storm.zookeeper.port: 2181

nimbus.seeds: [hd-master]
storm.local.dir: /tmp/storm/tmp
nimbus.host: hd-master
supervisor.slots.ports:
	-6700
	-6701
	-6702
	-6703
```

####	启动、测试

```shell
storm nimbus &> /dev/null &
storm logviewer &> /dev/null &
storm ui &> /dev/null &
	# master节点启动nimbus

storm sueprvisor &> /dev/null &
storm logviewer &> /dev/nulla &
	# worker节点启动


storm jar /opt/storm/example/..../storm-start.jar \
	storm.starter.WordCountTopology
	# 测试用例
stom kill WordCountTopology
```



<http://hadoop.apache.org/docs/r3.1.1>
