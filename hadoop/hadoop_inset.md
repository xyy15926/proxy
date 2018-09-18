#	Hadoop安装配置

##	安装

###	依赖

-	Java：具体版本查询<http://wiki.apache.org/hadoop/HadoopJavaVersions>
-	ssh：必须安装且保证sshd一直运行，以便使用hadoop脚本管理
	远端hadoop守护进程
	-	pdsh：建议安装获得更好的ssh资源管理

###	下载

下载合适的**binary**版本，以下以3.1.1版本为例

-	地址：<http://hadoop.apache.org/releases.html>

###	安装配置

设置`JAVA_HOME`为Java安装根路径

```cnf
JAVA_HOME = /path/to/java/root
	# etc/hadoop/hadoop-env.sh
```

Hadoop集群有三种启动模式

####	Standalone Operation

默认为单机模式，hadoop被配置为以非分布模式运行的一个独立Java
进程，对调试有帮助

```shell
# $ cd hadoop
$ mkdir input
$ cp etc/hadoop/*.xml input
$ bin/hadoop jar share/hadoop/mapreduce/hadoop-mapreduce-examples-3.1.1.jar grep input output 'dfs[a-z.]+'
$ cat output/*
```

####	Pseudo-Distributed Operation

在单节点（服务器）上以所谓的伪分布式模式运行，此时每个Hadoop
守护进程作为独立的Java进程运行

#####	配置

-	`etc/hadoop/core-site.xml`
	```xml
	<configuration>
		<property>
			<name>fs.defaultFS</name>
			<value>hdfs://localhost:9000</value>
		</property>
	</configuration>
	```

-	`etc/hadoop/hdfs-site.xml`
	```xml
	<configuration>
		<property>
			<name>dfs.replication</name>
			<value>`</value>
		</property>
	</configuration>
	```

#####	设置ssh免密登陆

```shell
$ ssh localhost
	# 检查是否可以免密ssh连接本机

$ ssh-keygen -r rsa -P '' -f ~/.ssh/id_rsa
$ cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
$ chmod 600 ~/.ssh/authorized_keys
	# 设置免密
```

#####	执行

```shell
$ bin/hdfs/ namenode -format
	# 格式化文件系统
$ sbin/start-dfs.sh
	# 启动NameNode和DataNode
	# 此时已可访问NameNode，默认http://localhost:9870/
$ bin/hdfs dfs -mkdir /user
$ bin/hdfs dfs -mkdir /user/<username>
	# 创建执行MapReduce任务所需的HDFS文件夹
$ bin/hdfs dfs -mkdir input
$ bin/hdfs dfs -put etc/hadoop/*.xml input
	# 复制文件至分布式文件系统
$ bin/hadoop jar share/hadoop/mapreduce/hadoop-mapreduce-examples-3.1.1.jar grep input output 'dfs[a-z]+'
	# 执行自带样例

$ bin/hdfs dfs -get output outut
$ cat output/*
	# 检查输出文件：将所有的输出文件从分布式文件系统复制
	# 至本地文件系统，并检查
$ bin/hdfs dfs -cat output/*
	# 或者之间查看分布式文件系统上的输出文件

$ sbin/stop-dfs.sh
	# 关闭守护进程
```

####	YARN on a Singel Node

配置

-	`etc/hadoop/mapred-site.xml`

	```cnf
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

-	`etc/hadoop/yarn-site.xml`

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

####	执行

```shell
$ sbin/start-yarn.sh
	# 启动ResourceManger守护进程、NodeManager守护进程
	# 即可访问ResourceManager的web接口，默认：http://localhost:8088/
$ sbin/stop-yarn.sh
	# 关闭守护进程
```

####	完全分布模式

Hadoop的java配置取决于以下两类配置文件

-	只读默认配置文件：`core-defualt.xml`、
	`hdfs-default.xml`、`mapred-default.xml`

-	随站点变化的配置文件：`etc/hadoop/core-site.xml`、
	`etc/hadoop/hdfs-site.xml`、`etc/hadoop/yarn-env.xml`

另外，可以使用`etc/hadoop/hadoop-env.sh`、
`etc/hadoop/yarn-env.sh`设置随站点变化的值，从而控制`bin/`中
的hadoop脚本行为



<http://hadoop.apache.org/docs/r3.1.1>
