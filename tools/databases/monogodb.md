MongoDb安装配置

1.	安装：建议访问https://docs.mongodb.com/manual/installation，参考官方教程进行安装（以下是对于3.6版本） 
	a.	配置包管理yum：添加/etc/yum.repos.d/mongodb-org-3.6.repo文件，内容：

		[mongodb-org-3.6]
		name=MongoDB Repository
		baseurl=https://repo.mongodb.org/yum/redhat/$releasever/mongodb-org/3.6/x86_64/
		gpgcheck=1
		enabled=1
		gpgkey=https://www.mongodb.org/static/pgp/server-3.6.asc

	b.	使用yum安装即可
	
2.	mongodb包括四个组件,可以按需安装

	a.	mongodb-org-sever：包括mongod守护和相关联的配置文件

	b.	mongodb-org-mongos：包括mongos守护

	c.	mongodb-org-shell：包括mongo shell

	d.	mongodb-org-tools：包括mongo-shell在内的一些工具包

	e.	mongodb-org：一个会安装以上4个组件的包

3.	
