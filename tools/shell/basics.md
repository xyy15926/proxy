#	Shell（Bash）常见问题

##	用户配置文件的执行顺序、情况

###	配置文件介绍

-	`/etc/profile`：所有用户登陆时都会执行一次，并从
	`/etc/profile.d`目录的配置文件中收集配置信息

-	`/etc/bashrc`：所有用户打开bash shell执行

-	`~/.bash_profile(.profile,.bash_login)`：仅当前用户登陆
	时执行一次，并且一般会在其中调用~/.bashrc

-	`~/.bashrc`：每次打开bash shell都会执行，而且里面会调用
	`/etc/bashrc`（不是很理解为啥，如果是系统级文件先执行）

-	`~/.bash_logout`：退出bash shell时执行

###	配置文件执行顺序

先执行系统级（对所有用户设置）的配置文件`/etc/profile`，
然后执行用户级的配置文件即`/home/user`目录下的文件，即用户
登陆的实际执行顺序：
	
-	`/etc/profile`
-	`~/.bash_profile`
-	`~/.bashrc`
-	`/etc/bashrc`

因为`~/.profile`中会执行一次`~/.bashrc`，所以需要注意命令是
应该放在`~/.profile`中执行，还是`~/.bashrc`中执行

-	单纯的命令执行，对系统环境无影响根据需要选择，只需要登陆
	时执行应放在`~/.profile`中执行

-	对系统环境有影响时，因为.profile中执行.bashrc的结果会保留
	下来，所以在图形界面下打开命令行会相当于执行`~/.bashrc`
	命令两次

##	可执行文件

###	4种执行方式

-	`$ file_name`：表示在当前shell执行文件，需要有文件的执行
	权限

-	`$ source/. file_name`：`source`和`.`意义一致，表示读取
	文件内容在shell中，然后在shell里执行文件内容，需要读权限

-	`$ sh file_name`：表示开启一个新的shell进程，并读取文件
	内容在新的shell中然后执行，同样的需读权限

###	举例

以下面的文件test.sh为例

```shell
#!/bin/bash
echo "fisrt"
sleep 1000
echo "second"
slepp 1000
```

-	`$ test.sh`：产生两个新进程test.sh和sleep，在second输出
	之前`<ctrl-c>`，会同时终止两个进程，不会继续输出second

-	`$ sh test.sh`：产生两个新进程，shell和sleep，在second
	输出之前`<ctrl-c>`，同样的会同时终止两个进程，不会继续
	输出second（实际上first是在新的shell里输出的）

-	`source/. test.sh`：产生一个新进程sleep，在second输出之前
	`<ctrl-c>`，只有sleep进程被终止，而test.sh的内容在当前
	shell里直接执行，当前shell没有被终止，second会继续输出

###	结论

-	如果需要对当前shell进行设置，应该使用source/.

-	否则这三种执行方式除了进程产生、权限要求应该没有差别

##	Shell常识

-	图形化app无法启动，可以在shell中启动，若由缺少依赖导致，
	在shell中可以显示缺少的依赖，而图形化界面可能无法看出

###	Shell命令常识

感觉上所有涉及到文件“输入”、“输出”的命令，一般都是将
“目标输出”放在“输入”前，如：zip，ln

###	环境变量

-	设置环境变量直接：`$ ENV_NAME=value`（`=`前后不能有空格）
	但是此时环境变量只能在当前shell中使用，需要`export`命令
	导出之后才可在子shell中使用

-	环境变量设置代理
	```shell
	export http_proxy=socks://ip
	export https_proxy=socks5://ip
	```

###	输入输出

-	标准输出和标准错误虽然都是在命令行上显示，但是两个是
	不同的流。输出重定向“>”只能将标准输出重定向，不会将
	标准错误重定向，其仍然会在命令行显示

