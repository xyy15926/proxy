#	Git常识

##	.gitingore忽略文件

###	.gitignore忽略规则

-	`!`开头：排除**应被忽略**

-	`/`**结尾**：忽略文件夹

-	`/`**开头**：git仓库根目录路径开始（可省略）

-	遵循一定的正则表达式
	-	`*`：多个字符，不包括`/`，因此多层级目录需要一一指定
	-	`?`：单个字符
	-	`[]`：匹配其中候选字符

###	.gitignore配置方式

-	仓库根目录创建`.gitignore`文件，添加忽略规则
	-	忽略规则针对当前项目
	-	`.gitignore`文件*默认*随仓库分发，所有人共用忽略规则
		（当然可以在`.gitignore`中配置忽略`.gitignore`）

-	设置全局忽略文件，对所有git项目适用
	```shell
	git config --global core.excludesfile /path/to/.gitignore
	```

-	修改`.git/info/exclude`文件，添加忽略规则
	-	对单一项目有效
	-	非所有人共用忽略规则

##	Config

###	User

```shell
$ git config user.name
$ git config user.email
	# 查看用户名、邮箱
$ git config --global user.name "username"
$ git config --global user.email "email"
	# 设置全局用户名、邮箱
$ git config --local user.name "username"
$ git config --local user.email "email"
	# 设置当前git repo用户名、邮箱
```

##	`log`

```shell
$ git log [<file_name>]
	# 查看文件提交历史
```

##	`show`

```c
$ git show <tag_no>
	# 查看改动内容
```






