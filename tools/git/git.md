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

##	`config`

-	修改配置类型：缺省查看配置
	-	`--global`：全局配置
	-	`--local`：局部repo配置

###	`user`

-	`user.name`：用户名
-	`user.email`：用户邮箱

```shell
$ git config user.name
$ git config --global user.name "username"
$ git config --local user.email "email"
```

###	`core`

```shell
$ git config --global core.autocrlf input
$ git config --global core.safecrlf true
```

-	`core.autocrlf`：提交、检出时换行符设置
	-	`input`：提交时转换为`<LF>`、检出时不转换
	-	`false`：提交、检出均不转换
	-	`true`：提交时转换为`<LF>`、检出时转换为`<CRLF>`

-	`core.safecrlf`：是否允许文件混用换行符
	-	`true`：拒绝提交包含混合换行符文件
	-	`false`：允许提交包含混合换行符文件
	-	`warn`：提交包含混合换行符文件时警告

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

