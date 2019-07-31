#	Shell脚本执行

##	脚本执行

###	`#!`

脚本首行注释

-	文件执行时，shell会将文件内容发送至`#!`之后的解释器上
-	不限于shell脚本，也适合其他类型的脚本语言
	-	当然要求`#`为脚本语言支持的注释符
-	建议在首行使用`env`命令而不是硬编码解释器路径，这样可以
	减少对机器的依赖
	```shell
	#!/usr/bin/env python
	#!/usr/bin/env shell
	```

###	4种执行方式

-	`$ file_name`：表示在当前shell执行文件，需要有文件的执行
	权限

-	`$ source/. file_name`：`source`和`.`意义一致，表示读取
	文件内容在shell中，然后在shell里执行文件内容，需要读权限

-	`$ sh file_name`：表示开启一个新的shell进程，并读取文件
	内容在新的shell中然后执行，同样的需读权限

####	例

以下面的文件`test.sh`为例

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

####	结论

-	如果需要对当前shell进行设置，应该使用source/.

-	否则这三种执行方式除了进程产生、权限要求应该没有差别

##	Shell环境

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

-	`0`：STDIN_FILENO，标准输入
-	`1`：STDOUT_FILENO，标准输出
-	`2`：STDERR_FILENO，标准错误

说明：

-	标准输出和标准错误虽然都是在命令行上显示，但是两个是
	不同的流。输出重定向`>`只能将标准输出重定向，不会将
	标准错误重定向，其仍然会在命令行显示

####	重定向

-	`<`：重定向标准输入
-	`>`：标准输出write重定向
-	`>>`：标准输出add重定向
-	`2>`：标准错误重定向

说明：

-	输出重定向就是`1 > output`，`command > output`只是省略
	`1`的简写
	-	命名自己不是输出，不能重定向
	-	是命令产生的标准输出重定向

-	可以通过`2>&1`将标准错误重定向到标准输出，从而将命令的
	标准输出、错误输出同时输出

	-	`&`这里表示**等效于标准输出（标准输出的引用）**
	-	`command>a 2>a`和`command>a 2>&1`看起来类似，实际上
		前者写法会打开两次`a`，stderr覆盖stdout，而后者是
		引用，不会覆盖、IO效率更高
	-	`2>&1`放在后面大概是因为需要获取stdout的引用，所以
		需要标准输出先给出重定向
	-	这种写法还有个**简写**`&>`、`>&`

-	可以重定向到设备，在\*nix系统中，设备也被视为文件
	```shell
	$ echo "hello" > /dev/tty01
	```
	特别的设备`/dev/null`可以作为不需要输出的重定向目标

####	管道

`|`：将一个程序的标准输出发送到另一个程序的标准输入

-	一些平台会并行启动以管道连接的程序，此时使用类似迭代器
	输出、输入能提高效率

###	Shell内置命令

> - `which`命令无法找到的命令

####	`set`

`set`：设置所使用的shell选项、列出shell变量

-	` `：不带参数，显示**全部**shell变量
-	`-a`：输出之后所有至`export`（环境变量）
-	`-b`：使被终止后台程序立即汇报执行状态
-	`-B`：执行括号扩展
-	`-C`：重定向所产生的文件无法覆盖已存在文件
-	`-d`：shell默认使用hash表记忆已使用过的命令以加速
	执行，此设置取消该行为
-	`-e`：若指令回传值不为0，立即退出shell
-	`-f`：取消使用通配符
-	`-h`：寻找命令时记录其位置???
-	`-H`：（默认）允许使用`!`加*<编号>*方式执行`history`中
	记录的命令
-	`-k`：命令**后**的`=`**也**被视为设置命令的环境变量
-	`-m`：监视器模式，启动任务控制
	-	后台进程已单独进程组运行
	-	每次完成任务时显示包含退出的状态行
-	`-n`：读取命令但不执行
	-	通常用于检查脚本句法错误
-	`-p`：允许*set-user/group-id*
	-	禁止处理`$ENV`文件、从文件中继承shell函数
-	`-P`：处理`cd`等改变当前目录的命令时，不解析符号链接
-	`-t`：读取、执行下条命令后退出
-	`-u`：使用未设置变量作为错误处理
-	`-v`：输入行被读取时，显示shell输出行
-	`-x`：显示简单命令的PS4扩展值（包括所有参数）、当前命令
	的环境变量
-	`-o`：option-name，下列之一
	-	`allexport`：同`-a`
	-	`braceexpand shell`：（默认）执行花括号扩展
	-	`emacs`：（默认）使用emacs风格命令行编辑接口
	-	`errexit`：同`-e`
	-	`errtrace`：同`-E`
	-	`functrace`：同`-T`
	-	`hashall`：同`-h`
	-	`histexpand`：同`-H`
	-	`history`：记录命令历史
	-	`ignoreeof`：读取EOF时不退出shell
	-	`interactive-comments`：允许交互式命令中出现注释
	-	`keyword`：同`-k`
	-	`monitor`：同`-m`
	-	`noclobber`：同`-C`
	-	`noexec`：同`-n`
	-	`noglob`：同`-f`
	-	`noglob`：currently accepted but ignored
	-	`nohash`：同`-d`
	-	`notify`：同`-b`
	-	`nounset`：同`-u`
	-	`physical`：同`-P`
	-	`pipfail`：管道命令返回值为最后返回值非0命令的状态，
		若没有非0返回值返回0
	-	`posix`：改变shell属性以匹配标准，默认操作不同于
		POSIX1003.2标准
	-	`priviledged`：同`-p`
	-	`verbose`：同`-v`
	-	`vi`：使用vi风格的命令编辑接口
	-	`xtrace`：同`-x`
-	`+`：加以上参数取消标志位设置（包括`o`参数）
-	`--`：给所有剩余参数设置标志位，没有剩余参数则unset
-	`-`：给所有剩余参数设置标志位

> - `$-`中存放有当前已设置标志位

####	`let`

`let`：执行计算的工具，用于执行一个、多个表达式

-	变量计算中不需要`$`标识变量
-	表达式中包含特殊字符，需要引起来

####	`declare`

`declare`：声明变量内容

-	`-a`：声明变量为普通数组
-	`-A`：声明变量为关联数组（下标支持字符串，类似字典）
-	`-i`：声明变量为整形
-	`-r`：声明变量只读
-	`-x`：设置为环境变量（`export`）
-	`-g`：在函数中创建全局变量
-	`+`：和以上联合使用，取消定义的变量类型

-	`-f`：列出脚本中定义的函数名称、函数体
-	`-F`：列出脚本中定义的函数名称
-	`-p`：查看变量属性、值

####	`read`

`read`：从标准输入读入变量内容

```shell
$ read [<option>] <var>[ <var2>...]
$ read -p "Enter names: " name1, nem
```

> - 读取值数目大于变量数目时，多余值均赋给最后变量

-	`-p`：命令行提示内容
-	`-n`：不换行
-	`-d`：分隔标记
-	`-t`：阻塞等待时常，之后返回非零
-	`-s`：隐藏输入
-	`-r`：反折号不被视为转义标记

```shell
cat README | while read -r line; do
	echo $line
done
```

####	`shift` 

`shift`：移除参数列表中头部部分参数，缺省移除1个

####	`getopts`

`getopts`：逐个读取解析单字符指示的参数

```shell
function func (){
	echo OPTIND: $OPTIND
	while getopts ":a:B:cdef" opt; do
	  case $opt in
		a) echo "this is -a the arg is ! $OPTARG at $OPTIND" ;;
		B) echo "this is -B the arg is ! $OPTARG at $OPTIND" ;;
		c) echo "this is -c the arg is ! $OPTARG at $OPTIND" ;;
		\?) echo "Invalid option: -$OPTARG" ;;
	  esac
	done
	echo OPTIND: $OPTIND
	echo $@
	shift $(($OPTIND - 1))
	echo $@

}
func -a 23 -B 1904-03-04 343 age
```

> - `$OPTARG`：参数值
> - `$OPTIND`：在参数列表中位移，初始值为1，常配合`shift`
	使用剔除已处理参数
> - 遇到未指定option则返回0

-	option字符后`:`：字符可以带有参数，赋给`$OPTARG`；否则
	仅表示标志，仅该字符被处理

-	option_string开头`:`可以避免错误输出
	-	`:`标记option不带参数时
		-	待匹配值设为`:`
		-	`$OPTARG`设为option
	-	option无效时
		-	待匹配值设为`?`
		-	`$OPTARG`设为option

> - 函数内部处理函数参数，否则处理脚本参数
> - 参数标识`-`可省略

