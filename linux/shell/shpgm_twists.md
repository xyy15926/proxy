#	shell编程技巧

###	检查命令是否成功

-	原版
	```shell
	echo abcdee | grep -q abcd

	if [ $? -eq 0 ]; then
		echo "found
	else
		echo "not found"
	fi
	```

-	简洁版
	```shell
	if echo abcdee | grep -q abc; then
		echo "found"
	else
		echo "not found"
	fi
	```

-	精简版
	```shell
	echo abcdee | grep -q abc && echo "found" || echo "not found"
	```

###	标准输出、错误输出重定向到`/dev/null`

-	原版
	```shell
	$ grep "abc" text.txt 1>/dev/null 2>&1
	```

-	简洁版
	```shell
	$ grep "abc" text.txt &> /dev/null
	```

###	`awk`使用

-	原版
	```shell
	$ sudo xm li | grep vm_name | awk `{print $2}`
	```
-	简洁版
	```shell
	$ sudo xm li | awk `/vm_name/{print $2}`
	```

###	逗号连接所有行

-	原版：`sed`
	```shell
	$ sed ":a;$!N;s/\n/,;ta" test.txt
	```

-	简洁：`paste`
	```shell
	$ paste -sd, /tmp/test.txt
	```

###	过滤重复行

-	原版：`sort`
	```shell
	$ sort text.txt | unique
	```

-	简洁版
	```shell
	$ sort -u text.txt
	```



