#	转义序列

##	ANSI转义序列

*ANSI*：一种*In-band Signaling*的转义序列标准，用于控制终端
上**光标位置、颜色、其他选项**

-	在文本中嵌入ANSI转义序列，终端会将ANSI转义序列解释为相应
	指令，而不是普通字符

-	所有ANSI转义序列都以ASCII字符`ESC`/`\033`/`\x1b`开头

###	*Constrol Seqence Introducer*

控制序列：`ESC [` + 若干`参数字节` + 一个`最终字节`

###	*ESC - But Not CSI-sequences*

ESC非控制转义序列：`ESC` + `目标字节`

##	C0、C1控制字符集

除`ESC`之外的*C0控制字符*在输出时也会产生和ANSI序列类似效果

-	`LF`：`\n`/`\x0a`类似于`ESC E`/`\x1bE`

###	标准C转义规则

-	`\newline`：反斜杠、换行被忽略
-	`\\`：反斜杠`\`
-	`\'`：单引号`'`
-	`\"`：双引号`"`
-	`\a`：`BEL`ASCII响铃
-	`\b`：`BS`ASCII退格
-	`\f`：`FF`ASCII进纸
-	`\n`：`LF`/`NL`ASCII换行，开启新行
-	`\r`：`CR`ASCII回车，“指针移至行首”
-	`\t`：`TAB`ASCII制表符
-	`\v`：`VT`垂直制表符
-	`\ooo`：八进制数`ooo`码位字符
-	`\xhh`：十六进制数`hh`码位字符





