#	PAC（Proxy Auto-Config）

-------

pac事实上是一个js脚本，定义如何根据浏览器器访问url不同，
自动选取适当proxy

##	主函数

	str = function FindProxyForUrl(url, host){
	}

-	参数
	-	`url`：浏览器访问的完整url地址，如：
		`http://bing.com`
	-	`host`：url中的host部分`bing.com`
-	返回值：字符串变量，表示应该使用何种”代理“，可以是
	以下元素，或者是使用`;`分隔的多个组合，此时浏览器依次
	尝试
	（“据说”每隔30min，浏览器就会尝试之前失败的“代理元素”）

	-	`DIRECT`：不使用代理，直接连接
	-	`PORXY host:post`：使用指定http代理@host:post
	-	`SOCKS host:post`：使用指定socks代理
	-	`SOCKS5 host:post`：使用指定的socks5代理

浏览器在访问每个url时都会调用该函数

##	可用预定义JS函数

###	host、domain、url相关js函数

-	`bool = isPlainHostName(host)`：`host`不包含域名返回
	`true`

		true = isPlainHostName("www")
		false = isPlainHostName("baidu.com")

-	`bool = dnsDomainIs(host, domain)`：`host`和`domain`相
	匹配返回`true`

		true = dnsDomain("www.google.com", ".google.com")
		false = dnsDomain("www.apple.com", ".google.com")

-	`bool = localHostOrDomainIs(host,domain)` ：`host`和
	`domain`匹配、`host`不包含`domain`返回`true`（按照函数名
	理解）

		true = localHostOrDomainIs("www.google.com", "www.google.com")
		true = localHostOrDomainIs("www", "www.google.com")
		false = localHostOrDomainIs("www.apple.com", "www.google.com")

-	`bool = isResolvable(host)`：成功解析`host`返回`true`

-	`bool = isInNet(host, pattern, mask)`：`host`处于
	`pattern`指定网段/地址返回`true`， `host`如果不是`ip`
	形式，将解析成`ip`地址后处理；`mask`指定匹配部分，`mask`
	就是和子网掩码类似

		isinnet("192.168.3.4", "192.168.0.0", "255.255.0.0") -> true
		isinnet("192.168.3.4", "192.168.0.0", "255.255.255.255") -> false

-	`str = myipaddress()`：字符串形式返回本机地址

-	`int = dnsdomainlevels(host)`：返回`host`中域名层级数

		0 = dnsdomainlevels("www")
		2 = dnsdomainlevels("www.google.com")

-	`bool = shexpmatch(str, shexp)`：`str`符合`shexp`
	正则表达式返回`true`

		shexpmatch("www.apple.com/downloads/macosx/index.html", "*/macosx/*") -> true.
		shexpmatch("www.apple.com/downloads/support/index.html", "*/macosx/*") -> false.

###	时间相关JS函数

-	`bool = weekdayrange(wd1, wd2, gmt)`：时间处于指定时间段
	返回`true`

		weekdayrange("mon", "fri") 星期一到星期五(当地时区)为true
		weekdayrange("mon", "fri", "gmt") 从格林威治标准时间星期一到星期五为true
		weekdayrange("sat") 当地时间星期六为true
		weekdayrange("sat", "gmt") 格林威治标准时间星期六为true
		weekdayrange("fri", "mon") 从星期五到下星期一为true(顺序很重要)

-	`bool = daterange(..)`：时间处于指定时间段返回`true`
	（包括首尾日期）

		daterange(1) 当地时区每月第一天为true
		daterange(1, "gmt") gmt时间每月的第一天为true
		daterange(1, 15) 当地时区每月1号到15号为true
		daterange(24, "dec") 在当地时区每年12月24号为true
		daterange(24, "dec", 1995) 在当地时区1995年12月24号为true
		daterange("jan", "mar") 当地时区每年第一季度(1月到3月)为true
		daterange(1, "jun", 15, "aug") 当地时区每年6月1号到8月15号为true
		daterange(1, "jun", 15, 1995, "aug", 1995) 当地时区1995年6月1号到8月15号为true
		daterange("oct", 1995, "mar", 1996) 当地时区1995年10月到1996年3月为true
		daterange(1995) 当地时区1995年为true
		daterange(1995, 1997) 当地时区1995年初到1997年底为true

-	`bool = timerange(..)`：时间处于指定时间段返回`true`

		timerange(12)中午12点到下午1点之间为true
		timerange(12, 13)同上例
		timerange(12, "gmt")在gmt时间中午12点到下午1点之间为true
		timerange(9, 17)上午9点到下午5点之间为true
		timerange(8, 30, 17, 00)上午8点30分到下午5点之间为true.
		timerange(0, 0, 0, 0, 0, 30)午夜0点到其后的30秒内为true

