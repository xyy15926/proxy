---
title: 
categories:
  - 
tags:
  - 
date: 2022-07-20 16:11:59
updated: 2022-07-20 19:34:00
toc: true
mathjax: true
description: 
---

##	Logging

![logging_flow](imgs/logging_flow.png)

-	Python3 Logging 包含三个模块
	-	`logging`：主模块，其中包含的类、函数根据功能可划分为 4 类
		-	记录器：代码可直接使用的接口
		-	处理器：将日志记录发送到适当目标
		-	过滤器：过滤需要输出的日志记录
		-	格式器：输出的日志记录样式
	-	`logging.handlers`：日志处理程序模块，大部分日志处理类定义于此
	-	`logging.config`：日志配置模块

> - `logging` *API*：<https://docs.python.org/zh-cn/3/library/logging.html>
> - `logging` 使用教程：<https://docs.python.org/zh-cn/3/howto/logging.html>

###	模块类

|类|功能|说明|
|-----|-----|-----|
|`logging.Logger`|记录器|勿直接实例化，可通过 `logging.getLogger` 等创建、获取|
|`logging.Handler`|处理器|勿直接实例化，一般用于派生子类|
|`logging.Formatter`|格式器|根据指定的格式抽取 `LogRecord` 属性，组合为可读字符串|
|`logging.Filter`|过滤器|实现复杂的过滤操作|
|`logging.LogRecord`|日志记录|有日志记录时由 `Logger` 自动创建，也可手动创建|
|`logging.LoggerAdapter`||将上下文信息传入日志记录调用|

-	说明
	-	日志格式字符串可选择 `%`、`{`、`$` 中任意风格
		-	风格由参数 `style` 指明，相应参数也应符合要求
			-	`%`：`%-formatting`，即 `printf` 风格，缺省值
			-	`{`：`str.format`
			-	`$`：`string.Template.substitute`
		-	格式字符串中的可用变量即 `LogRecocrd` 属性
	-	关联到处理器的过滤器会在事件被处理器

> - [`logging.Logger` 类](https://docs.python.org/zh-cn/3/library/logging.html#logging.Logger)
> - [`logging.Handler` 类](https://docs.python.org/zh-cn/3/library/logging.html#handler-objects)
> - [`logging.handlers` 模块](https://docs.python.org/zh-cn/3/library/logging.handlers.html)
> - `logging.LogRecord` 属性: <https://docs.python.org/zh-cn/3/library/logging.html#logrecord-attributes>

###	模块级函数

|函数|功能|说明|
|-----|-----|-----|
|`logging.getLogger`|创建、获取指定名称的 `Logger`|名称应用 `.` 分隔层级|
|`logging.getLoggerClass`|返回标准 `Logger` 类、或最近传给 `setLoggerClass` 类|避免 `setLoggerClass` 被覆盖|
|`logging.setLoggerClass`|通知日志记录系统使用指定类实例化为日志记录器| |
|`logging.debug`|在根日志记录器上记录 `DEBUG` 级消息| |
|`logging.info`|在根日志记录器上记录 `INFO` 级消息| |
|`logging.warning`|在根日志记录器上记录 `WARNING` 级消息| |
|`logging.error`|在根日志记录器上记录 `ERROR` 级消息| |
|`logging.critical`|在根日志记录器上记录 `CRITICAL` 级消息| |
|`logging.exception`|在根日志记录器上记录 `ERROR` 级消息| |
|`logging.log`|在根日志记录器上记录指定级消息| |
|`logging.disable`|重置所有 `Logger` 的记录级别| |
|`logging.addLevelName`|关联日志级别、级别名称|可用于自定义级别|
|`logging.getLevelName`|获取日期级别名|不匹配则返回 `Level %s`|
|`logging.makeLogRecord`|创建、返回 `LogRecord`| |
|`logging.setLogRecordSetFactory`|设置用于创建 `LogRecord` 的可调用对象|方便控制 `LogRecord` 的构造|
|`logging.getLogRecordSetFactory`|返回用于创建 `LogRecord` 的可调用对象| |
|`logging.basicConfig`|使用默认 `Formatter` 创建 `StreamHandler`，并将其加入根日志记录器|未为根日志记录器定义处理器时，直接记录消息将被自动调用|
|`logging.shutdown`|通过刷新、关闭所有处理程序通知日志记录系统停止| |

-	说明
	-	`logging` 中模块级函数是全局于解释器实例的，以便于跨文件记录日志

> - `logging` 模块级函数：<https://docs.python.org/zh-cn/3/library/logging.html#module-level-functions>

###	Logging 使用案例

-	基本使用
	-	`logging.basicConfig`：创建、配置根日志记录器
		-	若使用默认根日志记录器，甚至可忽略
	-	`logging.info` 等：在根日志记录器上记录不同等级日志

```python
import logging
logging.basicConfig(										# 创建、配置日志记录器
	filename="app.log",										# 输出至文件
	format="%(asctime)s: %(levelname)s: %(message)s",		# 配置记录格式
	level=logging.INFO)
logging.warning("A warning")
```

-	自定义配置日志记录器
	-	`logging.getLogger`：获取指定名称日志记录器
		-	之后调用其方法以配置记录器属性、关联处理器等
	-	`logging.StreamHandler`：创建流日志处理器
		-	大部分复杂日志处理器类位于 `logging.handlers` 模块中
	-	`logging.Formatter`：创建自定义日志格式器
	-	`logging.Filter`:创建自定义日志过滤器
	-	`logger.info` 等：在指定日期记录器上记录日志

```python
import logging
logger = logging.getLogger("example")										# 获取指定名称日志记录器
logger.setLevel(logging.INFO)
sh = logging.StreamHandler()												# 创建流日志处理器
sh.setLevel(logging.DEBUG)
fmter = logginng.Formatter("%(asctime)s: %(levelname)s: %(message)s")		# 配置记录格式
sh.setFormater(fmter)														# 日志处理器管理格式器
logger.addHandler(sh)														# 日期记录器添加处理器
logger.warning("A warning")
```

-	使用配置文件配置日志记录器
	-	`logging.config.fileConfig`：读取 *INI* 格式配置文件配置日志记录器
		-	依赖 `configparser` 模块解析配置文件
	-	`logging.config.dictConfig`：从字典配置日志记录器
		-	以支持 *YAML*、*JSON* 等其他格式的配置文件，甚至是 `pickle` 序列化的配置

> - *Logging* 常用指引：<https://docs.python.org/zh-cn/3/howto/logging.html>

