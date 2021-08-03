---
title: 配置文件笔记
categories:
  - Tool
  - Markup Language
tags:
  - Tool
  - Markup Language
  - Ini
  - Toml
  - Yaml
date: 2019-07-11 00:51:41
updated: 2021-07-16 17:40:58
toc: true
mathjax: true
comments: true
description: 配置文件笔记
---

##	Ini

##	Toml

##	Yaml

###	基本语法规则

-	大小写敏感
-	缩进代表层级关系
-	必须空格缩进
	-	不要求空格数目
	-	同层左对齐

###	数据结构

> - `-`、`:`、`?`等符号后总是需要空格
> - `#`表示注释

####	对象/映射

对象/映射：键值对集合，`:`表示、`{}`行内表示

```yaml
// `:`后要空格
key: value

// 多层级对象
key:
	child_key1: val1
	child_key2: val2
// 流式表示
key: {child_key1: val1, child_key2: value2}


// 复杂对象格式：键、值都是数组
// `? `（空格）表示复杂key
? 
	- complex_key1
	- complex_key2
// `: `（空格）表示复杂value
: 
	- complex_val1
	- complex_val2
```

####	数组

数组：`-`开头、`[]`行内表示

```yaml
// `[[ele1, ele2]]`
- 
	- ele1
	- ele2


// `pkey: [{key1: val1}, {key2: val2, key3: val3}]`
pkey: 
	- 
		key1: val1
	- 
		key2: val2
		key3: val3
```

####	标量

```yaml
boolean: 
	- TRUE				# true, True均可
	- FALSE				# false, False均可

float: 
	- 3.14
	- 3.14e+2			# 科学计数法

int: 
	- 13
	- 0b1010_1010_1010_1010			#二进制

null: 
	key: ~				# `~`表示null

string:
	- 'hello world'		# 单、双引号包括特殊字符
	- line1
	  line2				# 字符串可以拆成多行，换行转换为空格

datetime:
	- 2019-07-10						# ISO 8601格式，`yyyy-MM-dd`
	- 2019-07-10T17:53:23+08:00			# ISO 8601格式，`<date>T<time>+<timezone>`
```

###	特殊符号

-	`---`：表示文档开始
-	`...`：文档结束
	-	二者配合在文件中记录多个yaml配置项
-	`!!`：强制类型转换
-	`>`：折叠换行符为空格
-	`|`：保留换行符
-	`&`：锚点
	-	不能独立定义，即非列表、映射值
-	`*`：锚点引用
	-	可以多次引用
	-	被引用值可能会之后被覆盖
-	`<<`：合并内容
	-	主要配合锚点使用
	-	相当于unlist解构


```yaml
---							# 文档开始
string: 
	- !!str 13
	- !!str true
...							# 文档结束

---!!set					# 强转为set
- A1: &A1 {x: 1, y: 2}		# 定义名称为`A1`锚点
- A2: &A2 {a: 3, b: 4}		# 定义名称为`A2`锚点
- B: >						# 折叠换行符
	this line
	will collapse
- C: |						# 保留换行符
	this paragraph
	keeps the <CR>
- D: *A`					# 引用名为`SS`的锚点
- E:						# E等价于`{x:1, y:2, a:34, b:4}`
	<<: [*A1, *A2]
	a: 34
...
```

###	API

-	Java
	-	package：`org.yaml.snakeyaml.Yaml`
-	Python
	-	package：`PyYaml`
	-	`import yaml`

##	Xml

