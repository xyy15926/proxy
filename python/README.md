#	Python笔记约定

##	函数书写声明

```python
return = func(essential(type), optional=defaults/type,
	*args, **kwargs)
```

###	格式说明

####	参数

-	`essential`参数：`essential(type)`
	-	没有`=`
	-	在参数列表开头
	-	`(type)`：表示参数可取值类型

-	`optional`参数：`optional=defaults/type`
	-	`defaults`若为具体值，表示默认参数值
		-	默认值首字母大写
	-	默认参数值为`None`
		-	函数内部有默认行为
		-	`None`（默认行为等价参数值）
	-	`type`：之后表示可能取值类型

-	`args`参数：`[参数名]=defaults/type`
	-	首参数值为具体值表示函数默认行为（不是默认值，`args`
		参数没有默认值一说）
	-	其后为可取参数值类型
	-	说明
		-	参数名仅是**标记**作用，不能使用关键字传参
		-	`[]`：代表参数“可选”

-	`kwargs`参数：`[param_name=defaults/type]`
	-	参数默认为可选参数，格式规则类似`args`参数

-	POSITION_ONLY参数：`[param_name](defaults/type)`
	-	POSITION_ONLY参数同样没有默认值一说，只是表示默认
		行为方式（对应参数值）

补充：
-	参数名后有`?`表示参数名待确定
-	参数首字母大写表示唯一参数

####	返回值

返回值类型由返回对象**名称**蕴含

####	对象类型

-		`obj(type)`：`type`表示**包含**元素类型

####	其他

-	DF对象和Series对象都具有的函数属性列出DF对象

##	常用参数说明

以下常用参数如不特殊注明，按照此解释

###	Threading

-	`block/blocking = True/False`

	-	含义：是否阻塞
	-	默认：大部分为`True`（阻塞）

-	`timeout = None/num`

	-	含义：延迟时间，单位一般是秒
	-	默认：None，无限时间

