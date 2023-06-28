---
title: SQLAlchemy 常用基础
categories:
  - Python
  - DataBase
tags:
  - Python
  - DataBase
  - Briefs
  - SQL
date: 2023-03-10 14:35:03
updated: 2023-03-20 18:40:26
toc: true
mathjax: true
description: 
---

##	SQLAlchemy 概述

-	*SQLAlchemy* 由两组 *API* 组成
	-	*Core*：基础架构，管理数据连接、查询、结果交互、*SQL* 编程
	-	*ORM*：基于 *CORE* ，提供对象关系映射能力
		-	允许 *类-表* 映射、绘画
		-	扩展 *Core* 级别的 *SQL* 表达式语言，允许通过对象组合调用 *SQL* 查询

###	*Engine*

```python
from sqlalchemy import create_engine
engine = create_engine(
	url="{DIALECT}[+{DRIVER}]://USER:PWD@HOST/DBNAME[?KEY=VAL...]",
	echo=True, future=True)
```

-	*Engine* 引擎：连接到特定数据库的中心源
	-	提供工厂、数据库连接池
	-	为特定数据库实例仅需创建一个全局对象

|`Engine` 方法|描述|
|-----|-----|
|`Engine.connect([close_with_result])`|创建 *边走边做* 风格连接|
|`Engine.begin([close_with_result])`|创建 *开始一次* 风格连接|
|`Engine.execute(statement,*multiparams,**params)`|绑定参数、执行语句|

|`Engine` 相关函数|描述|
|-----|-----|
|`sa.create_engine(url, **kwargs)`|创建引擎|

-	`create_engine(url,**kwargs)`：创建引擎的工厂方法
	-	部分参数说明
		-	`url`：数据地址
			-	`DIALECT`：数据库类型
			-	`DRIVER`：数据库渠道，缺省为 *DBAPI*
			-	`USER:PWD`：用户名、密码
			-	`DBNAME`：数据库实例名
		-	`echo`：启用 *SQl* 记录，写入标准输出
		-	`future`：*SQLAlchemy 2.0* 风格标志

> - 创建 *Engine*：<https://www.osgeo.cn/sqlalchemy/tutorial/engine.html>
> - *Establishing Connectivity*：<https://docs.sqlalchemy.org/en/14/tutorial/engine.html>

###	*Transaction and DBAPI*

```python
with engine.connect() as conn:
	result = conn.execute(text("select * from tbl"))
	conn.commit()
```

-	`sqlalchemy.engine.base.Connection` 连接：表示针对数据库的开放资源
	-	可通过 `with` 语句将对象的使用范围限制在特定上下文中
		-	*边走边做* 风格
			-	`with` 执行块中包含多个事务，手动调用 `Connect.commmit()` 方法提交事务
			-	事务不会自动提交，上下文环境释放时会自动 `ROLLBACK` 回滚事务
		-	*开始一次* 风格
			-	`with` 执行块整体时视为单独事务
			-	上下文环境释放签自动 `COMMIT` 提交事务
	-	部分方法说明

|`Connection` 方法|描述|
|-----|-----|
|`Connection.commit()`|提交事务|
|`Connection.execute()`|执行语句|

-	`sqlalchemy.engine.cursor.Result` 结果：表示语句执行结果的可迭代对象
	-	说明
		-	`CursorResult.inserted_primary_key` 内部实现透明，根据数据库方言自动选择
			-	`CursorResult.lastrowid`
			-	数据库函数
			-	`INSERT...RETURNING` 语法

|`Result` 方法|描述|
|-----|-----|
|`CursorResult.all()`|获取所有行结果|
|`CursorResult.fetchall()`|执行语句|
|`CursorResult.mappings()`|生成 `MappingResult` 对象，可迭代得到 `RowMapping`|
|`CursorResult.inserted_primary_key`|插入数据主键值|
|`CursorResult.lastrowid`| |

-	`sqlalchemy.engine.row.Row` 行：语句执行结果行
	-	行为类似 `namedtuples`，支持多种访问方式
		-	元组赋值
		-	整数索引
		-	属性（*SQL* 语句中字段名）

##	*SQL* 语句、表达式 *API*

|列元素基础构造器|*SQL* 关键字|描述|对应 *Python* 算符|
|-----|-----|-----|-----|
|`and_(*clauses)`|`AND`|与|`&`|
|`or_(*clauses)`|`OR`|或|`|`|
|`not_(clause)`|`NOT`|非|`!`|
|`null()`|`NULL`|空|`== None`|
|`distinct(expr)`|`DISTINCT`|唯一值| |
|`case(*whens, **kw)`|`CASE`| | |
|`cast(expression,type_)`|`CAST`|
|`false()`|`false`、`0 = 1`（不支持场合）|否| |
|`true()`|`true`、`1 = 1`（不支持场合）|是| |

|`extract(field,expr,**kwargs)`|`EXTRACT`|
|`func.<ATTR>`|基于属性名称 `ATTR` 生成 *SQL* 函数|
|`column(text[,type_,is_literal,_selectable])`|列对象，模拟 `Column` 对象|
|`bindparam(key[,value,type_,unique,...])`|生成绑定表达式|
|`custom_op(opstring[,precedence,...])`|自定义运算符|
|`lambda_stmt(lmb[,enable_tracking,track_closure_variables,...])`| |
|`literal(value[,type])`| |
|`literal_column(text[,type_])`| |
|`outparam(key[,type_])`| |
|`quoted_name`| |
|`text()`||
|`tuple_(*clauses,**kw)`| |
|`type_coerce(expression,type_)`| |

-	说明
	-	`and_`、`or_` 对应 `&`、`|` 操作符
	-	绑定表达式为 *SQL* 表示式中形如 `:` 占位符
		-	表达式执行前绑定具体实参
		-	除 `bindparam` 显式创建外，传递给 *SQL* 表达式的所有 *Python* 文本值都以此方式设置

|列元素修饰构造器|描述|
|-----|-----|
|`all_(expr)`| |
|`any_(expr)`| |
|`asc(column)`| |
|`between(expr,lower_bound,upper_bound[,symmetric])`| |
|`collate(expression,collation)`| |
|`desc(column)`| |
|`funcfilter(func,*criterion)`| |
|`label(name,element[,type_])`| |
|`nulls_first(column)`| |
|`nulls_last(column)`| |
|`over(element[,partiion_by,order_by,range_,...])`| |
|`within_group(element,*order_by)`| |


|列元素类|含义|
|-----|-----|
|`BinaryExpression(left,right,operator[,...])`|`LEFT <OP> RIGHT` 表达式|
|`BindParameter(key[,value,...])`|绑定表达式|
|`Case`|`CASE` 表达式|
|`ClauseList`|操作符分割的多个子句|
|`ColumnElement`|*SQL* 中用于列、文
|`ColumnClause`|文本字符串创建的列表达式|
|`ColumnCollection`|
|`ColumnOperators`|用于 `ColumnElement` 的操作符|
|`Extract`|`EXTRACT` 子句、`extract(field FROM expr)`|
|`False_`|`FALSE` 关键字|
|`FunctionFilter`|函数 `FILTER` 关键字|
|`Label`|列标签 `AS`|
|`NULL`|`NULL` 关键字|
|`Operators`|比较、逻辑操作符基类|
|`Over`|`Over` 子句|
|`TextClause`|*SQL* 文本字面值|
|`True`|`TRUE` 关键字|
|`Tuple`|*SQL* 元组|
|`TypeCoerce`|*Python* 侧类型强转包装器|
|`UnaryExpression`|一元操作符表达式|
|`WithinGroup`|`WITHIN GROUP` 子句|
|`WrapsColumnExpression`|具名 `ColumnElement` 包装器|

-	列元素类由以上列元素基础构造器、列元素修饰构造器生成

###	操作符

-	操作符（方法）定义在 `Operators`、`ColumnOperators` 基类中
	-	在其衍生类可用，包括
		-	`Column`
		-	`ColumnElement`
		-	`InstructmentedAttribute`

|比较操作符|*SQL* 关键字|描述|
|-----|-----|-----|
|`ColumnOperators.__eq__()`|`=`| |
|`ColumnOperators.__ne__()`|`!=`、`<>`| |
|`ColumnOperators.__gt__()`|`>`| |
|`ColumnOperators.__lt__()`|`<`| |
|`ColumnOperators.__ge__()`|`>=`| |
|`ColumnOperators.__le__()`|`<=`| |
|`ColumnOperators.between()`|`BETWEEN...AND...`|区间比较|
|`ColumnOperators.in()`|`IN`|支持列表、空列表、元组各元素独立比较|
|`ColumnOperators.not_in()`|`NOT IN`|支持列表、空列表、元组各元素独立比较|
|`ColumnOperators.is_()`|`IS`|主要用于 `is_(None)`，即 `IS NULL`|
|`ColumnOperators.is_not()`|`IS NOT`| |
|`ColumnOperators.is_distinct_from()`|`IS DISTINCT FROM`| |
|`ColumnOperators.isnot_distinct_from()`|`IS NOT DISTINCT FROM`| |

-	说明
	-	*SQLAlchemy* 通过渲染临时、第二步再渲染为绑定参数列表的 *SQL* 字符串以实现 `ColumnOperators.in`
		-	参数可为不定长列表：`IN` 表达式绑定参数数量执行时才确定
		-	参数可为空列表：`IN` 表达式渲染为返回空行字查询
		-	参数可为元组
		-	参数可为子查询
	-	`ColumnOperators.__eq__(None)`（`null()`）会触发 `ColumnOperators.is_()` 操作符，即实务中一般无需显式调用 `is_()`

|字符串比较、操作|*SQL* 关键字|描述|
|-----|-----|-----|
|`ColumnOperators.like()`|`LIKE`| |
|`ColumnOperators.ilike()`|`lower(_) LIKE lower(_)`|大小写不敏感匹配|
|`ColumnOperators.notlike()`|`NOT LIKE`| |
|`ColumnOperators.notilike()`|`lower(_) NOT LIKE lower(_)`| |
|`ColumnOperators.startswith()`|`LIKE _ || '%'`| |
|`ColumnOperators.endswith()`|`LIKE '%' || _`| |
|`ColumnOperators.contains()`|`LIKE '%' || _ || '%'`| |
|`ColumnOperators.match()`|`MATCH`| |
|`ColumnOperators.regexp_match()`|`~ %(_)s`、`REGEXP`|正则匹配|
|`ColumnOperators.concat()`|`||`|字符类型 `+` 同|
|`ColumnOperators.regex_replace()`|`REGEXP_REPALCE(_,%(_)s,%(_)s)`|正则替换|
|`ColumnOperators.collate()`|`COLLATE`|指定字符集排序|

-	说明
	-	`ColumnOperators.match`、`ColumnOperators.regexp_match`、`ColumnOperators.regex_replace` 的支持、行为（方言）、结果依赖数据库后端

|算术操作符|*SQL* 关键字|描述|
|-----|-----|-----|
|`ColumnOperators.__add__()`、`__radd__()`|数值 `+`、字符 `||`|`+`|
|`ColumnOperators.__sub__()`、`__rsub__()`|`-`|`-`|
|`ColumnOperators.__mul__()`、`__rmul__()`|`*`|`*`|
|`ColumnOperators.__div__()`、`__rdiv__()`|`/`|`/`|
|`ColumnOperators.__mod__()`、`__rmod__()`|`%`|`%`|

|连接操作符|函数版本|*SQL* 关键字|描述|
|-----|-----|-----|-----|
|`Operators.__and__()`|`sa.and_()`|`AND`|`&` 与|
|`Operators.__or__()`|`sa.or_()`|`OR`|`|` 或|
|`Operators.__invert__()`|`sa.not_()`|`NOT`|`~` 非|

-	说明
	-	在 `Select.where`、`Update.where`、`Delete.where` 子句中 `AND` 在以下场合自动应用
		-	`.where` 方法重复调用
		-	`.where` 方法中传入多个表达式

> - *SQLAlchemy 1.4 Operator Reference*：<https://docs.sqlalchemy.org/en/14/core/operators.html>
> - *SQLAlchemy 1.4* 操作符参考：<https://www.osgeo.cn/sqlalchemy/core/operators.html>

###	*Selectable*

|基础构建器|*SQL* 关键字|描述|
|-----|-----|-----|
|`sa.select(*args,**kw)`|`SELECT`|创建 `SELECT` 子句|
|`sa.table(name,*columns,**kw)`|`TABLE`|创建 `TableClause`|
|`sa.values(*columns,**kw)`|`VALUES`|创建 `VALUES` 子句|
|`sa.exists(*args,**kwargs)`|`EXISTS`|存在，可直接被调用创建 `EXISTS` 子句|
|`sa.except_(*selects,**kwargs)`|`EXCEPT`|差集|
|`sa.except_all(*selects,**kwargs)`|`EXCEPT ALL`| |
|`sa.intersect(*selects,**kargs)`|`INTERSECT`|交集|
|`sa.intersect_all(*selects,**kwargs)`|`INTERSECT ALL`| |
|`sa.union(*select,**kwargs)`|`UNION`|并集|
|`sa.union_all(*select,**kwargs)`|`UNION ALL`| |

-	*Selectable* 可选择对象：由 `FromClause` 演变而来（类似 *Table*）
	-	`sa.excepts`、`sa.intersect`、`sa.union`、
#TODO

|修饰构建器|*SQL* 关键字|返回值|描述|
|-----|-----|-----|-----|
|`alias(selectable[,name,flat])`|`AS`|`Alias`|别名|
|`cte(selectable[,name,recursive])`| |`CTE`| |
|`join(left,right[,onclause,isouter,...])`|`JOIN`|`Join`| |
|`lateral(selectable[,name])`| |`Lateral`| |
|`outerjoin(left,right[,onclause,full])`|`OUTER JOIN`| | |
|`tablesample(select,sampling[,name,seed])`| |`TableSample`| |

|可选择对象类|描述|
|-----|-----|
|`Alias(*arg,**kw)`|别名|
|`AliasedReturnsRows(*args,**kw)`|别名类基类|
|`CompoundSelect(keyword,*selects,**kwargs)`|基于 `SELECT` 操作的基础|
|`CTE`|*Common Table Expression*|
|`Executable`|可执行标记|
|`Exists`|`EXISTS` 子句|
|`FromClause`|`FROM` 字句中可使用标记|
|`GenerativeSelect`|`SELECT` 语句基类|
|`HasCTE`|包含 *CTE* 支持标记|
|`HasPrefixes`| |
|`HasSuffixes`| |
|`Join`|`FromClause` 间 `Join`|
|`Lateral`|`LATERAL` 子查询|
|`ReturnRows`|包含列、表示行的最基类|
|`ScalarSelect`|标量子查询|
|`Select`|`SELECT` 语句|
|`Selectable`|可选择标记|
|`SelectBase`|`SELECT` 语句基类|
|`Subquery`|子查询|
|`TableClause`|最小 *Table* 概念|
|`TableSample`|`TABLESAMPLE` 子句|
|`TableValueAlias`|`table valued` 函数|
|`TextualSelect`|`TextCluse` 的 `SelectBase` 封装|
|`Values`|可作为 `FROM` 子句尾的 `VALUES`|

> - *SQLAlchemy 1.4 Selectables API*：<https://docs.sqlalchemy.org/en/14/core/selectable.html>
> - *SQLAlchemy 1.4* 可选择对象：<https://www.osgeo.cn/sqlalchemy/core/selectable.html>

###	*DML*

|*DML* 基础构建器|*SQL* 关键字|描述|
|-----|-----|-----|
|`sa.delete(table[,whereclause,bind,returning,...],**dialect_kw)`|`DELETE`| |
|`sa.insert(table[,values,inline,bind,...],**dialect_kw)`|`INSERT`| |
|`sa.update(table[,whereclause,values,inline,...],**dialect_kw)`|`UPDATE`| |

|*DML* 构建器类|描述|
|-----|-----|
|`Delete(table[,whereclause,...])`|`DELETE`|
|`Insert()`|`INSERT`|
|`Update()`|`UPDATE`|
|`UpdateBase()`|`INSERT`、`UPDATE`、`DELETE` 语句基础|
|`ValuesBase`|`INSERT`、`UPDATE` 中 `VALUES` 子句支持|

> - *SQLAlchemy 1.4 DML API*：<https://docs.sqlalchemy.org/en/14/core/dml.html>
> - *SQLAlchemy 1.4* 插入、删除、更新：<https://www.osgeo.cn/sqlalchemy/core/dml.html>

###	*SQL* 函数

|函数接口|描述|
|-----|-----|
|`AnsiFunction`|*ANSI* 格式定义的函数|
|`Function`|具名 *SQL* 函数|
|`FunctionElement`|*SQL* 函数基类|
|`GenericFunction`|通用函数|
|`register_function(identifier,fn[,package])`|关联函数名与可调用对象|

|`func` 中预定义|*SQL* 函数|描述|
|-----|-----|-----|-----|
|`array_agg`|`ARRAY_AGG`|聚合元素，返回 `sa.ARRAY` 类型|
|`max`|`MAX`| |
|`min`|`MIN`| |
|`count`|`COUNT`|缺省即 `COUNT(*)`|
|`char_length`|`CHAR_LENGTH`| |
|`concat`|`CONCAT`| |字符串连接|
|`grouping_sets`|`GTOUPING SETS`|创建多个分组集|
|`cube`|`CUBE`|生成幂集作为分组集|
|`next_value`| |下个值，需 `Sequence` 作为参数|
|`now`|`now`| |
|`random`|`RANDOM`| |
|`rollup`|`ROLLUP`| |
|`session_user`|`SESSION_USER`| |
|`sum`|`SUM`| |
|`localtime`|`localtime`| |
|`localtimestamp`|`localtimestamp`| |
|`current_date`|`CURRENT_DATE`| |
|`current_time`|`CURRENT_TIME`| |
|`current_timestamp`|`CURRENT_TIMESTAMP`| |
|`sysdate`|`SYSDATE`| |
|`user`|`USER`| |
|`current_user`|`CURRENT_USER`| |
|`coalesce`| | |

-	说明
	-	多个分组集可 `GROUP BY` 一次得到多个分组结果

|`func` 中预定义函数|假设聚合函数|描述|
|-----|-----|-----|
|`rank`|假设聚合函数 `rank`|在各组中位置，不|
|`dense_rank`|假设聚合函数 `dense_rank`| |
|`percent_rank`|假设聚合函数 `percent_rank`|在各组中百分比位置|
|`percentile_cont`|假设聚合函数 `percent_cont`| |
|`percentile_disc`|假设聚合函数 `percent_disc`| |
|`mode`|`mode`| |
|`cume_dist`|`cume_dist`|返回 `sa.Numeric` 类型|


-	*SQL* 函数军需通过 `func` 命名空间激活
	-	`sa.func` 命名空间下有部分常用 *SQL* 函数的 `GenericFunction` 预定义实现
		-	除预定义实现外，*SQLAlchemy* 不会检查 *SQL* 函数调用限制

> - *SQLAlchemy 1.4 SQL and Generic API*：<https://docs.sqlalchemy.org/en/14/core/functions.html>
> - *SQLAlchemy 1.4* 插入、删除、更新：<https://www.osgeo.cn/sqlalchemy/core/functions.html>

##	数据库类

|数据库实例类|描述|
|-----|-----|
|`sa.MetaData()`|数据库元数据|
|`sa.Table(*,table_name,metadata[,column,...])`|表|
|`sa.Column(*,column_name[,type,...])`|列|
|`sa.ForeignKey(key_name)`|外键|
|`Table.insert()`| |
|`Table.select(*args,**kw)`| |
|`sa.select(*args,**kw)`| |

|数据类型|描述|
|-----|-----|
|`sa.Integer()`| |
|`sa.String()`| |

-	说明
	-	`sa.String` 在某些数据库后端上支持不指定长度的初始化


|*DML* 方法|描述|
|-----|-----|
|`Insert.compile()`|编译为 *SQL* 语句|
|`Select.where()`| |

|`Column` 表达式|*SQL* 语句|
|-----|-----|
|`==`|`=`|
|`== None`|`is NULL`|
|`!=`|`!=`、`<>`|
|`>`|`>`|
|`+`|数值型 `+`、字符串 `||`、字符串 `concat`|
|`

-	`sa.sql.schema.Column`|























