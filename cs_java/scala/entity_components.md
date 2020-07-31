---
title: Scala基本实体
tags:
  - Java
  - Scala
categories:
  - Java
  - Scala
date: 2019-08-02 23:17:39
updated: 2019-08-02 23:17:39
toc: true
mathjax: true
comments: true
description: Scala基本实体
---

##	*Expression*

表达式：可计算的语句

-	*value*：常量，引用常量不会再次计算
	-	不能被重新赋值
	-	类型可以被推断、也可以显式声明类型
	-	可以被声明为`lazy`，只有被真正使用时才被赋值

```scala
val constant = 1
lazy val lazy_constant = 1
var variable = 1
```

-	*variable*：变量，除可重新赋值外类似常量

###	*Unified Types*

> - Scala中所有值都有类型，包括数值、函数

####	类型层次结构

![scala_unified_types_diagram](imgs/scala_unified_types_diagram.svg)

-	`Any`：顶级类型，所有类型超类
	-	定义一些通用方法
		-	`equals`
		-	`hashCode`
		-	`toString`

-	`AnyVal`：值类型
	-	有9个预定义非空值类型
		-	`Double`
		-	`Float`
		-	`Long`
		-	`Int`
		-	`Short`
		-	`Byte`
		-	`Char`
		-	`Boolean`
		-	`Unit`：唯一单例值`()`
	-	值类型可以按以下方向转换（非父子类之间）
		![scala_value_types_casting_diagram](imgs/scala_value_types_casting_diagram.svg)

-	`AnyRef`：引用类型
	-	所有非值类型都被定义为引用类型
	-	用户自定义类型都是其子类型
	-	若Scala被应用于Java运行环境中，`AnyRef`相当于
		`java.lang.object`

-	`Nothing`：底部类型，所有类型的子类型
	-	没有值是`Nothing`类型
	-	可以视为
		-	**不对值进行定义的表达式**的类型
		-	不能正常返回的方法
	-	用途
		-	给出非正常终止的信号：抛出异常、程序退出、死循环

-	`NULL`：所有引用类型的子类型
	-	唯一单例值`null`
	-	用途
		-	使得Scala满足和其他JVM语言的互操作性，几乎不应该
			在Scala代码中使用


####	基本数据类型

|数据类型|取值范围|说明|
|-----|-----|-----|
|`Byte`|$[-2^7, 2^7-1]$||
|`Short`|$[-2^{15}, 2^{15}-1]$||
|`Int`|$[-2^{31}, 2^{31}-1]$||
|`Long`|$[-2^{63}, 2^{63}]$||
|`Char`|$[0, 2^{16}-1]$||
|`String`|连续字符串|**按值比较**|
|`Float`|32bits浮点||
|`Double`|64bits浮点||
|`Boolean`|`true`、`false`||
|`Unit`|`()`|不带任何意义的值类型|

```scala
val x = 0x2F		// 16进制
val x = 41			// 10进制
val x = 041			// 8进制

val f = 3.14F
val f = 3.14f		// `Float`浮点
val d = 3.14
val d = 0.314e1		// `Double`

val c = 'A'
val c = '\u0022'
val c = '\"'		// `Char类型`

val str = "\"hello, world\""		// 字符串
val str = """"hello, world""""		// 忽略转义

var x = true
```

> - `Unit`值`()`在概念上与`Tuple0`值`()`相同（均唯一单例值）
	（但`Tuple0`类型不存在？？？）

####	*Tuples*

元组：不同类型值的聚集，可将不同类型的值放在同一个变量中保存

-	元组包含一系列类：`Tuple2`、`Tuple3`直到`Tuple22`
	-	创建包含n个元素的元组时，就是从中实例化相应类，用
		组成元素的类型进行参数化

-	特点
	-	比较：按值（内容）比较
	-	输出：输出括号包括的内容

-	访问元素
	-	`<tuple>._<n>`：访问元组中第n个元素（从1开始）
	-	元组支持模式匹配[解构]

-	用途
	-	需要从函数中返回多个值

```scala
val t3 = ("hello", 1, 3.14)
val (str, i, d) = t3
// 必须括号包裹，否则三个变量被赋同值
println(t3._1)

val t2 = (1, 3)
val t22 = 1 -> 3
// `->`同样可以用于创建元组
```

> - 基于内容的比较衍生出模式匹配提取
> - 如果元素具有更多含义选择`case class`，否则选择元组

####	*Symbol*

符号类型：起标识作用，在模式匹配、内容判断中常用

-	特点
	-	比较：按内容比较
	-	输出：原样输出

```scala
val s: Symbol = 'start
if (s == 'start) println("start......")
```

###	运算符号

-	任何具有单个参数的方法都可以用作**中缀运算符**，而运算符
	（狭义）就是普通方法
	-	可以使用点号调用
	-	可以像普通方法一样定义运算符
		```scala
		case class Vec(val x: Double, val y: Double){
			def +(that: Vec) =
				new Vec(this.x + that.x, this.y + that.y)
		}
		```

-	表达式中运算符优先级：根据运算符第一个字符评估优先级

	-	其他未列出字符
	-	`*`、`/`、`%`
	-	`+`、`-`
	-	`:`
	-	`+!`
	-	`<`、`>`
	-	`&`
	-	`^`
	-	`|`
	-	所有字母：`[a-zA-Z]`

####	数值运算

-	四则运算
	-	`+`
	-	`-`
	-	`*`
	-	`/`
	-	`%`

-	按位
	-	`&`、`|`、`^`、`~`
	-	`>>`/`<<`：有符号移位
	-	`>>>`/`<<<`：无符号移位

-	比较运算
-	`>`、`<`、`<=`、`>=`
-	`==`/`equals`：基于内容比较
-	`eq`：基于引用比较

-	逻辑运算
	-	`||`、`&&`

####	字符串运算

> - Scala中`String`实际就是Java中`java.lang.String`类型
> > -	可调用Java中`String`所有方法
> > -	并定义有将`String`转换为`StingOps`类型的隐式转换函数
		，可用某某些额外方法

-	`indexOf(<idx>)`
-	`toUppercase`、`toLowercase`
-	`reverse`
-	`drop(<idx>)`
-	`slice<<start>,<end>>`
-	`.r`：字符串转换为正则表达式对象

###	类型推断

-	编译器通常可以推断出
	-	表达式类型：工作原理类似推断方法返回值类型
	-	**非递归方法**的返回值类型
	-	多态方法、泛型类中泛型参数类型

-	编译器不推断方法的形参类型
	-	但某些情况下，编译器可以推断作为参数传递的匿名函数
		形参类型

-	不应该依赖类型推断场合
	-	公开可访问API成员应该具有显式类型声明以增加可读性
	-	避免类型推断推断类型太具体

		```scala
		var obj = null
		// 被推断为为`Null`类型仅有单例值，不能再分配其他值
		```

###	类型别名

类型别名：**具体类型**别名

```scala
// 泛型参数必须指定具体类型
type JHashMap = java.util.HashMap[String, String]
```

##	*Code Blocks*

代码块：使用`{}`包围的表达式组合

-	代码块中最后表达式的结果作为整个代码块的结果
	-	Scala建议使用纯函数，函数不应该有副作用
	-	所以其中基本应该没有语句概念，所有代码块均可视为
		表达式，用于赋值语句

> - `{}`：允许换行的`()`，Scala中`{}`基本同`()`

###	控制表达式

####	`if`语句

```scala
if(<eva_expr>){
	// code-block if true
}else if(<eva_expr>){
	// code-block
}else{
	// code-block if false
}
```

-	返回值：对应函数块返回值

####	`while`语句

```scala
while(<eva_expr>){
	// code-block if true
}
```

-	Scala中`while`作用被弱化、不推荐使用
-	返回值：始终为`Unit`类型`()`

####	`for`语句

```scala
for{
	<item1> <- <iter1>
	<item2> <- <iter2>
	if <filter_exp>
	if <filter_exp>
}{
}
```

-	以上在同语句中**多个迭代表达式**等价于嵌套`for`
-	返回值
	-	默认返回`Unit`类型`()`
	-	配合`yield`返回值迭代器（元素会被消耗）

> - 注意：迭代器中元素会被消耗，大部分情况不应该直接在嵌套
	`for`语句中使用

###	`match`模式匹配

-	模式匹配的候选模式
	-	常量
	-	构造函数：解构对象
		-	需伴生对象实现有`unapply`方法，如：`case class`
	-	序列
		-	需要类伴生对象实现有`unapplySeq`方法，如：
			`Seq[+A]`类、及其子类
	-	元组
	-	类型：适合需要对不同类型对象需要调用不同方法
		-	一般使用类型首字母作为`case`标识符`name`
		-	对密封类，无需匹配其他任意情况的case
		-	不匹配可以隐式转换的类型
	-	变量绑定

-	候选模式可以增加*pattern guards*以更灵活的控制程序

-	模式匹配可以视为**解构已有值，将解构结果map至给定名称**
	-	可以用于**普通赋值语句**中用于解构模式
	-	显然也适合于`for`语句中模式匹配

```scala
<target> match {
	// 常量模式 + 守卫语句
	case x if x % 2 == 0 => 
	// 构造函数模式
	case Dog(a, b) => 
	// 序列模式
	case Array(_, second) =>
	// 元组模式
	case (second, _*) =>
	// 类型模式
	case str: String =>
	// 变量绑定
	case all@Dog(name, _) =>
}

// 普通赋值语句中模式匹配
val all@Array(head, _*) = Array(1,3,3)
```

> - 可在普通类伴生对象中实现`unapply`、`unapplySeq`实现模式
	匹配，此单例对象称为*extractor objects*

####	`unapply`

`unapply`方法：接受实例对象、返回创建其所用的参数

-	构造模式依靠类伴生对象中`unapply`实现
-	`unapply`方法返回值应该符合
	-	若用于判断真假，可以返回`Boolean`类型值
	-	若用于提取单个`T`类型的值，可以返回`Option[T]`
	-	若需要提取多个类型的值，可以将其放在可选元组
		`Option[(T1, T2, ..., Tn)]`

> - `case class`默认实现有此方法

	```scala
	class Dog(val name: String, val age: Int){
	// 伴生对象中定义`unapply`方法解构对象
	object Dog{
		def unapply(dog: Dog): Option[(String, Int)]{
			if (dog!=null) Some(dog.name, dog.age)
			else None
		}
	}
	```

####	`unapplySeq`

`unapplySeq`方法：接受实例对象、返回创建其所用的序列

-	序列模式依靠类伴生对象中`unapplySeq`实现
-	方法应返回`Option[Seq[T]]`

> - `Seq[A+]`默认实现此方法

```scala
// `scala.Array`伴生对象定义
object Array extends FallbackArrayBuilding{
	def apply[T: ClassTag](xs: T*): Array[T] = {
		val array = new Array[T](xs.length)
		var i = 0
		for(x <- xs.iterator) {
			array(i) = x
			i += 1
		}
		array
	}
	def apply[x: Boolean, xs: Boolean*]: Array[Boolean] = {
		val array = new Array[Boolean](xs.length + 1)
		array(0) = x
		var i = 1
		for(x <- xs.iterator){
			array(i) = x
			i += 1
		}
		array
	}
	/* 省略其它`apply`方法 */
	def unapplySeq[T](x: Array[T]): Option[IndexedSeq[T]] =
		if(x == null) None
		else Some(x.toIndexedSeq)
}
```

####	正则表达式模式匹配

```scala
import scala.util.matching.{Regex, MatchIterator}
// `.r`方法将字符串转换为正则表达式
val dateP = """(\d\d\d\d)-(\d\d)-(\d\d)""".r
val dateP = new Regex("""(\d\d\d\d)-(\d\d)-(\d\d)""")

// 利用`Regex`的`unapplySeq`实现模式匹配
// 不过此应该是定义在`Regex`类而不是伴生对象中
for(date <- dateP.findAllIn("2015-12-31 2016-01-01")){
	date match {
		case dateRegex(year, month, day) =>
		case _ =>
	}
}
// `for`语句中模式匹配
for(dateP(year, month, day) <- dateP.findAllIn("2015-12-31 2016-01-01")){
}
```

> - `scala.util.matching`具体参见`cs_java/scala/stdlib`

##	*Functions*

函数：带有参数的表达式

```scala
// 带一个参数的匿名函数，一般用作高阶函数参数
Array(1,2,3).map((x: Int) => x + 1)
Array(1,2,3).map(_+1)			// 简化写法
// 具名函数
val fx: Float => Int = (x: Float) => x.toInt
```

-	函数结构
	-	可以带有多个参数、不带参数
	-	无法显式声明返回值类型

-	函数可以类似普通常量使用`lazy`修饰，当函数被使用时才会被
	创建

-	作高阶函数参数时可简化
	-	参数类型可推断、可被省略
	-	仅有一个参数，可省略参数周围括号
	-	仅有一个参数，可使用占位符`_`替代参数声明整体

> - Java：函数被编译为内部类，使用时被创建为对象、赋值给相应
	变量
> - Scala中函数是“一等公民”，允许定义高阶函数、方法
> > -	可以传入对象方法作为高阶函数的参数，Scala编译器会将
		方法强制转换为函数
> > -	使用高阶函数利于减少冗余代码

###	函数类型

函数类型：Scala中有`Funtion<N>[T1, T2, ..., TN+1]`共22种函数
类型，最后泛型参数为返回值类型

```scala
// 实例化`Function2[T1, T2, T3]`创建函数
val sum = new Function2[Int, Int, Int] {
	def apply(x: Int, y: Int): Int = x + y
}
val sum = (x: Int, y: Int) => x + y
```

###	偏函数

偏函数：只处理参数定义域中子集，子集之外参数抛出异常

```scala
// scala中定义
trait PartialFunction[-A, +B] extends (A => B){
	// 判断元素在偏函数处理范围内
	def isDefinedAt(?ele: A)
	// 组合多个偏函数
	def orElse(?pf: PartialFunction[A, B])
	// 方法的连续调用
	def addThen(?pf: PartialFunction[A, B])
	// 匹配则调用、否则调用回调函数
	def applyOrElse(?ele: A, ?callback: Function1[B, Any])
}
```

-	偏函数实现了`Function1`特质
-	用途
	-	适合作为`map`函数参数，利用模式匹配简化代码

```scala
val receive: PartialFunction[Any, Unit] = {
	case x: Int => println("Int type")
	case x: String => println("String type")
	case _ => println("other type")
}
```

##	*Methods*

方法：表现、行为类似函数，但有关键差别

-	`def`定义方法，包括方法名、参数列表、返回类型、方法体

-	方法可以接受多个**参数列表**、没有参数列表

	```scala
	def addThenMutltiply(x: Int, y: Int)(multiplier: Int): Int = (x+y) * multiplier

	def name: String = System.getProperty("user.name")
	```

-	Scala中可以嵌套定义方法

> - Java中全在类内，确实都是方法

###	*Currying*

柯里化：使用较少的参数列表调用多参数列表方法时会产生新函数，
该函数接受剩余参数列表作为其参数

-	多参数列表/参数分段有更复杂的调用语法，适用场景
	-	给定部分参数列表
		-	可以尽可能利用类型推断，简化代码
		-	创建新函数，复用代码
	-	指定参数列表中部分参数为`implicit`

```scala
val number = List(1,2,3,4,5,6,7,8,9)
numbers.foldLeft(0)(_ + _)
// 柯里化生成新函数
val numberFunc = numbers.foldLeft(List[Int]())_
val square = numberFunc((xs, x) => xs:+ x*x)
val cube = numberFunc((xs, x) => xs:+ x*x*x)

def execute(arg: Int)(implicit ec: ExecutionContext)
```

###	隐式转换

隐式转换：编译器发现类型不匹配时，在作用域范围内查找能够转换
类型的隐式转换

-	类型`S`到类型`T`的隐式转换由`S => T`类型函数的隐式值、
	或可以转换为所需值的隐式方法定义
	-	隐式转换只与源类型、目标类型有关
	-	源类型到目标类型的隐式转换**只会进行一次**
	-	若作用域中有多个隐式转换，编译器将报错

-	适用场合：以下情况将搜索隐式转换
	-	隐式转换函数、类：表达式`e`的类型`S`不符合期望类型
		`T`
	-	隐式参数列表、值：表达式`e`类型`S`没有声明被访问的
		成员`m`

-	隐式转换可能会导致陷阱，编译器会在编译隐式转换定义时发出
	警告，可以如下关闭警告
	-	`import scala.language.implicitConversions`到隐式
		转换上下文范围内
	-	启用编译器选项`-language: implicitConversions`

```scala
import scala.language.implicitCoversions

// `scala.Predef`中定义有
// 隐式转换函数
implicit def int2Integer(x: Int) =
	java.lang.Integer.valueOf(x)
// 隐式参数列表
@inline def implicitly[T](implicit e:T) = e
```

####	隐式转换函数、类

```scala
// 隐式转换函数
implicit def float2int(x: Float) = x.toInt

// 隐式类
implicit class Dog(val name: String){
	def bark = println(s"$name is barking")
}
"Tom".bark

// 隐式类最终被翻译为
implicit def string2Dog(name: String): Dog = new Dog(name)
```

-	隐式类通过隐式转换函数实现
	-	其主构造函数参数有且只有一个
	-	代码更简洁、晦涩，类和方法绑定

####	隐式值、参数列表

-	若参数列表中参数没正常传递，Scala将尝试自动传递
	**正确类型的隐式值**

-	查找隐式参数的位置
	-	调用包含隐式参数列表的方法时，首先查找可以直接访问、
		无前缀的隐式定义、隐式参数
	-	在伴生对象中查找与隐式候选类型相关的、有隐式标记的
		成员

-	说明
	-	隐式值不能是顶级值
	-	`implicit`**能且只能**作用于最后参数列表
	-	方法才能包含隐式参数列表，**函数不可**
	-	包含隐式参数列表方法不可偏函数化

-	例1

	```scala
	abstract class Monoid[A]{
		def add(x: A, y: A): A
		def unit: A
	}
	object ImplicitTest{
		// `implicit`声明该匿名类对象为隐式值
		implicit val stringMonoid: Monoid[String] = new Monoid[String]{
			def add(x: String, y: String): Strinng = x concat y
			def unit: String = ""
		}
		implicit val intMonoid: Monoid[Int] = new Monoid[Int] {
			def add(x: Int, y: Int): Int = x + y
			def unit: Int = 0
		}

		// 定义隐式参数列表
		def sum[A](xs: List[A])(implicit m: Monoid[A]): A =
			if (xs.isEmpty) m.unit
			else m.add(xs.head, sum(xs.tail))

		def main(args: Array[String]): Unit = {
			println(sum(List(1,2,3)))
			println(sum(List("a", "b", "c")))
		}
	}
	```

-	例2

	```scala
	trait Multiplicable[T] {
		def multiply(x: T): T
	}
	// 定义隐式单例对象
	implicit object MultiplicableInt extends Multiplicable[Int]{
		def multiply(x: Int) = x*x
	}
	implicit object MultiplicableString extends Mulitplicable[String]{
		def multiply(x: String) = x*2
	}
	// `T: Multiplicable`限定作用域存在相应隐式对象、或隐式值
	def multiply[T: Multiplicable](x: T) = {
		// 调用`implicitly`返回隐式对象、或隐式值
		val ev = implicitly[Multiplcable[T]]
		ev.multiply(x)
	}
	println(multiply(5))
	println(multiply(5))
	```

###	传名参数

传名参数：仅在被使用时触发实际参数的求值运算

```scala
def calculate(input: => Int) = input * 37
// 在参数类型前加上`=>`将普通（传值）参数变为传名参数
```

-	传名参数
	-	若在函数体中未被使用，则不会对其求值
	-	若参数是计算密集、长时运算的代码块，延迟计算能力可以
		帮助提高性能
-	传值参数：仅被计算一次

####	传名参数实现while循环

```scala
def whileLoop(condition: => Boolean)(body: => Unit): Unit =
	if(condition){
		body
		whileLoop(condition)(body)
	}

var i = 2
whileLoop(i > 0){
	println(i)
	i -= 1
}
```

###	默认参数

默认参数：调用时可以忽略具有默认值的参数

-	调用时忽略前置参数时，其他参数必须带名传递
-	跳过前置可选参数传参必须**带名传参**

> - Java中可以通过剔除可选参数的重载方法实现同样效果
> - Java代码中调用时，Scala中默认参数必须、不能使用命名参数

###	命名参数调用

-	调用方法时，实际参数可以通过其对应形参名标记
	-	命名参数顺序可以打乱
	-	未命名参数需要**按照方法签名中形参顺序**放在前面

##	Exception

异常处理：`try{throw...}catch{case...}`抛出捕获异常

```c
def main(args: Array[String]){
	try{
		val fp = new FileReader("Input")
	}catch{
		case ex: FileNotFoundException => println("File Missing")
		case ex: IOException => println("IO Exception")
	}finally{
		println("finally")
	}
}
```




