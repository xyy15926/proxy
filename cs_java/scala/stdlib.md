---
title: 标准库
tags:
  - Java
  - Scala
categories:
  - Java
  - Scala
date: 2019-08-01 20:04:09
updated: 2019-08-01 20:04:09
toc: true
mathjax: true
comments: true
description: 标准库
---

##	*Scala Package*

###	`Any`

```scala
class Any{
	// 判断对象是否是否为类型`T`实例
	def isInstanceOf[T]
	// 将对象强转为类型`T`实例
	def asInstanceOf[T]
}
```

```scala
1.asInstanceOf[String]				// 报错，未定义隐式转换函数
1.isInstanceOf[String]				// `false`
List(1),isInstanceOf[List[String]]	// `true`，泛型类型擦除
List(1).asInstanceOf[List[String]]	// 成功，泛型类型擦除
```

###	`Option`

```scala
class Option[T]{
	// `Some`实例：返回被被包裹值
	// `None`实例：返回参数（默认值）
	def getOrElse(default?: T)
}

class Some[T](s?: T) extends Option[T]
object None extends Option[_] ???
```

-	推荐使用`Option`类型表示可选值，**明示**该值可能为`None`

-	`Option`类型可以被迭代
	-	`Some(s)`：唯一迭代`s`
	-	`None`：空

	```scala
	val a = Some("hello")
	a.foreach(x => println(x.length))
	```

###	`Predef`

```scala
object Predef extends LowPriorityImplicits{
	// 返回运行过程中类型，具体实现由编译器填补
	def classOf[T]: Class[T] = null
}
```

###	`List`

##	`collection`

###	`mutable`

####	`Map`

```scala
val a=Map((3,4), 5->6)
// 两种创建`Map`、二元组形式等价
a.map{case (a, b) => println(a, b)}
// `{case...}`为偏函数（或`Function1`）
```

###	`immutable`

##	`reflect`

###	`runtime`

####	`universe`

> - `universe`：提供一套完整的反射操作，可以反思性的检查
	类型关系，如：成员资格、子类型

```scala
// 返回类型`T`“类型值”，可以用于比较
typeOf[T]
```

#####	*TypeTag*

-	`TypeTag`：提供**编译时具体类型的信息**
	-	能获取准确的类型信息，包括更高的泛型类型
	-	但无法获取运行时**值的类型信息**

	```scala
	import scala.reflect.runtime.universe.{TypeTag, TypeRef, typeTag}

	// 声明隐式参数列表
	def recognize[T](x: T)(implicit tag: TypeTag[T]): String =
		tag.tpe match {
			case TypeRef(utype, usymbol, args) =>
				List(utype, usymbol, args).mkString("\n")
		}

	val list: List[Int] = List(1,2)
	val ret = recognize(list)

	// 显式实例化`TypeTag`
	val tag = typeTag[List[String]]
	```

-	`WeakTypeTag`：提供**编译时包括抽象类型的信息**
	-	`WeakTypeTag`可以视为`TypeTag`的超集
	-	若有类型标签可用于抽象类型，`WeakTypeTag`将使用该标记

	```scala
	import scala.reflect.runtime.universe.{WeakTypeTag, TypeRef, weakTypeRef}

	// 声明隐式参数列表
	def recognize[T](x: T)(implicit tag: WeakTypeTag[T]): String =
		tag.tpe match {
			case TypeRef(utype, usymbol, args) =>
				List(utype, usymbol, args).mkString("\n")
		}
	abstract class SAClass[T]{
		// 抽象类型
		val list: List[T]
		val result = Recognizer.recognize(list)
		println(result)
	}
	new SAClass[Int] { val list = List(2,3)}

	// 显式实例化`WeakTypeTag`
	val tag = weakTypeTag[List[String]]
	```

> - 当需要`TypeTag[T]`、`WeakTypeTag[T]`类型的隐式值`tag`时，
	编译器会自动创建，也可以显式实例化
> - 以上类型探测**通过反射实现**，编译器根据传递实参推断泛型
	参数`T`，由此确定特定类型标签隐式值

###	`ClassTag`

`ClassTag`：提供关于**值的运行时信息**

-	不能在更高层次上区分，如：无法区分`List[Int]`、
	`List[String]`
-	是经典的老式类，为每个类型捆绑了单独实现，是标准的
	类型模式

```scala
import scala.reflect.{ClassTag, classTag}

// def extract[T: ClassTag](list: List[Any]) =
def extract[T](list: List[Any])(implicit ct: ClassTag[T]) =
	list.flatMap{
		case element: T => Some(element)
		// 以上被翻译为如下，告诉编译器`T`的类型信息
		// case element @ ct(_: T) =>
		// 否则泛型`T`被删除，编译不会中断，但是无法正确工作
		case _ => None
	}

val list: List[Any] = List(1, "string1", List(), "string2")
val rets = extract[String](list)

// 显式实例化`ClassTag[String]`
val ct = classTag[String]
```

-	当需要`ClassTag[T]`类型的隐式值`ct`时，编译器会自动创建
	-	也可以使用`classTag`显式实例化

-	`ClassTag[T]`类型值`ct`存在时，编译器将自动
	-	包装`(_:T)`类型模式为`ct(_:T)`
	-	将模式匹配中未经检查的类型测试模式转换为已检查类型

##	`util`

###	`matching`

####	`Regex`

```scala
import scala.util.matching.{Regex, Match}

class Regex(?pattern: String, ?group: String*){
	def findAllIn(source: CharSequence): MatchIterator
	def findAllMatchIn(source: CharSequence): Iterator[Match]
	def findFirstIn(source: CharSequence): Option[String]
	def findFirstMatchIn(source: CharSequence): Option[Match]
	def replaceAllIn(target: CharSequence, replacer: (Match) => String): String
	def replaceAllIn(target: CharSequence, replacement: String): String
	def replaceFirstIn(target: CharSequence, replacement: String): String
	// 此`unapplySeq`就应该是定义在类中
	def unapplySeq(target: Any): Option[List[String]] = target match{
		case s: CharSequence => {
			val m = pattern matcher s
			if (runMatcher(m)) Some((1 to m.groupCount).toList map m.group)
			else None
		}
		// 等价于重载
		case m: Match => unapplySeq(m.matched)
		case _ => None
	}
}

class Match{
	def group(?key: Int): String
	def group(?key: String): String
}
```

```scala
val keyValPattern: Regex = "([0-9a-zA-Z-#() ]+): ([0-9a-zA-Z-#() ]+)".r
// `String.r`可使任意字符串变成正则表达式
// 使用括号同时匹配多组正则表达式

val input: String =
"""backgroud-color: #A03300l
  |background-image: url(img/header100.png);""".stripMargin

for(patternMatch(attr, value) <- keyValPattern.findAllMatchIn(input))
	println(s"key: ${attr}, value: ${value}")
```

####	

