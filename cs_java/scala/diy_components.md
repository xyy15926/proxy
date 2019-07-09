#	Scala自定义实体

##	*Class*

> - Scala是纯面向对象语言，所有变量都是对象，所有操作都是
	方法调用

-	`class`定义类，包括类名、主构造参数，其中成员可能包括

	-	普通类可用`new`创建类实例

###	成员说明

-	常/变量：**声明时必须初始化**，否则需定义为抽象类，
	相应成员也被称为抽象成员常/变量
	-	可以使用占位符`_`初始化
		-	`AnyVal`子类：`0`
		-	`AnyRef`子类：`null`
-	抽象类型：`type`声明
-	方法、函数
-	内部类、单例对象：内部类、单例对象绑定至**外部对象**
	-	内部类是路径依赖类型
	-	**不同类实例中的单例对象不同**

> - Java：仍然会为匿名类生成相应字节码文件
> - Java：*public*类成员常/变量在编译后自动生成getter、
	setter（仅变量）方法
> > -	即使是公有成员也会被编译为`private`
> > -	对其访问实际上是通过setter、getter方法（可显式调用）

####	内部类型投影

类型投影：`Outter#Inner`类型`Inner`以`Outter`类型作为前缀，
`Outter`不同实例中`Inner`均为此类型子类

```scala
class Outter{
	class Inner
	def test(i: Outter#Inner) = i
}

val o1 = new Outter
val o2 = new Outter

// 二者为不同类型
val i1 = new o1.Inner
val i2 = new o2.Inner

// 类型投影父类作为参数类型
o1.test(i2)
o2.test(i1)
```

####	单例类型

单例类型：所有对象都对应的类型，任意对象对应单例类型不同

```scala
import scla.reflect.runtime.universe.typeOf
class Dog
val d1 = new Dog
val d2 = new Dog
typeOf[d1.type] == typeOf[d2.type]		// `false`
val d3: d1.type = d1					// 两者单例类型也不同
```

#####	链式调用

```scala
class Pet{
	var name: String = _
	var age: Float = _
	// 返回类型单例类型，保证类型不改变
	def setName(name: String): this.type = {
		this.name = name
		this
	}
	def setAge(age: Float): this.age = {
		this.age = age
		this
	}
}
class Dog extends Pet{
	var weight: Float = _
	def setWeight(weight: Float): this.type = {
		this.weight = weight
		this
	}
}

// 若没有设置返回类型为单例类型`this.type`，`setAge`之后
// 将返回`Pet`类型，没有`setWeight`方法，报错
val d = new Dog().setAge(2).setWeight(2.3).setName("Tom")
```

###	*Constructor*

```scala
// 带默认参数的、私有主构造方法
class Person private(var name: String, var age: Int=10){
	// 辅助构造方法
	def this(age: Int) = {
		// 必须调用已定义构造方法
		this("", age)
	}
}
```

-	主构造方法：类签名中参数列表、类体

	-	其参数（类签名中）被自动注册为类成员变、常量

		-	其可变性同样由`var`、`val`声明
		-	其访问控制类似类似类体中成员访问控制
		-	但缺省继承父类可见性，否则为`private[this] val`

		> - 参数中私有成员变、常量没有被使用，则不会被生成
		> - `case class`主构造方法中缺省为`public val`

	-	其函数体为整个类体，创建对象时被执行

	-	可类似普通方法
		-	提供默认值来拥有可选参数
		-	将主构造方法声明为私有

-	辅助构造方法：`this`方法
	-	辅助构造方法必须以先前已定义的其他辅助构造方法、或
		主构造方法的调用开始

> - 主构造方法体现于**其中参数被自动注册为类成员**

###	类继承

```scala
// `extends`指定类继承
class Student(name: String, age: Int, var studentNo: Int)
	extends Person(name: String, age: Int){
}
```

-	必须给出父类的构造方法（可以是辅助构造方法）
	-	类似强制性C++中初始化列表

-	构造函数按以下规则、顺序执行
	-	父类、特质构造函数
	-	混入特质：多个特质按从左至右
	-	当前类构造函数

-	可用`override`重写从父类继承的成员方法、常/变量
	-	重写父类抽象成员（未初始化）可省略`override`

	> - java中方法都可以视为C++中虚函数，具有多态特性

####	提前定义、懒加载

-	由于构造方法的执行顺序问题，定义于`trait`中的抽象成员
	常/变量可能会在初始化化之前被使用
-	可以考虑使用提前定义、懒加载避免

```scala
import java.io.PrintWriter
trait FileLogger{
	// 抽象变量
	val fileName: String
	// 抽象变量被使用
	val fileOutput = new PrintWriter(fileName: String)
	// `lazy`懒加载，延迟变量创建
	lazy val fileOutput = new PrintWriter(fileName: String)
}
class Person
class Student extends Person with FileLogger{
	val fileName = "file.log"
}
val s = new {
	// 提前定义
	override val fileName = "file.log"
} with Student
```

###	访问控制

> - 缺省：Scala没有`public`关键字，缺省即为*public*
> - `protected[<X>]`：在`X`范围内的类、子类中可见
> - `private[<X>]`：在`X`范围内类可见

-	`X`可为包、类、单例对象等作用域
	-	缺省为包含当前实体的上层包、类、单例对象，即类似Java
		中对应关键字

-	特别的`private[this]`表示**对象私有**
	-	只有对象本身能够访问该成员
	-	类、伴生对象中也不能访问

		```scala
		class Person{
			private[this] age = 12
			def visit() = {
				val p = new Person
				// 报错，`age`不是`Person`成员
				println(p.age)
			}
		}
		```

###	特殊类

####	`abstract class`

抽象类：不能被实例化的类

-	抽象类中可存在抽象成员常/变量、方法，需在子类中被具体化

```scala
// 定义抽象类
abstract class Person(var name: String, var age: Int){
	// 抽象成员常量
	val gender: String
	// 抽象方法
	def print: Unit
}

// 抽象类、泛型、抽象类型常用于匿名类
val p = new Person("Tom", 8){
	override gender = "male"
	// 覆盖抽象成员可以省略`override`
	def print: Unit = println("hello")
}
```

####	`case class`

样例类

-	和普通类相比具有以下默认实现
	-	`apply`：工厂方法负责对象创建，无需`new`实例化样例类
	-	`unapply`：解构方法负责对象解构，方便模式匹配
	-	`equals`：案例类按**值比较**（而不是按引用比较）
	-	`toString`
	-	`hashCode`
	-	`copy`：创建案例类实例的浅拷贝，可以指定部分构造参数
		修改部分值

-	类主构造函数参数缺省为`val`，即参数缺省不可变公开

-	用途
	-	方便模式匹配

> - 样例类一般实例化为不可变对象，不推荐实例化为可变

```scala
case class Message(sender: String, recipient: String, body: String)
val message4 = Message("A", "B", "message")
// 指定部分构造参数的拷贝
val message5 = message.copy(sender=message4.recipient, recipient="C")
```

##	*Object*

单例对象：有且只有一个实例的特殊类，其中方法可以直接访问，
无需实例化

-	单例对象由自身定义，可以视为其自身类的单例
	-	对象定义在顶层（没有包含在其他类中时），单例对象只有
		一个实例
		-	全局唯一
		-	具有稳定路径，可以被`import`语句导入
	-	对象定义在类、方法中时，单例对象表现类似惰性变量
		-	不同实例中单例对象不同，依赖于包装其的实例
		-	单例对象和普通类成员同样是路径相关的

-	单例对象是延迟创建的，在第一次使用时被创建
	-	`object`定义
	-	可以通过引用其名称访问对象

> - 单例对象被编译为单例模式、静态类成员实现

###	*Companion Object/Class*

> - 伴生对象：和某个类共享名称的单例对象
> - 伴生类：和某个单例对象共享名称的类

-	伴生类和伴生对象之间可以互相访问私有成员
	-	伴生类、伴生对象必须定义在同一个源文件中
-	用途：在伴生对象中定义在伴生类中不依赖于实例化对象而存在
	的成员变量、方法
	-	工厂方法
	-	公共变量

> - Java中`static`成员对应于伴生对象的普通成员
> - 静态转发：在Java中调用伴生对象时，其中成员被定义为伴生类
	中`static`成员（当没有定义伴生类时）

###	`apply`

`apply`方法：像构造器接受参数、创建实例对象

```scala
import scala.util.Random

object CustomerID{
	def apply(name: String) = s"$name--${Random.nextLong}"
	def unapply(customerID: String): Option[String] = {
		val stringArray:[String] = customer.ID.split("--")
		if (stringArray.tail.nonEmpty) Some(StringArray.head)
		else None
	}
}

val customer1ID = CustomerId("Tom")
customer1ID match {
	case CustomerID(name) => println(name)
	case _ => println("could not extract a CustomerID")
}
val CustomerID(name) = customer1ID
// 变量定义中可以使用模式引入变量名
// 此处即使用提取器初始化变量，使用`unapply`方法生成值
val name = CustomerID.unapply(customer2ID).get
```

> - `unapply`、`unapplySeq`参见*cs_java/scala/entity_components*

###	应用程序对象

应用程序对象：实现有`main(args: Array[String])`方法的对象

-	`main(args: Array[String])`方法必须定义在单例对象中

-	实际中可以通过`extends App`方式更简洁定义应用程序对象
	，`trait App`中同样是使用`main`函数作为程序入口

	```c
	trait App{
		def main(args: Array[String]) = {
			this._args = args
			for(proc <- initCode) proc()
			if (util.Propertie.propIsSet("scala.time")){
				val total = currentTime - executionStart
				Console.println("[total " + total + "ms")
			}
		}
	}
	```

> - 包中可以包含多个应用程序对象，但可指定主类以确定包程序
	入口

###	`case object`

样例对象：基本同`case class`

-	对于不需要定义成员的域的`case class`子类，可以定义为
	`case object`以提升程序速度
	-	`case object`可以直接使用，`case class`需先创建对象
	-	`case object`只生成一个字节码文件，`case class`生成
		两个
	-	`case object`中不会自动生成`apply`、`unapply`方法

###	用途示例

####	创建功能性方法
	
```scala
package logging
object Logger{
	def info(message: String): Unit = println(s"IFNO: $message")
}

// `import`语句要求被导入标识具有“稳定路径”
// 顶级单例对象全局唯一，具有稳定路径
import logging.Logger.info
class Test{
	info("created projects")
}
```

##	*Trait*

特质：包含某些字段、方法的类型

-	特点
	-	特质的成员方法、常/变量可以是具体、或抽象
	-	特质不能被实例化，因而也没有参数
	-	特质可以`extends`自类

-	用途：在类之间共享程序接口、字段，尤其是作为泛型类型和
	抽象方法
	-	`extend`继承特质
	-	`with`混入可以组合多个特质
	-	`override`覆盖/实现特质默认实现

-	子类型：需要特质的地方都可以使用特质的子类型替换

> - Java：`trait`被编译为`interface`，包含具体方法、常/变量
	还会生成相应抽象类

###	*Mixin*

混入：特质被用于组合类

-	类只能有一个父类，但是可以有多个混入
	-	混入和父类可能有相同父类
	-	通过Mixin `trait`组合类实现多继承

-	多继承导致歧义时，使用最优深度优先确定相应方法

####	复合类型

复合类型：多个类型的交集，指明对象类型是某几个类型的子类型

```scala
<traitA> with <traitB> with ... {refinement}
```

###	`sealed trait`/`sealed class`

密封特质/密封类：子类确定的特质、类

```scala
sealed trait ST

case class CC1(id: String) extends ST
case object CC2 extends ST
```

-	密封类、特质只能在**当前文件被继承**
-	适合模式匹配中使用，编译器知晓所有模式，可提示缺少模式

###	自类型

自类型：声明特质必须混入其他特质，尽管特质没有直接扩展其他
特质

-	特质可以直接使用已声明自类型的特质中成员

####	细分大类特质

```scala
trait User{
	def username: String
}
trait Tweeter{
	// 声明自类型，表明需要混入`User`特质
	this: User =>
	def tweet(tweetText: String) =
		println(s"$username: $tweetText")
}
class VerifiedTweeter(val username_: String)
	extends Tweeter with User{
	def username = s"real $username_"
}
```

####	定义`this`别名

```scala
class OuterClass{
	// 定义`this`别名，下面二者等价
	outer =>
	// outer: OuterClass =>
	val v1 = "here"
	class Innerclass {
		// 方便在内部类中使用
		println(outer.v1)
	}
}
```

##	泛型

-	泛型类、泛型trait
	-	泛型类使用方括号`[]`接受类型参数

-	泛型方法：按类型、值进行参数化，语法和泛型类类似
	-	类型参数在方括号`[]`中、值参数在圆括号`()`中
	-	调用方法时，不总是需要显式提供类型参数，编译器
		通常可以根据上下文、值参数类型推断

-	泛型类型的父子类型关系不可传导
	-	可通过**类型参数注释机制**控制
	-	或使用类型通配符（存在类型）“构建统一父类”

###	存在类型/类型通配符

存在类型：`<type>[T, U,...] forSome {type T; type U;...}`
，可以视为所有`<type>[]`类型的父类

```scala
// 可以使用通配符`_`简化语法
// def printAll(x: Array[T] forSome {type T})
def printAll(x: Array[_]) = {
	for (i <- x) {
		print(i + " ")
	}
	println()
}

val a = Map(1 -> 2, 3 -> 3)
match a {
	// 类型通配符语法可以用于模式匹配，但存在类型不可
	case m: Map[_, _] => println(m)
}
```

> - 省略给方法添加泛型参数

###	类型边界约束

-	`T <: A`/`T >: B`：类型`T`应该是`A`的子类/`B`的父类
	-	描述*is a*关系

-	`T <% S`：`T`是`S`的子类型、或能经过隐式转换为`S`子类型
	-	描述*can be seen as*关系

-	`T : E`：作用域中存在类型`E[T]`的隐式值

###	型变

型变：复杂类型的子类型关系与其组件类型的子类型关系的相关性

-	型变允许在复杂类型中建立直观连接，利用重用类抽象

```scala
abstract class Animal{
	def name: String
}
case class Cat(name: String) extends Animal
case class Cat(name: String) extends Animal
```

####	*Covariant*

协变：`+A`使得泛型类型参数`A`成为协变

-	类型`List[+A]`中`A`协变意味着：若`A`是`B`的子类型，则
	`List[A]`是`List[B]`的子类型

-	使得可以使用泛型创建有效、直观的子类型关系

```scala
def ConvarienceTest extends App{
	def printAnimalNames(animals: List[Animal]): Unit = {
		animals.foreach{ animal =>
			println(animal.name)
		}
	}

	val cats: List[Cat] = List(Cat("Winskers"), Cat("Tom"))
	val dogs: List[Dog] = List(Dog("Fido"), Dog("Rex"))
	printAnimalNames(cats)
	printAnimalNames(dogs)
}
```

####	*Contravariant*

逆变：`-A`使得泛型类型参数`A`成为逆变

-	同协变相反：若`A`是`B`的子类型，则`Writer[B]`是
	`Writer[A]`的子类型

```scala
abstract class Printer[-A] {
	def print(value: A): Unit
}
class AnimalPrinter extends Printer[Animal] {
	def print(animal: Animal): Unit =
		println("The animal's name is: " + animal.name)
}
class CatPrinter extends Printer[Cat]{
	def print(cat: Cat): Unit =
		println("The cat's name is: " + cat.name)
}

val myCat: Cat = Cat("Boots")
def printMyCat(printer: Printer[Cat]): Unit = {
	printer.print(myCat)
}

val catPrinter: Printer[Cat] = new CatPrinter
val animalPrinter: Printer[Animal] = new AnimalPrinter

printMyCat(catPrinter)
printMyCat(animalPrinter)
```

-	协变泛型
	-	子类是父类的特化、父类是子类的抽象，子类实例总可以
		替代父类实例
	-	协变泛型作为成员
		-	适合位于**表示实体的类型**中，方便泛化、组织成员
		-	作为[部分]输出[成员]

	> - 注意：**可变**的协变泛型变量不安全

-	逆变泛型
	-	子类方法是父类方法特化、父类方法是子类方法抽象，子类
		方法总可以替代父类方法
	-	逆变泛型提供行为、特征
		-	适合位于**表示行为的类型**中，方便统一行为
		-	仅作为输入

-	协变泛型、逆变泛型互补
	-	对包含协变泛型的某类型的某方法，总可以将该方法扩展
		为包含相应逆变泛型的类

```scala
trait Function[-T, +R]
// 具有一个参数的函数，`T`参数类型、`R`返回类型
```

####	*Invariant*

不变：默认情况下，Scala中泛型类是不变的

-	**可变**的协变泛型变量不安全

```scala
class Container[A](value: A){
	private var _value: A = value
	def getValue: A = _value
	def setValue(value: A): Unit = {
		_value = value
	}
}

// 若`A`协变
val catContainer: Container[cat] = new Container(Cat("Felix"))
val animalContainer: Container[Animal] = catCont
animalContainer.setValue(Dog("Spot"))
val cat: Cat = catContainer.getValue
```

###	*Type Erasure*

类型擦除：编译后删除所有泛型的类型信息

-	导致运行时无法区分泛型的具体扩展，均视为相同类型
-	类型擦除无法避免，但可以通过一些利用反射的类型标签解决
	-	`ClassTag`
	-	`TypeTag`
	-	`WeakTypeTag`

	> - 参见*cs_java/scala/stdlib*中

> - Java：Java初始不支持泛型，JVM不接触泛型
> > -	为保持向后兼容，泛型中类型参数被替换为`Object`、或
		类型上限
> > -	JVM执行时，不知道泛型类参数化的实际类
> - Scala、Java编译器执行过程都执行类型擦除

###	抽象类型

抽象类型：由具体实现决定实际类型

-	特质、抽象类均可包括抽象类型

	```scala
	trait Buffer{
		type T
		val element: T
	}
	```

-	抽象类型可以添加类型边界

	```scala
	abstract class SeqBuffer extends Buffer{
		type U
		type T <: Seq[U]
		def length = element.length
	}
	```

-	含有抽象类型的特质、类经常和匿名类的初始化同时使用

	```scala
	abstract class IntSeqBuffer extends SeqBuffer{
		type U = Int
	}
	def newIntSeqBuf(elem1: Int, elem2: Int): IntSeqBuffer =
		new IntSeqBuffer{
			type T = List[U]
			val element = List(elem1, elem2)
		}
		// 所有泛型参数给定后，得到可以实例化的匿名类
	```


-	抽象类型、类的泛型参数大部分可以相互转换，但以下情况无法
	使用泛型参数替代抽象类型

	```scala
	abstract class Buffer[+T]{
		val element: T
	}
	abstract class SeqBuffer[U, +T <: Seq[U]] extends Buffer[T]{
		def length = element.length
	}
	def newIntSeqBuf(e1: Int, e2: Int): SeqBuffer[Int, Seq[Int]] =
		new SeqBuffer[Int, List[Int]]{
			val element = List(e1, e2)
		}
	```

##	包和导入

###	`package`

> - Scala使用包创建命名空间，允许创建模块化程序

-	包命名方式
	-	惯例是将包命名为与包含Scala文件目录名相同，但Scala
		未对文件布局作任何限制
	-	包名称应该全部为小写

-	包声明方式：同一个包可以定义在多个文件中
	-	括号嵌套：使用大括号包括包内容
		-	允许包嵌套，可以定义多个包内容
		-	提供范围、封装的更好控制
	-	文件顶部标记：在Scala文件头部声明一个、多个包名称
		-	各包名按出现顺序逐层嵌套
		-	只能定义一个包的内容

-	包作用域：相较于Java更加前后一致
	-	和其他作用域一样支持嵌套
	-	可以直接访问上层作用域中名称
		-	行为类似Python作用域，优先级高于全局空间中导入
		-	即使名称在不同文件中

-	包冲突方案：Java中包名总是绝对的，Scala中包名是相对的，
	而包代码可能分散在多个文件中，导致冲突

	-	**绝对包名定位包**：`__root__.<full_package_path>`
		-	导入时：`import __root__.<package>`
		-	使用时：`val v = new __root__.<class>`
	-	**串联式包语句隐藏包**：包名为路径分段串，其中非结尾
		包被隐藏

> - Scala文件顶部一定要`package`声明所属包，否则好像不会默认
	导入`scala.`等包，出现非常奇怪的错误，如：没有该方法

###	`import`

-	`import`语句用于导入其他包中成员，相同包中成员不需要
	`import`语句

	```scala
	import users._
	import users.{User, UserPreferences}
	// 重命名
	import users.{UserPreferences => UPrefs}
	// 导入除`HashMap`以外其他类型
	import java.utils.{HashMap => _, _}
	```

-	若存在命名冲突、且需要从项目根目录导入，可以使用
	`__root__`表示从根目录开始

> - `scala`、`java.lang`、`object Predef`默认导入
> - 相较于Java，Scala允许在任何地方使用导入语句

###	包对象

包对象：作为在整个包中方便共享内容、使用的容器

```scala
// in file gardening/fruits/Fruit.scala
package gardening.fruits

case class Fruit(name: String, color: String)
object Apple extends Fruit("Appple", "green")
object Plum extends Fruit("Plum", "blue")
object Banana extends Fruit("Banana", "yellow")

// in file gardening/fruits/package.scala
package gardening
package object fruits{
	val planted = List(Apple, Plum, Banana)
	def showFruit(fruit: Fruit): Unit = {
		println(s"${fruit.name}s are ${fruit.color}")
	}
}
```

-	包对象中任何定义都认为时包自身的成员，其中可以定义
	包作用域级任何内容
	-	类型别名
	-	隐式转换

-	包对象和其他对象类似，每个包都允许有一个包对象
	-	可以继承Scala的类、特质
	-	注意：包对象中不能进行方法重载

> - 按照惯例，包对象代码通常放在文件`package.scala`中

##	*Annotations*

注解：关联元信息、定义

-	注解作用于其后的首个定义、声明
-	定义、声明之前可以有多个注解，注解顺序不重要

###	确保编码正确性注解

> - 此类注解条件不满足时，会导致编译失败

####	`@tailrec`

`@tailrec`：确保方法尾递归

	```scala
	import scala.annotations.tailrec

	def factorial(x: Int): Int = {
		@tailrec
		def factorialHelper(x: Int, accumulator: Int): Int ={
			if (x == 1) accumulator
			else factorialHelper(x - 1, accumulator * x)
		}
	}
	```

###	影响代码生成注解

> - 此类注解会影响生成字节码

####	`@inline`

`@inline`：内联，在调用点插入被调用方法

-	生成字节码更长，但可能运行更快
-	不能确保方法内联，当且仅当满足某些生成代码大小的
	启发式算法时，才会出发编译器执行此操作

####	`@BeanProperty`

`@BeanProperty`：为成员常/变量同时生成Java风格getter、setter
方法`get<Var>()`、`set<Var>()`

> - 对成员变/常量，类编译后会自动生成Scala风格getter、
	setter方法：`<var>()`、`<var>_$eq()`

###	Java注解

-	Java注解有用户自定义元数据形式
	-	注解依赖于指定的*name-value*对来初始化其元素

####	`@interface`

```scala
// 定义跟踪某个类的来源的注解
@interface Source{
	public String URL();
	public String mail();
}

// Scala中注解应用类似构造函数调用
// 必须使用命名参数实例化Java注解
@Source(URL="http://coders.com",
		mail="support@coders.com")
public class MyClass

// 若注解中只有单个[无默认值]元素、名称为`value`
// 则可以用类似构造函数的语法在Java、Scala中应用
@interface SourceURL{
	public String value();
	public String mail() defualt "";
}
@SourceURL("http://coders.com/")
public class MyClass
// 若需要给`mail`显式提供值，Java必须全部使用命名参数
@Source(URL="http://coders.com",
		mail="support@coders.com")
public class MyClass
// Scala可以继续利用`value`特性
@SourceURL("http://coders.com/"
		mail = "support@coders.com")
public class MyClass
```


