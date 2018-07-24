#	Rust自定义数据类型

##	结构体struct

rust不允许只将特定字段标记为可变（很正常，因为结构体应当
作为一个整体考虑）

-	定义结构体时字段不能添加`mut`
-	声明结构体时，语法上也难以做到，字段不是单独声明

>	结构体中若有字段是引用类型，需要添加生命周期

###	普通结构体

```rust
struct stct{
	field1: i32,
	field2: String,
}

let field1 = 1;
let stct={
	field1,
	field2: String::from("fy"),
}
//变量字段同名时字段初始化简略写法

let struct1 = stct{
	field1: 1,
	..struct2
}
//结构体更新语法
```

###	元组结构体

结构体名称提供的含义，但只有字段类型没有字段名，用于命名
元组、指定类型，区别于其他相同（结构）的元组

```rust
struct tuple_stct=(i32, i32, i32)
```

###	类单元结构体unit-like struct

不定义任何字段，类似于`()`（`()`一般用于泛型中占位，表示
当前类型为空，比如`T`表示返回值泛型参数，无返回值就可以使用
`()`代替，因为Rust中类似于`typedef`用于自定义类型），常用于
在某个类型上实现trait，但不需要在 类型内存储数据时发挥作用

##	枚举enum

rust枚举更像C语言中`enum`+`struct`

-	`enum`：定义了新的枚举类型，取值范围有限
-	`struct`：枚举成员可以关联数据类型，且可以定义方法

###	枚举类型

```rust
enum IpArr{
	V4,
	V6,
}
	//基础版本

enum IpArr{
	V4(u8, u8, u8, u8},
	V6(String),
}
	//附加数据版本

enum Message{
	Quit,
	Move{x: i32, y:i32},
	Write(String),
	ChangeColor(i32, i32, i32),
}
	//含有匿名结构体版本
```

###	标准库中的枚举

####	处理`null`值

```rust
enum Option<T>{
	Some<T>,
	None,
}
```
`Option`被包含在prelude中，包括其成员，rust标准库中唯一
支持创建任何类型枚举值的枚举类型。rust不允许像有效的`T`类型
数据一样处理`Option<T>`类型数据，要求在使用之前处理为`None`
的情况，此即能够保证在可能为空的值会被处理

####	处理潜在`panic`

```rust
enum Result<T, E>{
	Ok<T>,
	Err<E>,
}
```

##	方法、关联函数

```rust
impl Message{

	fn new() -> Message{
	}
		//关联函数associated functions，没有`self`作为参数

	fn fn1(&self) -> ret_type{
	}
		//在结构体（枚举、trait对像）的上下文中定义
		//第一个参数总是`self`，代表调用方法的结构体实例

	fn fn2(mut self) -> ret_type{
	}
}
```

###	方法 Methods

-	定义方法的好处主要在于组织性，将某类型实例能做的事均放入
	`impl`块

-	方法签名中`self`会由rust根据`impl`关键字后的“替换”为
	相应类型（运行过程中是当前实例）

-	方法可以获取`self`（当前实例）所有权，常用于将`self`转换
	为其他实例，防止调用者转换之后仍使用原始实例

-	方法是rust中少数几个可以“自动引用和解引用”的地方，因为
	方法中`self`类型是明确的（调用者类型也明确），rust可以
	根据方法签名自动为对象添加`&`、`&mut`或`*`以适应方法签名，
	所以rust调用方法只有`.`，没有`->`

###	关联函数 Associated Functions

与结构体相关联，不作用于一个结构体实例，常被用于返回一个
结构体新实例的构造函数

##	Trait

将方法（关联函数）签名（可以有默认实现）组合起来、定义实现
某些目的所必需的行为的集合

```rust
pub trait Summarizable{

	// 无默认实现
	fn author_summary() -> String;

	// 有默认实现
	fn summary(&self) -> String{
		String::from("Read more...{}", self.author_summary())
	}
}
//定义trait

impl Summarizable for Message{

	fn author_summary(&self){
	}

	fn summary(&self) -> String{
	}

}
//为类型实现trait，之后就可以和普通非trait方法一样调用
```

###	默认实现

-	trait中有默认实现的方法可以不重载，实现trait就可直接
	调用，没有默认实现的方法则需要全部实现

-	默认实现重载之后不可能被调用

-	默认实现可以调用**同trait**中的其他方法，包括没有默认
	实现的方法，如此trait可以实现很多功能而只需要实现少部分

	-	同trait：trait之间本就应该保持独立，这个是trait的
		意义

	-	因为实现trait一定要实现所有没有默认实现的方法，所以
		默认实现总是“可以调用”

###	孤儿规则 Orphan Rule

orphan rule：父类型不存在

仅**trait**或**类型**位于（之一）本地crate才能实现trait，
如果没有此限制，可能出现两个crate同时对相同类型实现同一trait
，出现冲突

###	`Box<trait>` Trait对像

trait对像指向一个实现了指定trait的类型实例，Rust类型系统在
编译时会确保，任何在此上下文中使用的值会实现其trait对像的
trait，如此无需在编译时知晓所有可能类型。

####	Trait对象、泛型Trait Bound对比

-	trait对像在运行时替代多种具体类型
	-	编译时都是同质的`Box<trait>`类型
	-	只关心值反映的信息而不是其具体类型，类似于动态语言中
		**鸭子类型**
	-	编译器无法知晓所有可能用于trait对象的类型，因此也
		不知道应该调用哪个类型的哪个方法，因此Rust必须使用
		**动态**分发

	```rust
	pub trait Draw{
		fn draw(&self);
	}
	pub struct Screen{
		pub components: Vec<Box<Draw>>,
			//`Box<Draw>`就是trait对像，可以代替任何实现了
			//`Draw`trait的值
	}
	impl Screen{
		pub fn run(&self){
			for component in self.components.iter(){
				component.draw();
			}
		}
	}
	pub struct Button{
		pub width: u32,
		pub height: u32,
		pub label: String,
	}
	impl Draw for Button{
		fn Draw{
		}
	}
	```

	```rust
	// 外部crate使用时
	extern crate rust::gui;
	use rust_gui::{Screen, Button, Draw};

	struct SelectBox{
		width: u32,
		height: u32,
		options: Vec<String>,
	}
		//此类型对于`Screen`是未知的，但是`components`中仍然能够
		//包含此类型
	impl Draw for SelectBox{
		fn draw(&self){
		}
	}
	fn main(){
		let screen = Screen{
			components: vec![
				Box::new(SelectBox{
					width: 75,
					height: 10,
					option: vec![
						String::from("yes"),
						String::from("maybe"),
					],
				}),
				Box::new(Button{
					width: 50,
					height: 10,
					label: String::from("OK"),
				}),
			],
		};
		
		screen.run();
	}
	```

-	trait bound泛型类型参数结构体在编译时单态化
	-	一次只能替代一个具体类型，多个类型之间不同质
	-	单态化产生的代码进行**静态分发**

	```rust
	pub struct Screen<T: Draw>{
		pub components: Vec<T>,
			//trait bound泛型参数`T`只能替代一种类型
			//不同的实现`Draw`trait类型不能放在同一个vector中
	}
	impl<T> Screen<T>
		where T: Draw{
		pub fn run(&self){
			for component in self.components.iter(){
				component.draw();
			}
		}
	}
	```

> - 鸭子类型：如果它走起来像一只鸭子，叫起来像一只鸭子，那么
	它就是一直鸭子

> - 静态分发：编译器知晓调用何种方法
> - 动态分发：编译器在编译时不知晓调用何种方法，生成在运行时
	确定调用某种方法的代码。动态分发阻止编译器有选择的内联
	方法代码，这会禁用部分优化，但获得了额外的灵活性

####	对象安全

trait对象要求对象安全，只有**对象安全**的trait才能组成trait
对象，这有一些复杂的规则，但是实践中只涉及

-	返回值类型不为`Self`：如果trait中的方法返回`Self`类型，
	而使用trait对象后就不再知晓具体的类型，那方法就不可能
	使用已经忘却的原始具体类型（`Clone`trait不是对象安全）
-	方法没有任何泛型类型参数：具体类型实现trait时会放入具体
	类型单态化，但是使用trait对象时无法得知具体类型


###	状态模式（面向对象设计）

-	值某些内部状态，其行为随着内部状态而改变
-	内部状态由一系列集成了共享功能的对象表现，每个状态对象
	负责自身行为和需要转变为另一个状态时的规则
-	值对不同状态的行为、何时状态转移不知情，需求改变时无需
	改变值持有的状态、值实现代码，只需更新某个状态对象代码
	或者是增加更多状态对象

```rust
pub struct Post{
	state: Option<Box<State>>,
	content: String,
}
impl Post{

	pub fn add_text(&mut self, text: &str){
		self.content.push_str(&str);
	}

	pub fn request_review(&mut self){
		if let Some(s) = self.state.take(){
			//`Option<T>.take()`返回值，并设置为`None`
			self.state = Some(s.request_review())
		}
	}

	pub fn approve(&mut self){
		if let Some(s) = self.state.take(){
			self.state = Some(s.approve())
		}
	}

	pub fn content(&self) -> &str{
		self.state.as_ref().unwrap().content(&self)
			//`Option<T>.as_ref()`返回`Option<&T>`，因为参数
			//是`&self`，只能获取不可变引用
	}
}

trait State{
	fn request_review(self: Box<Self>) -> Box<State>;
		//`self: Box<Self>`意味着这个方法调用只对`Self`
		//类型的`Box`指针有效，这里`Self`表示值类型，因为
		//值的类型到struct实现trait的时候才能确定，编译时
		//应该会替换成具体类型
		
		//这个方法会获取对象的所有权（消费）

		//返回值`Box<State>`是trait对象

	fn approve(self: Box<Self>) -> Box<State>;

	fn content<'a>(&self, post:&'a Post) -> &'a str{
		""
	}
}

struct Draft{}

impl State for Draft{
	fn request_review(self: Box<Self>) -> Box<State>{
		Box::new(PendingReview{})
	}

	fn approve(self: Box<Self>) -> Box<State>{
		self
	}
}

struct PendingReview{}

impl State for PendingReview{
	fn request_review(self: Box<Self>) -> Box<State>{
		self
	}

	fn approve(self: Box<Self>) -> Box<State>{
		Box::new(Published{})
	}
}

struct Published{}

impl State for Published{
	fn request_review(self: Box<Self>) -> Box<State>{
		self
	}

	fn approve(self: Box<Self>) -> Box<State>{
		self
	}

	fn content<'a>(&self , post:&'a Post) -> &'a str{
		&post.content
	}
}
```

##	高级trait

###	Associated Type

关联类型：将类型占位符和trait相关联的方式

-	可在trait方法中使用这些占位符类型
-	实现trait时需要指定为具体类型

```rust
pub trait Iterator{
	type Item;
		//关联类型`Item`，实现时需要指定具体类型
	fn next(&mut self) -> Option<Self::Item>;
		//trait方法（签名）中使用关联类型
}
```

关联类型可以看作时trait中“泛型”（弱化版）。只能实现一次
trait，因此关联类型也只能指定一次，保证了一定的抽象

###	默认泛型类型参数

使用泛型类型参数时，可为泛型指定默认类型
`<PlaceholderType = ConcreteType>`

-	扩展类型而不破坏现有代码（普通trait改为泛型trait不需要
	改变之前实现trait的代码）
-	在特殊情况下自定义trait及其中的方法

```rust
use std::ops::Add;
#[derive(Debug, PartialEq)]
struct Point{
	x: i32,
	y: i32,
}
impl Add for Point{
	//`Add`是`+`运算符对应的trait
	//`Add`有默认类型参数，此时未指定泛型参数的具体类型，
	//`RHS`将是默认类型
	type Output = Point;

	fn add(self, other: Point) -> Point{
		Point{
			x: self.x + other.x,
			y: self.x + other.y,
		}
	}
}

trait Add<RHS=Self>{
	//`Add`trait定义，包含有泛型参数，但是在实现该trait之前
	//应该必须要为泛型指定具体类型
	//`RHS=Self`就是*默认类型参数*语法，泛型参数`RHS`默认为
	//`Self`类型（`+`左值类型）
	//RHS：right hand side
	type Output;
	fn add(self, rhs: RHS) -> Self:Output;
}
```

###	运算符重载

Rust不允许**创建**自定义运算符、重载**任意**运算符，不过
`std::ops`中的运算符、相应的trait可以通过实现相关trait重载

```rust
use std::ops::Add;
#[derive(Debug, PartialEq)]
struct Millimeters(u32);
struct Meters(u32);

impl Add<Meters> for Millimeters{
	//`Add`trait中`RHS`不是默认类型`Self`，`Add<Meters>`
	//设置`RHS`为`Meters`
	//此运算符重载允许`Millmeters`类型和`Meters`类型能够
	//直接相加
	type Output = Millimeters;
	fn add(self, other: Meters) -> Millimeters{
		Millimters(self.0 + (other.0 * 1000))
	}
}
```

###	消歧义

Rust无法避免两个trait具有相同名称的方法，也无法阻止某类型
同时实现两个这样的trait（或者是类型已经实现同名方法），此时
需要明确指定使用哪个方法

```rust
trait Pilot{
	fn fly(&self);
}
trait Wizard{
	fn fly(&self);
}
struct Human;

impl Pilot for Human{
	fn fly(&self){
		println!("this is your captain speaking");
	}
}
impl Wizard for Human{
	fn fly(&self){
		println!("up!");
	}
}
impl Human{
	fn fly(&self){
		println!("waving arms furiously!");
	}
}

fn main(){
	let person = Human;
	Pilot::fly(&person);
		//`Pilot`trait中方法的消歧义写法
	Wizard::fly(&person);
	person.fly();
		//默认调用直接实现在**类型**上的方法
	Person::fly(&person);
		//`Person`类型中方法消歧义写法，一般不使用
}
```

###	Fully Qualified Syntax

方法获取`self`参数

-	不同类型、同方法名，Rust根据`self`类型可以判断调用何函数
-	同类型、同方法名，消歧义语法可以指定调用何函数

而对于关联函数，没有`self`参数，某类型有同名的两个关联函数
时，无法使用消歧义语法指定调用何函数，需要使用完全限定语法
`<Type as Trait>::function(receiver_if_method), next_args, ...)`

当然，完全限定语法可以用于所有trait方法、关联函数场合，其中
`recevier_if_method`即表示方法中`self`参数

```rust
trati Animal{
	fn baby_name() -> String;
}
struct Dog;
impl Dog{
	fn baby_name() -> String{
		String::from("Spot")
	}
}
impl Animal for Dog{
	fn baby_name() -> String{
		String::from("puppy")
	}
}
fn main(){
	println!("A baby dog is called a {}", Dog::baby_name());
		//调用`Dog`的关联函数
	println!("A baby dog-animal is called a {}", <Dog as Animal>::baby_name());
		//完全限定语法
}
```

###	Super Trait

有时某个trait可能需要使用另一个trait的功能，要求某类型实现
该trait之前实现被依赖的trait，此所需的trait就是超（父）trait

```rust
trait OutlinePrint: fmt::Display{
	//`OutlinePrint`trait依赖于`fmt::Display`trait
	//在实现`OutlinePrint`之前，需要实现`fmt::Display`
	fn outline_print(&self){
		let output = self.to_string();
		let len = output.len();
		println!("{}", "*".repeat(len + 4));
		println!("* {} *", output);
		println!("{}", "*".repeat(len+4));
	}
}
struct Point{
	x: i32,
	y: i32,
}
impl fmt::Display for Point{
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result{
		write!(f, "({}, {})", self,x, self.y)
	}
}
impl OutlinePrint for Point{}
	//`OutlinePrint`中所有方法均有默认实现
```

##	高级类型

###	Newtype Pattern

孤儿规则限制了只有trait、类型其中只有位于当前crate时，才能对
类型实现trait，使用newtype模式可以“绕过”该限制，即创建新的
元组结构体类型，其中只包含该类型一个成员，此时封装类型对于
crate是本地的。newtype概念源自于Haskell，此模式没有运行时
性能损失，封装类型在编译器时已经省略了

```rust
use std::fmt;
struct Wrapper(Vec<String>);
impl fmt::Display for Wrapper{
	fn fmt(&self, f: &mut fmt:Formatter) -> fmt:Result{
		write!(f, "[{}]", self.0.join(","))
	}
}
fn main(){
	let w = Wrapper(vec![String::from("hello"), String::from("world")]);
	prinlnt!("w = {}", w);
}
```

但是Newtype模式中`Wrapper`是一个新类型，其上没有定义方法，
需要手动实现`self.0`的所有方法。或者，为`Wrapper`实现
`Deref`trait，并返回`self.0`，但是此方式下`Wrapper`会具有
所有`self.0`的所有方法，如果需要限制封装类型行为，只能自行
实现所需的方法。

###	`Type`创建类型别名

`type`关键字可以给予现有类型别名

-	`type`不是创建新、单独类型，而是创建别名，而newtype模式
	则是真正创建了新类型，也因此无法像newtype模式一样进行
	类型检查

	```rust
	type Kilometers = i32;
		//`type`不是创建新、单独的类型，而是赋予别名，两个类型
		//将得到相同的对待
	let x: i32 = 5;
	let y: Kilometers = 10;
	println!("x + y = {}", x + y);
		//`Kilometers`类型和`i32`类型完全相同，直接进行运算
	```

-	类型别名主要用途是避免重复

	```rust
	type Thunk = Box::<Fn() + Send + `static>;
	let f: Thunk = Box::new(|| println!("hi"));
	fn takes_long_type(f: Thunk){
	}
	fn returns_long_type() -> Thunk{
	}
	```

###	Never Type

Rust中有一个特殊的类型`!`，被称为empty type（never type)
-	`!`的正式说法：never type可以强转为其他任何类型
-	无法被创建
-	用于发散函数（diverging functions，从不返回的函数）的
	返回值
	#todo 这个和无返回值函数有何区别

```rust
let guess: u32 = metch guess.trim().parse(){
	Ok(num) => num,
	Err(_) => continue,
		//`continue`的值即为`!`，`match`分支的返回值必须相同
		//而`!`没有值，因此确定`match`的返回值类型为`u32`
};

impl<T> Option<T>{
	pub fn unwrap(self) -> T{
		// `Option::unwrap`定义
		match self{
			Some(val) => val, 
			None => panic!("called `Option::unwrap()` on a `None` value"),
				// `panic!`是`!`类型，不返回值而是终止程序
		}
	}
}

println!("forever");
loop{
	// 循环永不结束，表达式值是`!`
	// 如果加上`break`就不再如此
	println!("for ever");
}
```

###	Dynamically Sized Types

动态大小类型：“DST”或者“uniszed type”，这些类型允许处理
在运行时才知道大小的类型。Rust需要知道特定类型值需要分配的
内存空间，同类型的值**必须**使用相同数量的内存，因此必须
**将动态大小类型的值至于某种指针之后**（此即动态大小类型的
黄金规则），并且使用某些额外的元信息存储动态信息的大小。

`str`就是动态大小类型，`&str`则是两个值：`str`的地址和长度，
这样`&str`就有了一个在编译时可以知道的大小，并且`str`可以和
所有类型的指针结合`Box<str>`或`Rc<str>`。同样的，trait也是
动态大小类型，为了使用trait对象，必须将将其放入指针之后。

####	`Sized` trait

Rust自动为编译器在编译时就知道大小的类型实现`Sized` trait，
且Rust隐式的为每个泛型增加了`Sized` bound

```rust
fn generic<T>(t: T){
}
fn generic<T: Sized>(t: T){
	//实际上按照此函数头处理
	//即默认情况下，泛型参数不能是DST
}

fn generic<T: ?Sized>(t: &T){
	//`?Sized`是特殊的语法，只能用于`Sized` trait不能用于
	//其他trait，表示泛型`T`可能不是`Sized`的，此时参数类型
	//不能是`T`，必须是指针类型的
}
```

##	泛型（generic）

```rust
fn largest<T>(list: &[T]) -> T {}
//函数签名中泛型

struct Point<T>{
	x: T,
	y: T,
}
struct Point<T, U>{
	x: T,
	y: U,
}
//结构体定义中泛型

enum Option<T>{
	Some(T),
	None,
}
enum Result<T, E>{
	Ok(T),
	Err(E),
}
//枚举定义中泛型
```

###	方法实现中泛型

-	`impl`后声明泛型`impl<T>`表示`Point<T>`中的`T`是泛型而
	不是具体类型，是对所有的泛型结构体实现

	```rust
	impl<T> Point<T>{
		fn x(&self) -> &T{
		}
	}
	```

-	`impl`后不声明泛型，则表示`Point<T>`中`T`为具体类型，
	此时仅对`Point<T>`**类型**实现方法

	```rust
	impl Point<f32>{
		fn x(&self) -> f32{
		}
	}
	//仅`Point<f32>`实现此方法，其他`T`类型没有
	```

-	结构体定义中的泛型和方法签名中泛型不一定一致

	```rust
	impl<T, U> Point<T, U>{
		fn mixup<V,W>(self, other:Point<V,W>) -> Point<T,W>{
			Point{
				x: self.x,
				y: other.y,
			}
		}
	}
	```

###	trait实现中的泛型

```rust
impl<T:Display> ToString for T{
}
// 这个时标准库中的一个例子，使用了trait bounds
// 不使用trait bounds，那感觉有些恐怖。。。

```
>	trait定义中的没有泛型，但是其中可以包含泛型方法，同普通
	函数

###	泛型代码的性能

rust在编译时将代码单态化（monomorphization）保证效率，所以
rust使用泛型代码相比具体类型没有任何性能损失

>	单态化：将泛型代码转变为实际放入的具体类型

```rust
let integer = Some(5);
let float = Some(5.0);
//单态化
enum Option_i32{
	Some(i32),
	None,
}
enum Option_f64{
	Some(f64),
	None,
}
let integer = Option_i32::Some(5);
let float = Option_f64::Some(5.0);
``` 


###	Trait Bounds

指定泛型的trait bounds：限制泛型不再适合任何类型，编译器
确保其被限制为实现特定trait的类型

-	指定函数泛型trait bounds限制参数类型

	```rust
	pub fn notify<T: Summarizable>(item:T){}
	// 一个trait bound

	pub fn some_fn<T: Display+Clone, U: Debug+Clone>(t:T, u:U) -> 32{}
	// 多个trait bounds

	pub fn some_fn<T, U>(t:T, u:U) -> 32
		where T:Display + Clone,
				U: Debug + Clone{
	}
	// where从句写法
	```

-	指定方法泛型trait bounds有条件的为某些类型实现

	```rust
	impl<T: Display+PartialOrd> Point<T>{
		fn cmp_display(&self){
		}
	}
	```

###	trait和泛型的比较

trait和泛型都是**抽象**方法

-	trait从方法角度抽象
	-	定义一组公共“行为”
	-	“标记（trait bounds）”特定**类型（泛型）**

-	泛型从类型的角度抽象
	-	为**一组（trait bounds）**类型定义“项”`struct`、`enum`
	-	为**一组（trait bounds）**类型实现函数、trait

-	trait的定义中不应该出现泛型
	-	trait本意应该是定义一组“行为”，需要特定类型实现其方法
		（当然有的方法有默认实现），其对应的“对象”不是类型而
		是方法，与泛型用途无关
	-	trait中定义泛型无意义，trait只是一个“包裹”，真正实现
		的是其中的方法，如有必要，定义含有泛型参数的方法即可
	-	若trait中可以使用泛型，则**有可能**对不同的泛型具体
		类型实现“相同”（函数签名没有泛型参数）函数
		（在trait中有关联类型提供略弱与泛型的功能）

