---
title: Rust 标准库数据类型
categories:
  - Rust
tags:
  - Rust
  - Datatype
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:08
toc: true
mathjax: true
comments: true
description: rust标准库数据类型
---

##	通用集合类型

###	`Vec<T>`

vector允许在单独数据结构中存储多于一个的值，它们在内存中相邻
排列，vector被丢弃时，其中的数据也会被丢弃

####	存储不同类型
vector只能存储相同类型的值，因为vector必须在编译前知道所有
存储元素所需内存、允许的元素类型，否则对vector进行操作可能
会出错。但是可以使用枚举类型存储"不同类型"（Message为例）

```rust
vec![Message::Write(String::from("ab"), Message::Move{x:5, y:6}]
```

####	常用方法

```rust
let mut v:Vec<i32> = Vec::new();
let mut v = Vec::with_capacity(3);
	//为vector预先分配空间，比`new`稍有效率
	//但是这个为啥不用指定类型啊，这样怎么分配空间
let v = vec![1, 2, 3];

v.push(5)
let third:&i32 = &v[2]
	//尝试获取一个引用值，如果越界则报错
let third:Option<&i32> = v.get(2)
	//尝试获取Option<&i32>，越界则返回None

let mut v = vec![100, 32, 57]
for i in &mut v{
	*i += 50;
}

let iter = v.iter();
	//不可变引用迭代器
let iter_mut = v.iter_mut();
	//可变引用迭代器
let iter_owner = v.into_iter();
	//所有权引用迭代器
```

使用enum+match就能保证处理所有类型，不会出错

###	字符串

通常意义上的字符串往往是以下两种的”综合“

-	rust核心语言中的字符串slice`&str`：存储在别处的utf-8
	编码字节序列的引用

	>	字符串slice是&str类型，这个好像体现了rust引用更像
		指针，字符串字面值应该是一系列字节序列（流）存储，
		所以”返回值“应该是”首地址“，因此是引用类型

-	rust标准库中`String`类型，是`Vec<u8>`的封装
	-	可增长
	-	可变
	-	有所有权
	-	utf-8编码

####	索引字符串

因此`String`类型不能索引获取字符，索引操作预期是常数时间，
而utf-8字列序列并不能保证在常数时间内获取“字符”，rust需要
从头检查。另外，字符串中可能有不可见字符（如发音字符），
即**字形簇**和**字符串**不等价，此时索引的意义也不明确。

更加有价值的是使用`[]`和range创建一个字符串slice需要注意的是
如果字符串slice不是有效处utf-8编码序列，程序会在运行时
`panic!`

```rust
let len = String::from("Здравствуйте").len()
	//`len`返回的是utf-8编码序列的长度20，不是“字符”数目12

let hello = "Здравствуйте";
let s = &hello[0..4];
```
####	遍历字符串

-	返回字符Unicode值`char`类型
	```rust
	for c in "नमस्ते".chars() {
		println!("{}", c);
	}
	```
	将会打印出6个字符，两个不可见的发音字符
	```rust
	न
	म
	स
		्
	त
		े
	```
-	返回字节byte值`u8`类型
	```rust
	for b in "नमस्ते".bytes() {
		println!("{}", b);
	}
	```

####	`String`常用方法

```rust
let mut s = String::new();
let s = String::from("initial contet");

let data = "initial content";
let s = data.to_string();
let s = "initial content".to_string();

let mut s = String::from("foo");
s.push_str("bar");
let s2 = "bar";
s.push_str(&s2);
println!(s2);
	//此时，`s2`仍然可以打印出来，因为`push_str`的参数是它的
	//一个引用，应该是方法中对`&&str`类型做了处理？
s.push('l');

let s1 = String::from("hello, ");
let s2 = String::from("world";
let s3 = s1 + &s2;
	//此时`s1`所有权被转移给`s3`不能再使用

let s1 = String::from("tic");
let s2 = String::from("tac");
let s3 = string::from("toc");
//let s = s1 + "-" + &s2 + "-" + &s3;
let s = format!("{}-{}-{}", s1, s2, s3);
	//`format!`宏是更好地连接字符串的方法，且不会获取任何
	//参数的所有权
```

###	HashMap

HashMap键、值必须是同质的，相对来说使用频率较低，没有引入
`prelude`，使用之前需要用`use`关键字引入

`HashMap`默认使用一种密码学安全的哈希函数，它可以抵抗拒绝
服务（Denial of Service, DoS）攻击。然而这并不是可用的最快的
算法，不过为了更高的安全性值得付出一些性能的代价。可指定不同
*hasher*来切换为其它函数。*hasher*是一个实现了`BuildHasher`
trait的类型

####	常用函数

```rust
use std::collections::HashMap;

let mut scores = HashMap::new();
scores.insert(String::from("Blue"), 10);
scores.insert(String::from("Yellow"), 30);
	//对于没有实现`copy`trait的`string`类型，所有权将转移给
	//HashMap
scores.insert(String::from("Blue", 20);
	//之前存储的值被覆盖

scores.entry(String::from("Yellow")).or_insert(90);
scores.entry(String::from("Red")).or_insert(90);
	//`entry`以想要检查的键作为参数，返回`Entry`类型的枚举
	//代表可能存在的值，`or_insert`方法在键对应的值存在时
	//返回值`Entry`（实际上时值的可变引用），否则将参数作为
	//新值插入，并返回修改后的`Entry`

let text = "hello world wonderful word";
let mut map = HashMap::new();
for word in text.split_whitespace(){
	let count = map.entry(word).or_insert(0);
	*count += 1;
}
	//这段将在`map`中存储`text`中个单词出现次数

let teams = vec![String::from("Blue"), String::from("Yello")];
let initial_scores = vec![10, 30];
let scores: HashMap<_,_> = teams.iter.zip(initial_scores.iter()).collect();
	//`collect`可能返回很多不同的数据结构，需要显式指定`scores`
	//的类型`HashMap<_,_>`

let team_name = String::from("Blue");
let team_score = scores.get(&team_name);

for (key, val) in &scores{
	println!("{}:{}", key, val);
}
```

##	智能指针

-	指针pointer：包含内存地址的变量，这个地址引用（指向）
	其他数据
-	智能指针smart pointer：一类数据结构，表现类似于指针，
	拥有额外的元数据和功能

Rust中最常见的指针是**引用reference**，除了引用数据没有其他
特殊功能，也没有任何额外开销。

智能指针通常由结构体实现，区别于常规结构体的特在于实现了
`Deref`和`Drop`trait

>	事实上，`String`和`Vec<T>`也是智能指针

###	`Deref`trait、`DerefMut`trait

-	`Deref`trait：重载解**不可变引用**运算符`*`
-	`DerefMut`trait：重载解**可变引用**引用运算符`*`

允许智能指针结构体实例表现得像引用，可以让代码兼容智能指针
和引用

```rust
fn main(){
	let x = 5;
	let y = &x;
	let z = Box::new(5);

	assert_eq!(5, x);
	assert_eq!(5, *y);
	assert_eq!(5, *z);
```

####	自定义类型实现`Deref`trait

```rust
struct MyBox<T>(T);
	//`Box<T>`从本质上被定义为包含一个元素的元组结构体
	//类似定义自定义类型`MyBox`
impl<T> MyBox<T>{
	fn new(x: T) -> MyBox<T>{
		MyBox(X)
	}
}

use::std::Deref;
impl<T> Deref for MyBox<T>{
	type Target = T;

	fn deref(&self) -> &T{
		&sell.0
	}
		//`deref`返回引用，因为大部分使用解引用时不希望获取
		//`MyBox`内部值的所有权
}
//为`MyBox<T>`实现`Deref`trait

fn main(){
	let x = 5;
	let y = MyBox::new(x);

	println!("y = {}", *y);
		//对于`*y`Rust实际在底层`*(y.deref())`
}
```

`DerefMut`trait类似

####	隐式解引用强制多态Deref Coercions

将实现了`Deref`trait或`DerefMut`trait类型的引用转换为其他
类型的引用，通过**多次**的**隐式**转换使得实参和型参类型
一致，（这些解析发生在编译时，没有运行时损失)避免多次使用
`&`和`*` 引用和解引用，也使得代码更容易兼容智能指针和引用。

```rust
fn hello(name: &str){
	println!("hello, {}", name);
}
fn main(){
	let m = MyBox::new(String::from("Rust"));
	hello(&m);
}
```

实参类型T和型参类型U满足（间接）
-	`T: Deref<Target = U>`：`&T`转换为`&U`
-	`T: Deref<Target = U>`：`&mut T`转换为`&U`
-	`T: DerefMut<Target = U>`：`&mut T`转换为`&mut U`

相当于在引用外面添加任意层`&(*_)`、`&mut(*_)`，直到实参类型
和型参类型一致

###	`Drop`trait

`Drop`trait要求实现`drop`方法（析构函数destructor），获取
`&mut self`可变引用，智能指针离开作用域时运行`drop`方法中的
代码，用于释放类似于文件或网络连接的资源，编译器会自动插入
这些代码。

-	`Drop`trait会自动清理代码
-	所有权系统确`drop`只会在值不再使用时被调用一次

>	#todo：获取&mut self，那么之前不能获取可变引用了？

```rust
struct CustomSmartPointer{
	data: String,
}

impl Drop for CustomSmartPointer{
	fn drop(&mut self){
		println!("Dropping CustomSmartPointer with data `{}`!", self.data);
	}
}

fn main(){
	let c = CustomSmartPointer{ data: String::from("pointer1")};
	let d = CustomSmartPointer{ data: String::from("pointer2")};
	println!("CustomSmartPointer created!");
}
```

输出顺序如下，变量以被创建时相反的顺序丢弃

```rust
CustomSmartPointer created!
Dropping CustomSmartPointer with data `pointer1`!
Dropping CustomSmartPointer with data `pointer2`!
```

Rust不允许显示调用`drop`函数，因为Rust仍然会在值离开作用域时
调用`drop`函数导致**double free**的错误。如果需要提早清理，
可以使用`std::mem::drop`函数（已经位于prelude中）。

```rust
fn man(){
	let c = CustomSmartPointer{ data: String::from("some data")};
	println!("CumstomSmartPointer Created");
	drop(c);
		//调用`std::mem::drop`函数，不是`c.drop()`方法
	println!("CustomSmartPointer dropped before the end of main");
```


###	`Box<T>`

在堆上存储数据，而栈上存放指向堆数据的指针，常用于
-	类型编译时大小未知，而想要在确切大小的上下文中使用
-	大量数据希望在拷贝时不转移所有权
-	只关心数据是否实现某个trait而不是其具体的类型
	（trait对像）

```rust
let b = Box::new(5);
println!("b = {}", b);
	//可以像数据存储在栈上一样访问数据
	//box离开作用域时，栈上和指向的堆上的数据都被释放
```

####	创建递归类型

Rust需要在编译时知道类型占用的空间，而递归类型（recursive
type）中值的一部分可以时相同类型的另一个值，所以Rust无法知道
递归类型占用的空间。而box大小已知，可以在递归类型中插入box
创建递归类型

```rust
enum List{
	Cons(i32, Box<list>),
	Nil,
		//代表递归终止条件的规范名称，表示列表的终止
		//不同于`null`或`nil`
}

use List::{Cons, Nil};

fn main(){
	let list = Cons(1,
		Box::new(Cons(2,
			Box::new(Cons(3,
				Box::new(Nil))))));
}
```

###	`Rc<T>`

Rc：引用计数reference counting，记录一个值引用的数量判断
这个值是否仍然被使用，如果值只有0个引用，表示没有任何有效
引用，可以被清理。

`Rc<T>`允许多个不可变引用，让值有多个**所有者**共享数据，
引用计数确保任何所有者存在时值有效。用于在堆上分配内存供
程序多个部分读取，且在无法在编译时确定哪部分最后结束使用
（否则令其为所以者即可）

>	`Rc<T>`只适合单线程场景

```rust
enum List{
	Cons(i32, Rc<List>),
	Nil,
}
	//使用`Rc<T>`代替`Box<T>`，可以构造共享List

use List::{Cons, Nil};
use std::rc::Rc;

fn main(){
	let a = Rc::new(Cons(5, Rc::new(Cons(10, Rc::new(Nil)))));
	println!("count after creating a = {}", Rc::strong_count(&a));
		//1

	let b = Cons::(3, Rc::clone(&a));
		//`Rc::clone`并不像大多数`clone`方法一样对所有数据
		//进行深拷贝，只会增加引用计数，允许`a`和`b`**共享**
		//`Rc`中数据的所有权
		//这里可以调用`a.clone()`代替`Rc::clone(&a)`，但是
		//习惯上使用`Rc::cloen(&a)`
	println!("count after creating b = {}", Rc::strong_count(&a));
		//2

	{
		let c = Cons::(4, Rc::clone(&a));
		println!("count after creating c = {}", Rc::strong_count(&a));
			//3
	}

	println!("count after c goes out of scope = {}", Rc::strong_count(&a));
		//2
}
```

###	`RefCell<T>`

`RefCell<T>`是一个遵守*内部可变性*模式的类型，允许通过
**不可变引用**更改`T`值。实际上仍然是通过可变引用更改值，
只是获得的`T`的可变引用在`RefCell<T>`内部。

>	`RefCell<T>`同样只能应用于单线程场景

可以理解为，将Rust**静态引用**改成**时分复用**引用，Rust在
运行时进时引用检查，只要保证在运行时任意时刻满足引用规则
即可。

####	内部可变性interior mutability

Rust中的一个设计模式，允许在有不可变引用时改变数据，这违反
了引用规则，因此在该模式中使用`unsafe`代码模糊Rust通常的
可变性和引用规则。但是引用规则依然适用，只是在运行时检查，
会带来一定运行时损失。

在确保代码运行时遵守借用规则，即使编译器不能保证，可以选择
使用运用内部可变性模式的类型，涉及的`unsafe`代码被封装进
安全的API中，外部类型依然不可变


####	`Ref`、`RefMut`

-	`Ref = RefCell<T>.borrow()`：获取`T`的不可变引用
-	`RefMut = RefCell<T>.borrow_mut()`：获取`T`的一个可变引用

`Ref`和`RefMut`均是实现`Deref`trait的智能指针，`RefCell<T>`
记录当前活动的`Ref`和`RefMut`指针，调用`borrow`时，不可变
引用计数加1，`Ref`离开作用域时不可变引用计数减1

```rust
pub trait Messenger{
	fn send(&self, msg: &str);
}

pub struct LimitTracker<'a, T:'a + Messenger>{
	Messenger:&'a T,
	value: usize,
	max : usize,
}
	//这个结构体有`‘a`和`T`两个泛型参数，且`T`还以生命周期
	//注解作为trait bound

impl<'a, T> LimitTracker<a', T>
	where T: Messenger{
	//这里的`T`就没有加上`‘a`作为trait bound
	pub fn new(messenger: &T, max: usize) -> LimitTracker<T>{
		LimitTracker{
			messenger,
			value: 0,
			max,
		}
	}

	pub fn set_value(&mut self, value:usize){
		self.value = value;
		let percentage_of_max = self.max as f64 / self.max as f64;
		if percentage_of_max >= 0.75{
			self.messenger.send("Warning: over 75% of quota has been used!");
		}
	}

}

#[cfg(test)]
mod tests{
	use supper::*;
	use std::cell:RefCell;

	struct MockMessenger{
		sent_messages: RefCell<Vec<String>>,
	}
	impl MockMessenger{
		fn new() -> MockMessenger{
			MockMessenger{ sent_messages: RefCell<vec![]> }
		}
	}
	impl Messenger for MockMessenger{
		fn send(&self, message: &str){
			self.sent_messages.borrow_mut().push(String::from(message));
		}
	}

	#[test]
	fn it_send_an_over_75_percent_warning_message(){
		let mock_messenger = MockMessenger::new();
		let mut limit_tracker = LimitTracker::new(&mock_messenger, 100);
		limit_tracker.set_value(80);

		assert_eq!(mock_messenger.sent_messages.borrow().len(), 1);
	}
}
```



####	`Rc<RefCell<T>>`

`T`值可以修改，且可以被多个所有者拥有

```rust
#[derive(Debug)]
enum List{
	Cons(Rc<RefCell<i32>>, Rc<List>),
	Nil
}

use List::{Cons, Nil};
use std::rc::Rc;
use std::cell::RefCell;

fn main(){
	let value = Rc::new(RefCell::new(5));
	let a = Rc::new(Cons(Rc::clone(&value), Rc::new(Nil)));
	let b = Cons(Rc::new(RefCell::new(6)), Rc::clone(&a));
	let c = Cons(Rc::new(RefCell::new(10)), Rc::clone(&a));
	
	*value.borrow_value += 10;
	println!("a after = {:?}", a);
	println!("b after = {:?}", b);
	println!("c after = {:?}", c);
}
```

####	`RefCell<Rc<T>>`

`T`值不能改变，但是`Rc<T>`整体可以改变，此时可能出现引用循环
，导致内存泄露。引用循环是程序逻辑上的bug，Rust无法捕获。

```rust
use List::{Cons, Nil};
use std::rc::Rc;
use std::cell::RefCell;
enum List{
	Cons(i32, RefCell<Rc<list>>),
	Nil,
}
impl List{
	fn tail(&self) -> Option<&RefCell<R<List>>>{
		match *self{
			Cons(_, ref item) => Some(item),
			Nil => None,
		}
	}
}

fn main(){
	let a = Rc::new(Cons(5, RefCell::new(Rc::New(Nil))));
	println!("a initial rc count ={}", Rc::strong_count(&a));
		//1
	println!("a next item = {:?}", a.tail());
	
	let b = Rc::new(Cons(10, RefCell::new(Rc::clone(&a))));
	println!("a rc count after b creating = {}", Rc::strong_count(&a));
		//2
	println!("b initial rc count = {}", Rc::strong_count(&b));
		//1
	println!("b next item = {:?}"<, b.tail());

	if let Some(link) = a.tail(){
		*link.borrow_mut() = Rc::clone(&b);
			//此时`a`、`b`循环引用，离开作用域时，两个值的
			//引用计数因为`a`、`b`被丢弃而减1，但是它们互相
			//引用，引用计数保持在1，在堆上不会被丢弃
	}
	println!("b rc count after changing a = {}", Rc::strong_count(&b));
		//2
	println!("a rc count after changing a = {}", Rc::strong_count(&a));
		//2
}
```

####	`Weak<T>`

强引用`Rc<T>`代表共享`Rc`实例的引用，代表所有权关系，
而弱引用`Weak<T>`不代表所有权关系，不同于`Rc<T>`使用
`strong_count`计数，`Weak<T>`使用`weak_count`计数，即使
`weak_count`无需为0，`Rc`实例也会被清理（只要`strong_count`
为0）

-	`Weak<T>`指向的值可能已丢弃，不能像`Rc<T>`一样直接解引用
	，需要调用`upgrade`方法返回`Option<Rc<T>>`
-	`Weak<T>`避免`Rc<T>`可能导致的引用循环

```rust
use std::rc::{Rc, Weak};
use std::cell::RefCell;

#[derive(Debug)]
struct Node{
	value: i32,
	parent: RefCell<Weak<Node>>,
	Children: RefCell<Vec<Rc<Node>>>,
}

fn main(){
	let leaf = Rc::new(Node{
		value: 3,
		parent: RefCell::new(Weak::new()),
		children: RefCell::new(vec![]),
	});

	println!(
		"leaf strong = {}, weak = {}",
		Rc::strong_count(&leaf),
		Rc::weak_count(&leaf),
	);
		//strong = 1, weak = 0

	{
		let branch = Rc::new(Node{
			value: 5,
			parent: RefCell::new(Weak::new()),
			children: RefCell::new(vec![Rc::clone(&leaf)),
		});
		*leaf.parent.borrow_mut() = Rc::downgrade(&branch);
			//`downgrade`返回`Weak<T>`

		println!(
			"branch strong = {}, weak = {}",
			Rc::strong_count(&branch),
			Rc::weak_count(&branch),
		);
			//strong = 1, weak = 1

		println!(
			"leaf strong = {}, weak = {}",
			Rc::strong_count(&leaf),
			Rc::weak_count(&leaf),
		);
			//strong = 2, weak = 0
	}

	println!("leaf parent = {:?}", leaf.parent.borrow().upgrade());
		//`upgrade`返回`Option<Rc<T>>`，此例中因为`branch`
		//离开作用域已经丢弃，这里返回`None`
	println!(
		"leaf strong = {}, weak = {}",
		Rc::strong_count(&leaf),
		Rc::weak_count(&leaf),
	);
		//strong = 1, weak = 0
		//如此不会造成引用循环导致内存泄露
}
```

###	比较

|智能指针|数据拥有者|引用检查|多线程|
|--------|----------|--------|------|
|`Box<T>`|单一所有者|编译时执行可变（不可变）引用检查|是|
|`Rc<T>`|多个所有者|编译时执行不可变引用检查|否|
|`RefCell<T>`|单一所有者|运行时执行不可变（可变）引用检查|否|
|`Weak<T>`|不拥有数据|编译时执行可变（不可变）检查|否|



##	常用枚举类型

```rust
Option<T>{
	Some<T>,
	None,
}

some = Some(9);
some.take();
	//`take`获取`some`的值所有权作为返回值，并设置`some`为
	//`None`

Result<T, U>{
		Ok<T>,
		Err<U>,
}
```
