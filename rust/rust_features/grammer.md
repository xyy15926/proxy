#	Rust语法

##	Rust是基于表达式的语言

表达式返回一个值，而语句不返回，rust中除两种语句外，全是
表达式

-	`let`引入绑定
	-	可变绑定赋值是表达式，返回空tuple
	-	声明后初始化绑定？#todo
-	表达式语句：表达式后跟分号转换为语句

代码中rust希望语句后跟语句，使用分号分隔表达式，所以rust
看起来和其他大部分分号结尾语言相似

`{}`包裹的代码块内最后一“句”没有以";"结尾，那么是表达式，且
返回该表达式的值，整个代码块可以看作是表达式，否则为语句，
没有返回值，函数同

##	模式匹配

###	Refutability（可反驳性）

-	refutable（可反驳的）：对某些可能值匹配失败的模式，
	`if let`、`while let`只能接受可反驳的模式，因为这就用于
	处理可能的失败
-	irrefutable（不可反驳的）：能匹配任何传递的可能值，`let`
	语句、函数参数、`for`循环只能接受不可反驳的模式，因为
	通过不匹配值的程序无意义

>	可能值是指“类型”相同，可用于匹配的值

```rust
let Some(x) = some_optiona_value;
	//`Some(x)`是refutable模式，若`some_optional_value`为
	//None，则此时无法成功匹配，此语句可能无法正常工作
if let x = 5 {
	//`x`是inrefutable模式，对所有可能值都可以匹配，此语句
	//无意义
	println!("{}", x);
}
```

###	Refutable

####	`match`控制流

-	各分支模式同值“类型”必须完全一致才能匹配
-	返回类型必须一致，如果有返回值
-	匹配必须是穷尽的，可以使用通配符`_`（匹配所有的值）代替
	-	match是匹配到就退出，不像switch一样会继续下沉
	-	通配符不总是需要的，对于枚举类型只要含有所有枚举
		成员的分支即可

```rust
let x = Some(5);
let y = 10;

match x{
	Some(50) => println!("got 50"),
		//`Some(50)`这个模式规定了“值”
	Some(y) => println!("matched, y = {:?}", y),
		//match表达式作用域中的`y`会覆盖周围环境的`y`
	_ => println!("default case, x = {:?}", x),
}
println!("at the end: x = {:?}, y = {:?}", x, y);
```

####	`if let[else]`简洁控制流

-	只匹配关心的一个模式
-	可以添加`else`语句，类似`match`通配符匹配
-	`if let`、`else if`、`else if let`等相互组合可以提供更多
	的灵活性
	-	`else if`可以不是模式匹配
	-	各`if`后的比较的值可以没有关联
-	没有穷尽性检查，可能会遗漏一些情况

```rust
fn main(){
	let favorite_color: Option<&str> = None;
	let is_tuesday = false;
	let age: Result<u8, _> = "34".parse();

	if let Some(color) = favorite_color{
		//模式匹配
		//注意这里是`=`而不是一般值比较`==`
		//`while`同
		println!("favorite color is {}", color};
	}else if is_tuesday{
		//普通`if`条件语句
		println!("Tuesday is green day!");
	}else if let Ok(age) = age{
		//`age`变量覆盖原变量
		//此时`age`是`u8`类型
		if age > 30 {
			//因此此条件不能提出，因为两个`age`变量不同，
			//不能共存在同一条语句中
			println!("Using purple as the background color");
		}else{
			println!("Using orange as the background color");
		}
	}else{
		println!("Using blue as the background color");
	}
}
```

###	Irrefutable

####	`while let`条件循环

和`if let`条件表达式类似，循环直到模式不匹配

```rust
fn main(){
	let mut stack = Vec::new();
	stack.push(1);
	stack.push(2);
	stack.push(3);
	while let Some(top) = stack.pop(){
		println!("{}", top);
	}
}
```

####	`for`解构

```rust
fn main(){
	let v = vec!['a', 'b', 'c'];
	for (index, value) in v.iter().enumerate(){
		//解构tuple
		println!("{} is at index {}", value, index);
	}
}
```

####	`let`解构

`let`语句本“应该“看作是模式匹配

```rust
let PARTTERN = EXPRESSION;
	//这样就和`if let`模式匹配中`=`一致
	//应该可以把`=`看作是模式匹配的符号

let x = 5;
	//这里`x`是一个模式“变量”
let (x, y, z) = (1, 2, 4);
	//`(x, y, z)`是模式“3元元组“
	//解构元组
```

####	函数参数

类似于`let`语句，函数参数也“应该”看作是模式匹配

```rust
fn foo(x: i32){
	//`x`表示模式“变量”
}
fn print_coordinates(&(x, y): &(i32, i32)){
	//这里`(x, y)`是模式“元组”
	//但是这里处于函数变量类型的要求，有点像元组结构体
	//调用时使用元组即可
	println!("Current location: ({}, {})", x, y);
}
```
###	模式匹配用法

####	`|`、`...`“或“

-	`|`“或”匹配多个模式
-	`...`闭区间范围模式，仅适用于数值、`char`值

```rust
let x = 1;
match x {
	1 => println!("one"),
		//`1`是模式“字面值”
	2 | 3 => println!("two or three"),
		//`|`分隔，匹配多个模式
	5...10 => println!("through 5 to 10"),
		//`...`表示匹配一个**闭区间**范围的值
		//这个语法只能用于数字或者是`char`值，因为编译器会
		//检查范围不为空，而只有数字、`char`值rust可以判断
	_ => println!("anything"),
}

let x = 'c';
match x {
	'a'...'j' => println!("early ASCII letter"),
	'k'...'z' => println!("late ASCII letter"),
	_ => println!("something else"),
}
```

####	`_`、`..`”忽略“

-	`_`忽略整个值

	```rust
	fn foo(_: i32, y: i32){
		//函数签名中忽略整个值
		println!("this code only use the y parameter: {}", y);
	}
	fn main(){
		foo(3, 4);

		let mut setting_value = Some(5); let new_setting_value = Some(10);
		match (setting_value, new_setting_value){
			(Some(_), Some()) => {
				//嵌套`_`忽略部分值
				//此时没有任何值所有权
				println!("can't overwrite an exsiting customized value");
			}
			_ => {
				setting_value = new_setting_value;
			}
		}
		println!("setting is {:?}", setting_value);
	}
	```

-	`_var`变量名前添加下划线忽略未使用变量，此时值所有权
	仍然会转移，只是相当于告诉编译器忽略该未使用变量
	```rust
	fn main(){
		let _x = 5;
			//两个变量未使用，但是编译时只有一个warning
		let y = 10;

		let s = Some(5);
		if let Some(_s) = s {
			//值所有权已转移，`s`不能继续使用
			//编译器忽略未使用`_s`，不给出warning
			println!("get a integer");
		}
	}
	```

-	`..`忽略剩余值
	-	`..`的使用必须无歧义
	-	对于结构体即使只有一个field，需使用`..`忽略剩余值，
		不能使用`_`
	```rust
	fn main(){
		let numbers = (2, 4, 6, 8, 10);
		match numbers{
			(first, .., last) => {
				println!("Some numbers: {}, {}", first, last);
			}
		}
	}
	```

```rust
let (x, _, z) = (1, 2, 4);
	//`_`忽略模式中各一个值
let (x, .., z) = (1, 2, 3, 4);
	//`..`忽略模式中多个值
```

####	解构结构体

```rust
struct Point{
	x: i32,
	y: i32,
}

fn main(){

	let p = Point{x: 0, y: 7};
	let Point{x: a, y: b} = p;
		//`Point{x: a, y: b}`是模式”Point结构体“
		//解构结构体

	let p = Point{x: 0, y: 7};
	let Point{x, y} = p;
		//模式匹配解构结构体简写
		//只要列出结构体字段，模式创建相同名称的变量

	let p = Point{x: 0, y: 7};
	match p {
		Point {x, y: 0} => println!("on the x axis at {}", x),
		Point {x: 0, y} => println!("on the y axis at {}", y),
		Point {x, y} => println!("on neither axis: ({}, {})", x, y),
			//这里没有`_`通配符，因为`Point {x, y}`模式已经
			//是irrefutable，不需要
	}
}
```

####	解构枚举

```rust
enum Message{
	Quit,
	Move{x: i32, y:i32},
	Write(String),
	ChangeColor(i32, i32, i32),
}
fn main(){

	let msg = Message::ChangeColor(0, 160, 255);

	let p = match msg{
		//这里`match`返回值必须类型完全相同
		Message::Move{x, y} if x == 0 => (x, y),
			//对于`Message`中的匿名结构体类型的成员，匿名
			//结构体没有枚举类型外的定义、名称，无法、也
			//不应该直接获取结构体
		Message::Write(ref str) => {
			//`Message::Write`不是`Message`的一个枚举成员
			//必须`Message::Write(str)`才是（能够匹配）
			println!("write {}", str);
			(1,1)
		},
		Message::ChangeColor(..) => (1,0),
			//类似的，需要使用`..`忽略值，仅`ChangeColor`
			//不是`Message`成员
		_ => {
			println!("quit");
			(0,0)
		},
	};
}
```

####	`&`、`ref`、`ref mut`”引用“

-	`&`匹配引用，“获得”值

	```rust
	let points = vec![
		Point {x: 0, y: 0},
		Point {x: 1, y: 5},
		Point {x: 10, y: -3},
	}

	let sum_of_squares: i32 = points
		.iter()
		.map(|&Point {x, y}| x * x + y * y)
			//`&`匹配一个引用
		.sum();
	```

-	`ref`匹配值，“获得”不可变引用

	```rust
	let robot_name = Some(String::from("Bors"));
	match robot_name{
		Some(ref name) => println!("found a name: {}", name),
			//使用`ref`获取不可变引用才能编译成功
			//否则所有权转移，之后报错
		None => (),
	}
	println!("robot_name is: {:?}", robot_name);
	```

-	`ref mut`匹配值，“获得”可变引用

	```rust
	let robot_name = Some(String::from("Bors"));
	match robot_name{
		Some(ref mut name) => *name = String::from("NewName"),
		None => (),
	}
	println!("robot_name is: {:?}", robot_name);
	```

####	`if`match guard

匹配守卫*match guard*：放在`=>`之前的`if`语句，`match`
分支的额外条件，条件为真才会继续执行分支代码

```rust
let num = Some(4);
let y = 5;
match num{
	Some(x) if x < y => println!("'less than y: {}", x),
	Some(x) => println!("{}", x),
	None => (),
}

let x = 4;
let y = false;
match x {
	4 | 5 | 6 if y => println!("yes"),
		//`4 | 5 | 6`整体作为一个模式，match guard作用于
		//模式整体，而不是单独的`6`
	_ => println!("no"),
}
```

####	`@`绑定

`@`允许在创建存放值的变量时，同时测试值是否匹配模式

```rust
enum Message{
	Hello { id: i32},
}

let msg = Message::Hello { id: 5 };
match msg{
	Message::Hello{ id: id_variable @ 3...7 } => {
		//匹配结构体模式（值绑定`id_variable`）&&值在`3...7`范围
		println!("Found an id in range: {}", id_variable)
	},
	Message::Hello{ id: 10...12 } => {
		//此分支模式指定了值的范围，但是没有绑定值给变量`id`
		//结构体匹配简略写法不能应用与此
		println!("Found an id in another range")
	}
	Message::Hello{ id } => {
		//此分支结构体匹配简略写法，值绑定于`id`
		println!("Found some other id: {}", id)
	},
}
```

##	闭包closures

闭包是可以保存进变量或作为参数传递给其他函数的匿名函数，
可以在一个地方创建闭包，而在不同的上下文执行闭包。和函数
的区别在于，其可以捕获调用者作用域中的值，当然这会有性能
损失，如果不需要捕获调用者作用域中的值可以考虑使用函数

```rust
let closures = |param1, param2|{
	...
	expression
}
```

-	闭包参数：**调用者**使用，
	-	创建闭包赋值给变量，再通过变量调用闭包
	-	创建闭包作为参数传递，其他函数调用
-	捕获环境变量：创建闭包作为参数传递，直接使用周围环境变量

###	闭包类型推断和注解

闭包不要求像函数一样需要在参数和返回值上注明类型，函数需要
类型注解因为其是需要暴露给的显示接口的一部分，而闭包不用于
作为对外暴露的接口

-	作为匿名函数直接使用，或者存储在变量中
-	通常很短，使用场景上下文比较简单，编译器能够推断参数和
	返回值类型

当然，闭包也可以添加注解增加明确性

```rust
fn  add_one_v1   {x: u32} -> u32 { x + 1 };
let add_one_v2 = |x: u32| -> u32 { x + 1 };
let add_one_v3 = |x|             { x + 1 };
let add_one_v4 = |x|               x + 1;
	//闭包体只有一行可以省略`{}`
```
Rust会根据闭包出调用为每个参数和返回值推断类型，并将其锁定，
如果尝试对同一闭包使用不同类型的参数调用会报错

```rust
let example_closure = |x| x;
let s = example_closure(String::from("hello"));
	//此时已经锁定闭包参数、返回值类型
let n = example_closure(5);
	//尝试使用`i32`类型调用闭包会报错
```

###	`Fn`trait bound

每个闭包实例有自己独有的匿名类型，即使两个闭包有相同的签名，
其类型依然不同。为了定义使用闭包的结构体、枚举、函数参数，
（这些定义中都需要指定元素类型），需要使用泛型和trait 
bound

-	`FnOnce`：获取从周围环境捕获的变量的所有权，因此只能调用
	一次，即`Once`的含义
-	`Fn`：获取从周围环境捕获的变量的不可变引用
-	`FnMut`：获取从周围环境捕获的变量的可变引用

所有的闭包都实现了以上3个trait中的一个，Rust根据闭包如何
使用环境中的变量推断其如何捕获环境，及实现对应的trait

```rust
struct Cacher<T>
	where T: Fn(u32) -> u32{
		//`T`的类型中包括`Fn`、参数、返回值三个限定
	calculation: T,
	value: Option<u32>,
}
```

>	函数实现了以上**全部**3个`Fn`trait

###	`move`关键字

`move`关键字强制闭包获其捕获的环境变量的所有权，在将闭包
传递给新线程以便将数据移动到新线程时非常实用

```rust
fn main(){
	let x = vec![1, 2, 3]
	let equal_to_x = move |z| z == x;
	println!("can't use x here: {:?}", x);
		//此时`x`的所有权已经转移进闭包，不能在闭包外使用
	let y = vec![1, 2, 3];
	assert!(equal_to_x(y));
```

##	迭代器Iterator

迭代器负责遍历序列中的每一项和决定序列何时结束的逻辑。Rust
中迭代器时惰性的，直到调用方法”消费“迭代器之前都不会有效果

###	`Iterator`trait

迭代器都实现了标准库中`Iterator`trait

```rust
trait Iterator{
	type Item;
		//定义`Iterator`的关联类型
	fn next(&mut self) -> Option<Self::Item);
		//参数是`&mut self`，要求迭代器是`mut`类型
		//`next`方法改变了迭代器中用来记录序列位置的状态
	//methods with default implementation elided
}
```

###	`next`方法

`next`是`Iterator`唯一要求被实现的方法，其返回迭代器中封装
在`Some`中的一项（**消费**迭代器中的一项），迭代器结束时，
返回`None`。

```rust
#[test]
fn iterator_demostration(){
	let v1 = vec![1, 2, 3];
	let mut v_iter = v1.iter();
		//注意`v_iter`声明为`mut`
		//使用`for`循环时无需使`v1_iter`可变，`for`会获取其
		//所有权并在后台使其可变

	assert_eq!(v1_iter.next(), Some(&1));
	assert_eq!(v1_iter.next(), Some(&2));
	assert_eq!(v1_iter.next(), Some(&3));
		//真的很难理解，rust中&integer是怎么比较的
	assert_eq!(v1_iter.nett(), None));
}
```

###	消费适配器Comsuming Adaptors

`Iterator`trait中定义，调用`next`方法，消耗迭代器

```rust
#[test]
fn iterator_sum(){
	let v1 = vec![1, 2, 3];
	let v1_iter = v1.iter();

	let total:i32 = v1_iter.sum();
		//`sum`获取迭代器所有权，`v1_iter`不能继续使用
	assert_eq!(total, 6);
```

###	迭代器适配器Iterator Adaptors

`Iterator`trait中定义，将当前迭代器变为其他迭代器，同样是
惰性的，必须调用消费适配器以便获取迭代适配器的结果

```rust
let v1:Vec<i32> = vec![1, 2, 3];
v1.iter().map(|x| x + 1);
	//这里因为没有调用消费适配器，其实没有做事
let v2:Vec<_> =  v1.iter().map(|x| x + 1).collect();
assert_eq!(v2, vec![2, 3, 4])

```

```rust
#[derive(PartialEq, Debug)]
struct Shoe{
	size: u32,
	style: String,
}

fn shoes_in_my_size(shoes: Vec<Shoe>, show_size: u32) -> Vec<Shoe>{
	shoes.into_iter()
			//获取vector所有权的迭代器
		.filter(|s| s.size == show_size)
			//这里使用闭包获取外部环境变量
		.collect()
}

#[test]
fn filter_by_size(){
	let shoes = vec![
		Shoe{size: 10, style: String::from("sneaker")},
		Shoe{size: 13, style: String::from("sandal")},
		Shoe{size: 10, style: String::from("boot")},
	];

	let in_my_size = shoes_in_my_size(shoes, 10);

	assert_eq!(
		in_my_size,
		vec![
			Shoe{size: 10, style: String::from("sneaker")},
			Shoe{size: 10, style: String::from("boot")},
		]
	);
}
```

###	实现`Iterator`

```rust
struct Counter{
	count: i32,
}
impl Counter{
	fn new() -> Counter{
		Counter{count: 0}
	}
}

impl Iterator for Counter{
	type Item = u32;
		//迭代器将返回`u32`值集合
	
	fn next(&mut self) -> Option<Self::Item>{
		self.count += 1;

		if self.count < 6{
			Some(self.count)
		}else{
			None
		}
	}
}

#[test]
fn using_other_iterator_trait_methods(){
	let sum: u32 = Counter::new().zip(Counter::new().skip(1))
								.map(|(a, b)| a * b)
								.filter(|x| x % 3 == 0)
								.sum();
	assert_eq!(18, sum);
```

##	并发（并行）

多线程可以改善性能，但是也会增加复杂性

-	竞争状态Race Conditions：多个线程以不一致的顺序访问资源
-	死锁Dead Lock：线程互相等待资源释放，阻止继续运行
-	只会在特定情况出现、无法稳定重现的bug

线程模型

-	`1:1`模型：一个OS线程对应一个语言线程，语言调用操作系统
	API创建线程，性能较好
-	`M:N`模型：语言有自己的线程实现，其提供的线程称为
	**绿色**(green)线程，M个绿色线程对应N个OS线程，更好的
	运行控制、更底的上下文切换成本

Rust为了更小的运行时（这里表示二进制文件中语言自身提供的
代码）考虑，标准库中只提供了`1:1`线程模式实现。可以通过一些
crate扩展`M:N`线程模式。

###	`spawn`创建新线程

`std::thread::spawn`接受一个闭包作为参数，返回`JoinHandle`
类型的句柄。作为`spawn`参数的闭包和一般的闭包有些不同，
线程直接独立执行，所以此时闭包捕获外部环境变量不能按照默认
的获取不可变引用，因为此时捕获的变量值可能已经被丢弃，必须
使用`move`关键字获取所有权，而一般的闭包是顺序执行的，没有
特殊需要可以直接获取不可变引用，而能够保证值不被丢弃。

```rust
use std::thread;
use std::time:Duration;
fn main(){
	let handle = thread::spawn(|| {
		//`thread::spawn`接受一个闭包作为参数，返回
		//`JoinHandle`类型的值（不是引用）
		for i in 1..10{
			println!("number {} from spwaned thread!", i);
			thread::sleep(Duration::from_millis(1));
		}
	);

	for i in 1..5{
		println!("number{} from main thread!", i);
		thread::sleep(Duration::from_millis(1));
	}

	handle.join().wrap();
		//`JoinHandle.join()`将阻塞直到其对应的线程结束
		//如果调用`join`，spawn线程可能无法执行完毕
		//因为主线程执行完，整个进行结束
		//注意`join`调用的位置决定阻塞的位置

	let v = vec![1, 2 , 3];
	let handle = thread::spawn(move || {
		//这里`move`关键字将捕获的外部环境中变量`v`所有权
		//移入spawn线程，否则无法正常编译
		prinln!("here's a vector: {:?}", v);
	});
	handle.join().unwrap();
}
```

###	消息传递

Rust中实现消息传递并发的主要工具是通道（channel）

####	`mpsc::channel`

mpsc：multiple producer single consumer，多个生产者，单个
消费者，即Rust标准库实现通道的方式允许多个产生值的发送端，但
只能有一个消费这些值的接收端。发送端或接收端任一被丢弃时，
意味着通道被关闭

```rust
use std::thread;
use std::sync:mpsc;
use std::time:Duration;

fn main(){
	let (tx, rx) = mpsc::channel();
		//`tx`表示发送端，`rx`表示接收端

	let tx1 = mpsc::Sender::clone(&tx);
		//clone发送端创建多个生产者

	thread::spawn(move || {
		let vals = vec![
			String::from("hi"),
			String::from("from"),
			String::from("the"),
			String::from("thread"),
		];

		for val in vals {
			tx1.send(val).unwrap();
				//`send`会获取参数所有权归接收者所有，避免
				//值被其他线程丢弃、修改导致意外结果

				//`send`返回`Result<T, E>`，如果接收端被丢弃
				//将没有发送值的目标，将返回`Err<E>`
			thread::sleep(Duration::from_secs(1));
		}
	});

	thread::spawn(move || {d
		let vals = vec![
			String::from("move"),
			String::from("messages"),
			String::from("for"),
			String::from("you"),
		];
		
		for val in vals{
			tx.send(val).unwrap();
			thread::sleep(Duration::from_secs(1));
		}
	});

	let received = rx.recv().unwrap();
		//`recv`将阻塞直到接收到一个值，返回`Result<T, E>`
		//通道发送端关闭时`recv`将返回`Err<E>`表明不会有新值

	let received = rx.try_recv().unwrap();
		//`try_recv`不阻塞，立刻返回`Result<T, E>`，`Err<E>`
		//表示当前没有消息
		//可以再循环中多次调用`try_recv`，有消息进行处理，
		//否则进行其他工作直到下次调用`try_recv`

	for received in rx{
		//将接收端`rx`当作迭代器使用，返回接收到值
		println!("Got: {}", received);
	}
}
```

###	共享状态

（任何语言）通道都类似于单所有权，一旦值通过通道传送，将无法
再次使用，而共享内存类似于多所有权

####	`Mutex<T>`

互斥器mutex：mutual exclusion，任意时刻只允许一个线程访问
数据

-	线程在访问数据之前需要获取互斥器的锁lock，lock是作为
	互斥器一部分的数据结构，记录数据所有者的排他访问权
-	处理完互斥器保护的数据之后，需要解锁，这样才能允许其他
	线程获取数据

`Mutex<T>`类似于线程安全版本的`RefCell<T>`（`cell`族），
提供了内部可变性，`Mutex<T>`有可能造成死锁，如一个操作需要
两个锁，两个线程各持一个互相等待。

`Arc<T>`原子引用计数atomically reference counted，则是
线程安全版本的`Rc<T>`，而线程安全带有性能惩罚，如非必要，
使用单线程版本`Rc<T>`性能更好。

```rust
use std::sync::{Mutex, Arc};
use std::thread;

fn main(){
	let counter = Arc::new(Mutex::new(0));
		//因为需要在多个线程内引用，所以需要使用多所有权
		//数据结构，而`Rc`不是线程安全的，需要使用线程安全
		//`Arc`
	let mut handles = vec![];

	for _ in 0..10{
		let counter = Arc::clone(&counter);
		let handle = thread::spawn(move || {
			let mut num = counter.lock().unwrap();
			//`lock`返回一个`MutexGuard`类型的智能指针，
			//实现了`Deref`指向其内部数据，`Drop`当
			//`MutexGuard`离开作用域时自动释放锁

			//只有`lock`才能获取值，类型系统保证访问数据之前
			//获取锁；而锁的释放自动发生，保证锁一定会释放

			//这里发生了一次强制解引用多态，将`counter`
			//解引用为`Mutex<T>`类型
			*num += 1;
		});

		handles.push(handle);
	}

	for handle in handles{
		handle.join().unwrap();
	}
	println!("Result: {}", counter.lock.unwrap());
}
```

####	`Sync`trait、`Send`trait

Rust的并发模型大部分属于标准库的实现，但是
`std::marker::Send`和`std::marker::Sync`时内嵌于语言的

-	`Send`：表明类型所有权可以在线程间传递，几乎所有类型都是
	`Send`的，`Rc<T>`是其中的一个例外，因为`Rc<T>`clone之后
	在两个线程间可能同时更新引用计数，trait bound保证无法将
	不安全的`Rc<T>`在线程间传递。任何全部由`Send`组成的类型
	会自动标记为`Send`
-	`Sync`：表明类型可以安全的在多线程中拥有其值的引用，对于
	任何类型，如果`&T`是`Send`的，那么`T`就是`Sync`的。
	`Cell<T>`系列不是`Sync`的，`Mutex<T>`是。基本类型是`Sync`
	的，全部由`Sync`组成的类型也是`Sync`的

`Send`和`Sync`是标记trait，不需要实现任何方法，全部是`Send`
或`Sync`组成的类型就是`Send`或`Sync`，一般不需要手动实现
它们，而且手动实现这些标记trait涉及编写不安全代码

