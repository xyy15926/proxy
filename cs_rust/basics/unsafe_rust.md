---
title: Unsafe Rust
categories:
  - Rust
tags:
  - Rust
  - Unsafe
date: 2019-03-21 17:27:37
updated: 2019-02-17 11:57:08
toc: true
mathjax: true
comments: true
description: Unsafe Rust
---

不安全的Rust存在原因

-	Rust在编译时强制执行内存安全保证，但这样的静态分析是
	保守的，有些代码编译器认为不安全，但其实合法
-	底层计算机硬件的固有的不安全性，必须进行某些不安全操作
	才能完成任务

因此需要通过`unsafe`关键字切换到**不安全的Rust**，开启存放
不安全代码的块，只能在不安全Rust中进行的操作如下

-	解引用裸指针，
-	调用不安全的函数、方法
-	访问或修改可变静态变量
-	实现不安全trait

需要注意的是，`unsafe`不会关闭借用检查器或其它Rust安全检查，
在不安全Rust中仍然会检查引用，`unsafe`关键字只告诉编译器忽略
上述4中情况的内存安全检查，此4种的内存安全由用户自己保证，
这就保证出现内存安全问题只需要检查`unsafe`块。可以将不安全
代码封装进安全的抽象并提供API，隔离不安全代码。

##	解引用裸指针（raw pointer）

-	`*const T`：`T`类型不可变裸指针
-	`*mut T`：`T`类型可变裸指针

裸指针的上下文中，裸指针意味着指针解引用后不能直接赋值，
裸指针和引用、智能指针的区别

-	允许忽略借用规则，允许同时拥有不可变和可变指针，或者
	多个相同位置（值）的可变指针
-	不保证指向有效的内存
-	允许为空
-	不能实现任何自动清理功能

```rust
let mut num = 5;
let r1 = &num as *const i32;
let r2 = &num as *mut i32;
	//`as`将不可变引用和可变引用强转为对应的裸指针类型
	//同时创建`num`的可变裸指针和不可变裸指针
	//创建裸指针是安全的
unsafe{
	println!("r1 is: {}", *r1);
	println!("r2 is: {}", *r2);
		//解引用裸指针是不安全的，需要放在`unsafe`块中
}

let address = 0x012345usize;
	//创建任意地址
let r = address  as *const i32;
	//创建指向任意内存地址的裸指针
```

##	调用不安全的函数或方法

不安全函数和方法类似常规，在开头有`unsafe`关键字标记，表示
函数含有*内存不安全*的内容，Rust不再保证此函数内存安全，需要
程序员保证。

但是包含不安全代码并不意味着整个函数都需要标记为不安全，相反
将不安全代码封装于安全函数中是隔离`unsafe`代码的方法。应该
将**不安全代码与调用有关**的函数标记为`unsafe`。

```rust
unsafe fn dangerous() {}
	//`unsafe`关键字表示此函数为不安全函数，含有内存不安全
	//内容，需要程序员自身保证其内存安全
	//但是，包含不安全代码的函数不意味着整个函数都需要标记为
	//不安全，相反的，将不安全代码封装进安全函数是常用的

	//不安全函数体也是`unsafe`块，在其中进行不安全操作时，
	//不需要包裹于`unsafe`块

unsafe{
	dangerous();
	//调用不安全函数也需要在`unsafe`块中，表示调用者确认此
	//“不安全”函数在此上下文中是*内存安全*
}
```
调用不安全的函数时也需要放在`unsafe`中，表示程序员确认此函数
在调用上下文中是内存安全的。

###	`split_at_mut`的实现

```rust
let mut v = vec![1, 2, 3, 4, 5, 6];
let r = &mut v[..];
let (a, b)  r.split_at_mut(3);
	//以index=3分隔为两个列表引用（左开右闭）

assert_eq!(a, &mut [1, 2, 3]);
assert_eq!(b, &mut [4, 5, 6]);
```

`split_at_mut`方法无法指通过安全Rust实现，一个大概的“函数”
实现可以如此

```rust
use std::slice;

fn split_at_mut(slice: &mut [i32], mid: usize) -> (&mut [i32], &mut [i32]) {
	//这里根据生命周期省略规则省略了生命周期注解

	//在所有权里就有提到，这里不也是可变引用吗，为啥这样
	//还可以通过编译，是对方法中的`self`有特殊的处理吗

	let len = slice.len();
	let ptr = slice.as_mut_ptr();
		//`as_mut_ptr`返回`*mut T`可变裸指针
	
	assert!(mid <= len);
	
	unsafe{
		(slice::from_raw_parts_mut(ptr, mid),
			//`from_raw_parts_mut`根据裸指针和长度两个参数
			//创建slice，其是不安全的，因为其参数是一个
			//裸指针，无法保证内存安全，另外长度也不总是有效
		slice::from_raw_parts_mut(ptr.offset(mid as isize), len - mid))
			//`offset`同样是不安全的，其参数地址偏移量无法
			//保证始终有效
	}
}
```

###	使用`extern`函数调用外部代码

`extern`关键字用于创建、使用外部函数接口

> -	外部函数接口FFI：foreign function interface，编程语言
	用以定义函数的方式，允许不同（外部）编程语言调用这些
	函数
> - 应用程序接口ABI：application binary interface，定义了
	如何在汇编层面调用函数


```rust
extern "C" {
	//`"C"`定义了外部函数所使用的ABI
	fn abs(input: i32) -> i32;
	//希望调用的其他语言中的（外部）函数签名
}
fn main(){
	unsafe{
		println!("absolute value of -3 according to C: {}", abs(-3));
	}
}
```

`extern`块中声明的函数总是不安全的，因为其他语言并不强制执行
Rust的内存安全规则，且Rust无法检查，因此调用时需要放在
`unsafe`块中，程序员需要确保其安全

###	通过其他语言调用Rust函数

```rust
#[no_mangle]
	//告诉Rust编译器不要mangle此函数名称
pub extern "C" fn call_from_c(){
	//此函数编译器为动态库并从C语言中链接，就可在C代码中访问
	println!("just called a Rust function from C!");
}
```
>	mangle发生于编译器将函数名修改为不同的名称，这会增加
	用于其他编译器过程中的额外信息，但是会使其名称难以阅读
	而不同的编程语言的编译器mangle函数名的方式可能不同

##	访问或修改可变静态变量

全局变量：Rust中称为静态（static）变量

```rust
static HELLO_WORLD: &str = "Hello, world!";
	//静态变量（不可变）
fn main(){
	println!("name is: {}", HELLO_WORLD);
}
```

-	名称采用`SCREAMING_SNAKE_CASE`写法，必须标注变量类型
-	只能存储`‘static`生命周期的引用，因此无需显著标注
-	不可变静态变量和常量（不可变变量）有些类似
	-	静态变量值有固定的内存地址，使用其总会访问相同地址
	-	常量则允许在任何被用到的时候复制数据

访问不可变静态变量是安全的，但访问、修改不可变静态变量都是
不安全的，因为可全局访问的可变数据难以保证不存在数据竞争，
因此在任何可能情况，优先使用智能指针，借助编译器避免数据竞争

```rust
static mut COUNTER： u32 = 0;
	//可变静态变量

fn add_to_count(inc: u32){
	unsafe {
		COUNTER += inc;
		//修改可变静态变量
	}
}

fn main(){
	add_to_count(3);

	unsafe{
		println!("COUNTER: {}", COUNTER);
		//访问可变静态变量
	}
}
```

##	实现不安全trait

存在方法中包含编译器不能验证的不变量的trait时不安全的，可以
在`trait`前增加`unsafe`将trait生命为`unsafe`，且实现trait
也需要标记为`unsafe`

```rust
unsafe trait Foo{
}
unsafe impl Foo for i32{
}
```

如为裸指针类型实现（标记）`Send`、`Sync`trait时需要标记
`unsafe`，因为Rust不能验证此类型可以安全跨线程发送或多线程
访问，需要自行检查
