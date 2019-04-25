#	语言设计

##	编程范型

###	*Functional Programming*

函数式编程

-	程序使用紧密的函数调用表示，这些函数可以进行必要计算，
	但是不会执行人场改变程序状态（如：赋值）的操作
-	函数就是数据值，可以像对待其他数据值一样对其进行操作

##	语言基础

###	错误类型

> - *trapped errors*：导致程序终止执行错误
> > -	除0
> > -	Java中数组越界访问
> - *untrapped errors*：出错后程序继续执行，但行为不可预知，
	可能出现任何行为
> > -	C缓冲区溢出、Jump到错误地址

###	程序行为

> - *forbidden behaviours*：语言设计时定义的一组行为，必须
	包括*untrapped errors*，*trapped errors*可选
> - *undefined behaviours*：未定义为行为，C++标准没有做出
	明确规定，由编译器自行决定
> - *well behaved*：程序执行不可能出现*forbidden behaviors*
> - *ill behaved*：否则

###	语言类型

![language_types](imgs/language_types.png)

> - *strongly typed*：强类型，偏向于不能容忍**隐式类型转换**
> - *weakly typed*：弱类型，偏向于容忍**隐式类型转换**
> - *statically typed*：静态类型，编译时就知道每个变量类型，
	因为类型错误而不能做的事情是语法错误
> - *dynamically typed*：动态类型，编译时不知道每个变量类型
	，因为类型错误而不能做的事情是运行时错误

-	静态类型语言不定需要声明变量类型
	-	*explicitly typed*：显式静态类型，类型是语言语法的
		一部分，如：C
	-	*implicitly typed*：隐式静态类型，类型由编译时推导
		，如：ML、OCaml、Haskell

-	类型绑定
	-	强类型倾向于**值类型**，即类型和值绑定
	-	弱类型倾向于**变量类型**，类型和变量绑定，因而偏向于
		容忍隐式类型转换

##	*polymorphism*

多态：能将相同代码应用到多种数据类型上方式

-	相同对象收到不同消息、不同对象收到相同消息产生不同动作

###	*Ad hoc Polymorphism*：

*ad hoc polymorphism*：接口多态，为类型定义公用接口

-	函数重载：函数可以接受多种不同类型参数，根据参数类型有
	不同的行为

> - *ad hoc*：for this, 表示专为某特定问题、任务设计的解决
	方案，不考虑泛用、适配其他问题

###	*Parametric Polymorphism*

*parametric polymorphism*：参数化多态，使用抽象符号代替具体
类型名

-	定义数据类型范型、函数范型

-	参数化多态能够让语言具有更强表达能力的同时，保证类型安全
-	例
	-	C++：函数、类模板
	-	Rust：trait bound

> - 在函数式语言中广泛使用，被简称为*polymorphism*

###	*Subtyping*

*subtyping/inclsion polymorphism*：子类多态，使用基类实例
表示派生类

-	子类多态可以用于限制多态适用范围

-	子类多态一般是**动态解析**的，即函数地址绑定时间
	-	非多态：编译期间绑定
	-	多态：运行时绑定

-	例
	-	C++：父类指针
	-	Rust：trait bound



