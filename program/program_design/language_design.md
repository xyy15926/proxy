#	语言设计

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
> - *well behaved*：程序执行不可能出现*forbidden behaviors*
> - *ill behaved*：否则

###	语言类型

[lanuage_types](imgs/languages_types.png)

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

