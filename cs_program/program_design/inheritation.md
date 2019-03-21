#	继承、Mixin

##	多重继承

###	多重继承问题

-	结果复杂化：相较于单一继承明确的父类，多重继承父类和子类
	之间的关系比较复杂

-	优先顺序模糊：子类同时继承父类和“祖先类”时，继承的方法
	顺序不明确

-	功能冲突：多重继承的多个父类之间可能有冲突的方法

###	多重继承优点

#todo

###	解决方案

####	规格继承

使用`inteface`、`traits`这些结构实现**“多重继承”**
（普遍意义上的单一继承）

-	类只能**单一继承**父类
-	但是可以继承**多个**`interface`、`traits`，其中只能包含
	方法，而没有具体实现

#####	方案问题

-	即使继承`interface`也需要重新实现方法
-	只是通过规定解决多重继承的问题，但是也损失优点

#####	弥补技巧**delegate**

**delegate**（代理）

1.	对`interface itf`实现一个公用实现`class impl`

2.	其他“继承”`itf`的类`itf_cls`声明一个`itf`类型的变量
	`itf_var`

3.	将公用实现的一个实例赋值给`itf_var`

4.	这样`itf_cls`中`interface itf`方法的实现就可以直接使用
	`itf_var`调用`impl`的方法

```java
interface itf{
	pulic void itf_func()=0;
}

class itf_impl implements itf{
	public void itf_func(){
	}
}

class itf_cls implements itf, others{
	itf itf_var;

	public itf_cls(String args[]){
		itf_var = new itf_impl;
	}

	public void func(){
		itf_var.func();
	}
}
```

#####	实现继承

方法+实现的集合，普遍意义上多重继承

-	类可以继承多个父类
-	多个父类没有要求其中不能包含方法的实现

#####	方案问题

即多重继承的普遍问题

#####	弥补技巧**mixin**

**mixin**（混入）

-	每个类只“逻辑上”继承一个类，其他只继承**mixin类**
	-	mixin类单一职责
	-	mixin对宿主类（子类）无要求
	-	宿主类（子类）不会因为去掉mixin类而受到影响，不调用
		超类方法避免引起MRO查找顺序问题

-	mixin思想同规则继承
	-	均是将父类划分为
		-	**逻辑父类**
		-	**逻辑特性族**
	-	但是规则继承是**禁止**（普遍意义）多重继承，而mixin
		只是有效方案，并没有从规则上真正禁止“多重继承”
	-	mixin对开发者要求更严格，需要自查是否符合mixin原则

```python
class MixinCls:
	pass

class SubCls(SuperCls, MixinCls1, MixinCls2):
	pass
```

