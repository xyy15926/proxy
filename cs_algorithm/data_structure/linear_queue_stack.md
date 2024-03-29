---
title: Stack&Queue
categories:
  - Algorithm
  - Data Structure
tags:
  - Algorithm
  - Data Structure
  - Linear
  - Stack
  - Queue
date: 2019-03-21 17:27:37
updated: 2019-03-13 14:56:03
toc: true
mathjax: true
comments: true
description: Stack&Queue
---

-	从数据结构来看，栈和队列也是线性表，但其基本操作是线性表
	操作的子集
-	从数据类型来看，栈和队列是和线性表不同的抽象数据类型

##	*Stack*

栈：限定在**表尾/栈顶**进行插入和删除、操作受限的线性表

> - *top*：栈顶，表尾端
> - *bottom*：栈底，表头端

-	栈的修改是按照`LIFO`的方式运转，又称后进先出的线性表
	-	入栈：插入元素
	-	出栈：删除栈顶元素

-	*last in first out*/栈应用广泛，对实现递归操作必不可少

###	顺序存储结构

####	顺序栈

顺序栈：顺序映像存储栈底到栈顶的数据元素，同时附设指针指示
栈顶元素在顺序栈帧中的位置

```c
typedef struct{
	SElemType * base;
	SElemType *top;
	int stacksize;
}SqStack;
```

> - `base`永远指向栈底，`top`指向栈顶元素**下个位置**
> > -	`base==NULL`：栈不存在
> > -	`top==base`：表示空栈
> - 栈在使用过程中所需最大空间难以估计，因此一般初始化设空栈
	时不应限定栈的最大容量，应分配基本容量，逐渐增加

###	链式存储结构

##	*Queue*

队列：限定在**表尾/队尾**进行插入、在**表头/队头**进行删除的
受限线性表

> - *rear*：队尾，允许插入的一端
> - *front*：队头，允许删除的一端

-	队列是一种*FIFO*的线性表
-	队列在程序设计中经常出现
	-	操作系统作业排队
	-	图的广度优先遍历

###	链式存储结构

####	链队列

链队列：使用链表表示的队列

```c
typedef struct QNode{
	QElemType data;
	struct QNode  * next;
}QNode, *QueuePtr;
typedef struct{
	QueuePtr front;
	QueuePtr rear;
}LinkQueue;
```

> - `front`：头指针
> - `rear`：尾指针

-	为方便同样给链队列添加头结点，令头指针指向头结点，此时
	空队列判断条件为头、尾指针均指向头结点
-	链队列的操作即为单链表插入、删除操作的特殊情况的，需要
	同时修改头、尾指针

###	顺序存储结构

####	循环队列

循环队列：使用顺序结构存储队列元素，附设两个指针分别指示对头
、队尾的位置

-	为充分利用数组空间，将数组视为**环状空间**

```c
typedef struct{
	QElemType * base;
	int front;
	int rear;
}
```

> - `front`：头指针
> - `rear`：尾指针
> - 循环队列时，无法通过`rear==front`判断队列空、满，可以在
	环上、环外设置标志位判断

-	C语言中无法用动态分配的一维数组实现循环队列，必须设置
	最大队列长度，如果无法确定，应该使用链队列

##	*Deque*

双端队列：限定删除、插入操作在表两端进行的线性表

-	输出受限双端队列：一个端点允许删除、插入，另一个端点只
	允许插入
-	输入受限双端队列：一个端点允许删除、插入，另一个端点只
	允许删除
-	栈底邻接的两个栈：限定某个端点插入元素只能从该端点删除

> - 看起了灵活，但是实际应用不多




###	Priority Queue优先队列

-	用于从一个动态改变的候选集合中选择一个优先级高的元素
-	主要操作
	-	查找、删除最大元素
	-	插入元素
-	实现
	-	可以基于数组、有序数组实现
	-	基于heap的优先队列实现更好


