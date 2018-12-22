#	树

##	总述

###	Free tree

自由树：**连通**、**无回路图**，具有一些其他图不具有的重要
特性

-	边数总比顶点数少一：$|E|=|V|-1$
	-	这个是图为一棵树的必要条件，但不充分
	-	若图是连通的，则是充分条件

-	任意两个顶点之间总是存在简单路径

###	Forest

森林：**无回路**但不一定连通的图

-	其每个连通分量是一棵树

###	Rooted Tree

有根树：存在根节点的自由树

-	树中任何两个节点间**总存在简单路径**，所以可以任选自由树
	中某节点，作为有根树的根

-	有根树远远比自由树重要，所以也简称为树
	-	根一般放在树的顶层，第0层
	-	之后节点根据和根的距离放在相应层数

###	Ordered Tree

有序树：所有顶点的所有子女都是有序的有根树

###	应用

-	常用于描述层次关系
	-	文件目录
	-	企业的组织结构
	-	字典的实现
	-	超大型的数据集合的高效存储
	-	数据编码

-	用于分析递归算法
	-	*state-space tree*：状态空间树，强调了两种算法设计
		技术：回溯、分支界限

###	结构

-	*ancestor*：从根到该顶点上的简单路径上的所有顶点
-	*proper ancestor*：除自身外的所有祖先顶点
-	*parent*：从根到顶点简单路径中，最后一条边的另一端节点
-	*parental*：至少有一个子女的顶点
-	*child*：
-	*sibling*：具有相同父母的顶点
-	*leaf*：没有子女的顶点
-	*descendent*：所有以该顶点为祖先的顶点
-	*proper descendent*：不包括顶点自身的子孙
-	*subtree*：顶点的所有子孙、连接子孙的边构成以该顶点为根的
	子树
-	*depth*：根到该顶点的简单路径的长度
-	*height*：根到叶节点的最长简单路径的长度

##	Binary Tree

二叉树：所有顶点子女个数不超过2个，每个子女不是父母的
*left child*就是*right child*的有序树

-	二叉树的根是另一棵二叉树顶点的左（右）子女

-	左右子树也是二叉树，所以二叉树可以递归定义

-	涉及二叉树的问题可以用递归算法解决

###	表示方法

出于计算方便，常由代表树顶点的一系列节点表示

-	每个节点包含相关顶点的某些信息
-	两个分别指向左右子女的指针

####	简单表示

简单的在父节点中加入与子女相同数量的指针

-	这种表示方法在不同子女数目相差很大时不方便

####	First Child-next Silbling Representaion

先子女后兄弟表示法

-	每个节点只包含两个指针，左指针指向第一个子女，右指针指向
	节点的下一个兄弟

-	节点的所有兄弟通过节点右指针被单独的链表连接

可高效的将有序树改造成一棵二叉树，称为关联二叉树

###	Complete Binary Tree

完全二叉树：essentially complete，树的每层都是满的，除了最后
一层最右边的元素（一个或多个）可能有缺位

####	特点

-	只存在一棵n个节点完全二叉树，高度为
	$\lfloor log_2 n\r floor$

-	可以使用数组H存储完全二叉树

	-	从上到下、从左到右记录堆元素

	-	为方便，可以从1开始记录堆中的n个元素，H[0]至置空，
		或者存放限位器（值大于任何元素）

	-	父母节点键会位于数组前$\lfloor n/2 \rfloot$个位置中
		，叶子节点位于后$\floor \lceiling n/2 \rceiling$

	-	对位于父母位置i的键，其子女位于2i、2i+1，相应的对于
		子女位置i的键，父母位于$\lfloor i/2 \rfloor$

####	应用

-	堆

###	二叉树高度

将空树高度定义为-1

####	算法

```c
Height(T):
	// 递归计算二叉树的高度
	// 输入：二叉树T
	// 输出：T的高度
	if T = null_set
		return -1
	else
		return max{Height(T_left), Height(T_right)} + 1
```

####	特点

-	检查树是否为空是这个算法中最频繁的操作

-	算法时间效率

	-	树的节点数为n，则根据加法操作次数满足递推式
		$A(n(T))=A(n(T_{left})) + A(n(T_{right})) + 1$，
		得到$A(n(T)) = n$

	-	考虑为树中每个节点的**空子树**添加**外部节点**得到
		扩展树，则外部节点数量x满足$x=n+1$
		
	-	检查树是否为空次数即为扩展树节点数目
		$C(n(T))=n+x=2x+1$

###	二叉树遍历

####	Preorder Traversal

```c
Preorder(T):
	visit(T)
	if T_left not null:
		visit(T_left)
	if T_right not null:
		visit(T_right)
```

####	Inorder Traversal

```c
Inorder(T):
	if T_left not null:
		visit(T_left)
	visit(T)
	if T_right not null:
		visit(T_right)
```

####	Postorder Tranversal 

```c
Postorder(T):
	if T_left not null:
		visit(T_left)
	visit(T)
	if T_right not null:
		visit(T_right)
```

####	特点 

-	不是所有关于二叉树的算法都需要遍历两棵子树，如：查找、
	插入、删除只需要遍历两颗子树中的一棵，所以这些操作属于
	减可变规模（减治法）

