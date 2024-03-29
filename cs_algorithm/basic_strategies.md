---
title: 算法设计策略
categories:
  - Algorithm
tags:
  - Algorithm
date: 2019-05-30 00:38:54
updated: 2019-05-30 00:38:54
toc: true
mathjax: true
comments: true
description: 算法设计策略
---

##	蛮力法

Brute Force：简单直接解决问题的方法，常常直接基于问题的描述
和所涉及的概念定义

###	特点

-	蛮力法可以解决各种问题，实际上可能是唯一几乎可以解决所有
	问题的方法
-	对某些重要问题，蛮力法可以产生合理算法，具备实用价值，
	且不必限制实例规模
-	如果解决问题实例不多，蛮力法速度可接受，设计高效算法可能
	不值得
-	蛮力法可以用于研究、教学目的，如：衡量同样问题其他更高效
	算法

###	案例

因为基本所有问题都可以使用蛮力得到**理论可行**的解决方法，
所以这里只包含实际可行、有价值算法

-	排序
	-	选择排序
	-	冒泡排序
-	查找
	-	顺序查找
	-	蛮力字符串匹配
-	几何
	-	最近对问题
	-	凸包问题
-	组合
	-	背包问题
	-	旅行商问题
	-	分配问题
-	图处理
	-	深度优先搜索
	-	广度优先搜索

##	*Recursion*

> - *reduction*：简化论，仅仅通过理解构成对象某一部分就可以
	理解整个对象
> - *holism*：整体论，总体总比构成它的部分更为重要

递归：将大问题通过简化成相同形式的小问题来解决问题

-	递归的思考需要从整体角度考虑
-	需要习惯于采用递归的稳步跳跃理念
-	关键在于如何正确的将原始问题分解为有效的子问题
	-	更加简单，且和原问题求解形式相同
	-	最终能达到简单情况，并正确解决
	-	能重新组合子问题的解得到原始问题的解

> - *recursive leap of faith*：递归的稳步跳跃理念，任何更
	简单的递归调用将正确工作

###	*Recursive Paradigm*

递归范型：递归函数体形式

```cpp
if(test for simple case){
	// 非递归解决简单问题
} else {
	// 将问题简化为某种形式的子问题（一个或多个）
	// 递归调用函数解决每个子问题
	// 将子问题的解组合得到原问题的解
}
```

###	返回值问题

-	无返回值：没有返回值回退过程、处理函数
	-	**全局变量、引用参数记录结果**
	-	**参数传递结果，最末栈直接使用**

-	有返回值：有返回值回退过程
	-	可以完全类似无返回值函数调用，此时返回值无实际价值
	-	也可以最终结果在初始主调函数中得到

> - 不方便使用全局变量记录结果时，可以给**真实递归调用函数**
	添加记录结果的引用参数，外层包装其、提供实参
> - 是否有返回值不是递归调用的关键，关键是**是否继续调用**

##	减治法

*Decrease and Conquer*：利用问题给定实的解和同样问题较小
实例解之间某种关系，可以自底向上、自底向上的运用该关系

-	自顶向下会自然导致递归算法，但是还是非递归实现较好
-	自底向上往往是迭代实现的，从求解问题较小实例开始

减治法有3种主要变化形式

###	*Decrease-by-Constant*

减常量：每次算法迭代总是从实例中减去相同常量

-	*新问题规模 = 原问题规模 - constant*
-	一般来说这个常量为1

###	*Decrease-by-A-Contant-Factor*

减去常量因子：在算法迭代过程中总是减去相同的常数因子

-	*新问题规模 = 原问题规模 / constant-factor*
-	常数因子一般为2

###	*Variable-Size-Decrease*

减可变规模：算法每次迭代时，规模减小模式不同

###	案例

-	减常量法
	-	数值计算
		-	自顶向下递归计算指数
		-	利用指数定义自底向上计算指数
	-	排序
		-	插入排序
		-	希尔排序
	-	图问题
		-	拓扑排序
	-	组合
		-	生成排列
		-	生成子集

-	减常因子法
	-	数值计算
		-	递归的计算$a^{n/2}$计算指数
		-	俄式乘法
		-	约瑟夫斯问题
	-	查找
		-	数值问题

-	减可变规模
	-	数值计算
		-	计算最大公约数的欧几里得算法
			$$gcd(m, n) = gcd(n, m mod n)$$
	-	排序
		-	顺序统计量
	-	查找
		-	差值查找
		-	二叉查找树
	-	组合
		-	拈游戏

##	分治法

*Divide-and-Conquer*

-	将问题划分为同一类型的若干子问题，子问题规模最好相同
-	对子问题求解
	-	一般使用递归方法
	-	规模足够小时，有时也会利用其他算法
-	有必要则合并子问题的解得到原始问题答案

###	案例

-	查找
	-	求解二叉树高度
	-	遍历二叉树
-	数值计算
	-	大整数乘法
	-	Strassen矩阵乘法
-	几何
	-	最近对问题
	-	凸包问题

##	变治法

> - 输入增强
> - 时空权衡

变治法分成两个阶段工作

-	“变”：出于某种原因，把问题实例变得容易求解
-	“治”：对实例问题进行求解

根据对问题的变换方式，可以分为3类

###	*Instance Simplification*

实例化简：变换为同样问题的更简单、更方便的实例

-	预排序：

###	*Representation Change*

改变表现：变换为同样实例不同表现

###	*problem reduction*

问题化简：变换为算法已知的另一个问题的实例

###	典例

-	排序
	-	预排序（线性表）
		-	比较计数排序
		-	分布计数排序

-	查找
	-	预排序
		-	检验线性表唯一性
		-	寻找线性表众数
		-	查找线性表中元素
	-	字符串模式增强
		-	*Horspool*算法
		-	*Boyer-Moore*算法
		-	*KMP*算法
		-	最长公共子串

-	数值
	-	高斯消元法
		-	前向消去法
		-	部分选主元法
		-	反向替换法
	-	数值计算
		-	霍纳法则
		-	二进制（计算）幂
		-	欧几里得算法
	-	极大、极小值转换
	-	极值转换为求导数为0点
	-	线性规划：在极点求解
	-	整数规划

-	图
	-	把问题转换为状态图求解

-	Hash（散列）

##	动态规划

*Dynamic Programming*：记录、再利用子问题结果

-	记录子问题解，试图避免不必要、重复子问题求解
	-	否则就是普通递归，不能避免重复求解
	-	或者说动态规划就是**普通递归+子问题解记录**

-	适合解决的问题特点

	-	**离散**最优化问题：如递推关系中包含*max*、*min*、
		*sum*等，递推式中
		-	**因变量**待求解最优化问题
		-	**自变量**则为问题中涉及的离散变量

	-	**交叠**子问题构成复杂问题，需遍历比较

	-	问题具有**最优子结构**，由*最优解定理*，最优解由
		子问题最优解构成

###	求解空间

求解空间：问题涉及的两个、多个取离散值变量，根据递推关系考虑
**离散变量待求解问题可能组合**

-	离散变量包括
	-	明显离散列表
	-	**因变量限制条件**

-	可利用变量间限制条件组合因变量
	-	默认各变量组合是笛卡尔积
	-	由于限制条件可以减少变量组合取值数量

-	某变量可能为其他变量提供搜索空间
	-	即其**每个取值均需**和其他变量组合、搜索
	-	此时可以不将其计入求解变量、动态规划表，而是遍历其
		（如：找零问题中零钱）

###	递推式、递推关系

递推式、递推关系：将原问题分解为较小、交叠子问题的方法


-	动态规划递推中包含重要思想**有序组合**

	-	有序**剔除遍历**相关变量，一般单向剔除，有些变量需要
		考虑双向，视为两个有约束条件的独立变量
		（如：最优二叉树）

	-	因为只需要求解全集最优解，所以只需要考虑
		**部分有序子集**，直观类比**矩阵可逆**只需要判断
		**顺序主子式**

-	原问题解可能无法直接得到递推关系

	-	原始问题非求数值解
		-	应该寻找数值中间解构建递推关系，
		-	再利用动态规划表得到最终解（最长子序列）

	-	原始问题是求**宽松范围解即解不要求包含断点**的解
		-	以**各个元素分别作为端点**的解构建递推关系
		-	以最优者即为宽松范围解（最长子序列）

-	递推式中应优先考虑限制条件：减少搜索空间
	-	单个自变量限制条件
	-	**两端都需要变化的变量视为两个独立、相互约束变量**

###	动态规划表

动态规划表：存储已求解交叠子问题解，相同问题查表避免重复求解

-	动态规划表结构
	-	n维结构化数据表（常用）
		-	n为自变量数量（组合变量视为单个变量）
		-	每维对应某离散变量
	-	**字典**：适合自顶向下动态规划

-	求解问题时
	-	将问题分解为交叠子问题
	-	先查动态规划表，尝试从表中得到问题解，否则求解问题，
		记录于动态规划表
	-	并用于求解更大规模问题，直至原问题求解完成

###	分类

####	自底向上（经典）

自底向上：求解给定问题所有较小子问题，最终得到的原始问题解

-	计算用**所有**小问题解、填充动态规划表格
	-	常逐行、逐列填充动态表（求解子问题）
	-	一般**先填充初始化变量对应维度（即位于内部循环）**
		-	先初始化行，在初始化列过程中可以在循环中填充行
		-	先初始化列，在初始化行过程中可以在循环中填充列
	-	**循环**保证所需子问题已经求解完毕

-	特点
	-	自底向上没有对问题整体的全局把握，必须求解全部子问题
	-	无需递归栈空间

####	自顶向下

自顶向下：整体把握原始问题，只计算对求解原问题的有必要子问题

-	用自顶向下方式求解给定问题
	-	先将问题分解子问题
	-	求解必要的子问题
		-	先检查相应动态规划表中该问题是否已求解，
		-	否则求解子问题，记录解于动态规划表
	-	所有子问题求解完毕，回溯得到原问题解

-	特点
	-	整体把握问题整体，避免求解不必要子问题
	-	需要通过回溯（递归栈）得到问题最优解的构成

###	*Principle of Optimality*

最优化法则：最优化问题的任一实例的最优解，都是由其子问题实例
的最优解构成

-	最优化法则在大多数情况下成立，但也存在少数例外：寻找图
	中最长简单路径

-	在动态规划算法中，可以方便检查最优化法则是否适用

> - 动态规划的大多数应用都是求解最优化问题

###	典例

-	组合问题
	-	币值最大化问题
	-	找零问题
	-	硬币问题
	-	背包问题
	-	最优二叉查找树
	-	最长公共子序列
-	查找问题
	-	字符串
		-	最长上升子序列
		-	最长公共字串
		-	编辑距离

##	贪婪技术

贪婪法：通过一系列步骤构造问题的解，每步对目前构造的部分解
作扩展，直到得到问题的完整解

###	特点

-	只能应用于最优问题，但可以作为一种通用设计技术

-	贪婪法每步条件

	-	*feasible*：必须可行，满足问题约束
	-	*locally optimal*：是当前所有步骤中所有可行选择的
		最佳局部选择
	-	*irrevocable*：选择一旦做出不能更改

-	希望通过通过一系列局部最优选择产生全局最优解

	-	有些问题能够通过贪婪算法获得最优解
	-	对于无法通过贪婪算法获得最优解的问题，如果满足于、
		关心近似解，那么贪婪算法依然有价值

###	正确性证明

证明贪婪算法能够获得全局最优解方法

-	数学归纳法

-	证明在接近目标过程中，贪婪算法每步至少不比其他任何算法差

-	基于算法的输出、而不是算法操作证明贪婪算法能够获得最优解

###	拟阵

###	典例

-	图
	-	Prim算法
	-	Kruskal算法
	-	Dijkstra算法

-	组合
	-	哈夫曼树（编码）

##	回溯法

*Backtracing*：每次只构造解的一个满足约束**分量**，然后评估
此**部分构造解**

-	尝试对部分构造解进行**进一步**构造（构造下个分量），若
	存在不违反问题约束的下个分量，则接受**首个**合法选择

-	若无法得到下个分量合法选择，则不必考虑之后分量，此时进行
	回溯，将部分构造解最后一个分量替换为下个选择

> - 回溯法核心就是对状态空间树进行剪枝，忽略无法产生解的分支

###	适合问题

-	适合处理含有约束条件、困难的组合问题

	-	往往只需要求出*feasible solution*
	-	问题往往有精确解，但是没有高效算法求解

-	回溯法目标是最终输出：n元组$(x_1, x_2, \cdots, x_n)$

	-	其中元素$x_i$为有限线性集$S_i$的一个元素
	-	元组可能需要满足额外约束

###	状态空间树

-	回溯法会显式、隐式的生成一棵空间状态树

	-	树根表示查找解之前的初始状态

	-	树的第$i$层节点表示对第$i$个分量的选择

		-	应该、可以认为是经过$i$次**可能**选择后的由$i$个
			元素组成的解分量整体$(x_1, x_2, \cdots, x_i)$

	-	叶子节点

		-	在**完整树**中为**无希望解分量、完整解**之一
		-	构造中的树为**无希望分量、未处理解分量**之一

-	大部分情况下，回溯算法的状态空间树按照深度优先方式构造

	-	如果当前节点（解分量）有希望，向解分量添加
		**下个分量下个选择**得到新的节点，处理新节点

	-	如果当前节点无希望，回溯到节点父母重新处理

####	算法

```c
Backtrack(X[1..i])
	// 回溯算法通用模板
	// 输入：X[1..i]一个解的前i个有希望的分量
	// 输出；代表问题解的所有元组
	if X[1..i] 是一个解
		write X[1..i]
	else
		for 满足约束的x \in S_{i+1} do
			// 不符合约束的不处理，即回溯
			// 符合约束则继续深度优先搜索
			X[i+1] = x
			Backtrack(X[1..i+1])
```

####	Promising

> - *Promising*：有希望，当前解分量（节点）仍然有可能导致
	完整解，满足
> > -	当前解分量符合约束：新添加分量个体约束、解分量整体
		约束
> > -	当前解分量节点仍有未处理的子女节点
> - *Nonpromising*：没希望，当前解分量不满足有希望两个条件
	，无法导致完整解

注意：有希望不能采用递归定义：是否有希望是当前状态的结果，
当时并不知道、不需要知道子女状态（是否有希望）

####	约束判断位置

处理节点时，对子女的约束条件有两种说法

-	*添加下个分量满足约束的选择*：这里是将约束**说法上**提前
	考虑

	-	此说法可能适合约束只需要考虑最后分量的情况
	-	此种情况下的*有希望*只需要满足：解分量节点有未处理、
		合法子女
	-	是这里回溯法部分的说法

-	*添加下个分量下个选择*：这里是将约束**说法上**延后考虑

	-	此说法可能适合约束需要考虑解分量整体的情况
	-	此种情况下的*有希望*就是前面条件
	-	是这里状态空间树的说法

-	但其实两者是一样的，只是说法不同

	-	前一个说法绘制状态空间树也同样需要
		**绘制不满足约束的节点*

	-	后一个说法也不定会直接把元素添加解分量中

###	算法特点

-	回溯法时间效率不稳定，无法保证优良性能

	-	回溯法对状态空间树剪枝，是对穷举法的改进，避免考虑
		某些无效解

	-	回溯法在最坏情况下必须生成指数、甚至更快增长的状态
		空间中所有解

	-	但回溯法至少可以期望在期望时间能，对规模不是很小的
		问题在可接受时间内求解

	-	即使回溯法没能消去状态空间中任何元素，其依然提供了
		一种特定的解题方法，方法本身就有价值

-	回溯法状态空间树规模基本不能通过分析法求解

	-	可以通过生成一条根到叶子的随机路径，按照生成路径中
		不同选择的数量${c_i, i=1,2,\cdots,n}$信息估计规模

	-	树节点数量为:$1 + \sum_{i=1}^n \prod_{j=1}^i c_j$

	-	可以多做几次估计取平均值

-	有些技巧可以用于缩小状态空间规模

	-	组合问题往往有对称性，如：n皇后问题中第个皇后只需要
		考虑前一半位置

	-	把值预先分配给解的分量

	-	预排序

##	分支界限法

分支界限法：类似于回溯法，但是用于求*optimal solution*

-	在回溯法的基础上，比较**叶子节点边界值**、目前最优解

	-	叶子边界值：节点对应部分解向量衍生的解集合在目标函数
		值上的最优边界

	-	对最小化问题，边界值为下界；对最大化问题，边界值为
		上界

-	类似于在回溯法的约束条件中增加：节点最优边界必须超越当前
	最优值

	-	随着深度增加，节点最优边界逐渐紧密，节点更容易被终止

-	分支界限法适合问题、算法特点类似回溯法

###	状态空间树

分支界限空间树和节点生成顺序有关

-	*best-first branch-and-bound*：最佳优先分支边界策略，
	在当前树未终止**叶子**中，选择拥有最佳边界的节点作为最有
	希望节点，优先处理

	-	最优边界比较范围是**全局比较**，不仅仅局限于
	-	这种策略可能会得到较好的结果，消除更多分支，甚至有时
		只需计算一个完整解元组就能消除其他所有分支
	-	当然，最优解最终可能属于其他分支，这种策略也不一定
		能够加速算法

-	顺序策略：类似于回溯法，优先处理最近、有希望节点

###	边界函数

发现好的边界函数比较困难

-	希望函数容易计算，否则得不偿失
-	函数不能过于简单，否则无法得到紧密边界，尽可能削剪状态
	空间树分支
-	需要对具体问题各个实例进行大量实验，才能在两个矛盾的要求
	之间达到平衡

## 迭代策略

迭代策略：从某些可行解出发，通过重复应用一些简单步骤不断改进

-	这些步骤会通过一些小的、局部的改变生成新可行解
-	并使得目标函数更加优化
-	当目标函数无法再优化时，把最后可行解返回

###	问题

-	需要一个初始可行解

	-	平凡解
	-	其他算法（贪婪算法）得到近似解
	-	有些问题得到初始可行解也不简单

-	对可行解的改变需要考虑

-	局部极值问题

##	NP-Hard近似算法

NP-Hard组合优化问题即使是分支界限法也不能保证优良性能，考虑
使用近似算法快速求解

-	近似算法往往是基于特定问题的启发式算法
-	有些应用不要求最优解，较优解可能足够
-	且实际应用中常常处理不精确的数据，这种情况近似解更合适

> - *Heuristic Algorithm*：启发式算法，来自于经验而不是数学
	证明的*经验规则*

###	Perfermance Ratio

算法性能比：$R_A = \min\{c|\{r(s_a) \leqslant c\}$

-	$r(s_a) = \frac {f(s_a} {f(s^{*})}$：优化函数$f$在近似解
	$s_a$下的*accuracy ratio*

	-	这里$f$为最小化问题
	-	若$f$为最大化问题，则取倒数使得精确率总大于1
	-	比值越接近1，近似解质量越高

-	$R_A$即为：问题所有实例中，最差（大）精确率

	-	有些问题没有有限性能比的近似算法，如：旅行商问题
		（除非$P = NP$）

-	$R_A$是衡量近似算质量的主要指标

	-	需要寻找的是$R_A$接近1的算法
	-	某些简单算法性能比趋于$\infty$，这些算法也可以使用，
		只是需要注意其输出
	-	算法也被称为$R_A$近似算法

####	旅行商问题无有限近似比算法

-	若存在有限近似比算法，则
	$\exists c, f(s_a) \leqslant cf(s^{*})$

-	将哈密顿回路问题图G变换为旅行商图$G^{'}$，G中原有边距离
	为1，不存在边距离为$cn+1$

-	近似算法能在多项式时间内生成解
	$s_a, f(s_a) \leqslant cf(s^{*}) = cn$，

	-	若存在哈密顿回路，则旅行商问题中最优解$s^{*} = n$
	-	否则，旅行商问题最优解$s^{*} > cn+1$

-	则近似算法能在多项式时间解决哈密顿回路问题，而哈密顿回路
	问题为NPC问题，除非$P = NP$


###	说明

-	虽然大多数NP-Hard问题的精确求解算法，对于可在多项式时间
	相互转换问题难度级别相同，但是近似算法不是，某些问题的
	求良好近似解比其他问题简单的多

	-	因为近似算法是基于特定问题的，不具有普遍性

-	某些组合优化难题具有特殊的实例类型，这些类型在实际应用
	中比较重要，而且也容易求解



