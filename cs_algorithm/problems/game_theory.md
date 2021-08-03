---
title: 博弈论
categories:
  - Algorithm
  - Problem
tags:
  - Algorithm
  - Problem
  - Game Theory
date: 2019-04-05 01:01:03
updated: 2019-04-05 01:01:03
toc: true
mathjax: true
comments: true
description: 博弈论
---

##	总述

##	约瑟夫斯问题

n个人围成圈编号{0..n-1}，从1号开始每次消去第m个人直到最后一个
人，计算最后人编号$J(n)$。

###	减1法

考虑每次消去1人后剩余人的编号情况

-	还剩k人时，消去1人后，以下个人编号为0开始，将剩余人重新
	编号，得到每个人在剩k-1人时的编号

-	相较于剩k人时，剩k-1人时每个人编号都减少m，即每个人在剩
	k人时编号满足

	$$J_k = (J_{k-1} + m) \% k$$

-	考虑只剩1人时，其编号为0，则可以递推求解


####	算法

```c
Joseph_1(n, m):
	// 减1法求解Joseph问题
	// 输入：人数n、消去间隔m
	// 输出：最后留下者编号
	j_n = 0
	for k from 2 to n:
		j_n = (j_n + m) % k
```

####	特点

-	算法效率
	-	时间效率$\in O(n)$

###	减常因子

剩余人数$k >= m$时考虑所有人都报数一次后剩余人的编号变化情况

-	还剩k人时，报数一圈后消去`k//m`人，以最后被消去人的下个人
	编号为0开始，将剩余人重新编号，得到剩`k-k/m`人时的编号

-	相较于剩k人时，剩`k-k//m`人时每个人编号满足

	$$\begin{align*}
	J_k & = \left \{ \begin{array}{l}
		J_{k - d} + d * m, & J_{k - d} < n\%m \\
		J_{k - d} // (m-1) * m + (J_{k - d} - n\%m)
			\% (m - 1), & J_{k - d} > n\%m
	\end{array} \right. \\
	& = \left \{ \begin{array}{l}
		s + n, & s < 0 \\
		s + s // (m-1), & s >= 0
	\end{array} \right.
	\end{align*}$$

	> - $d = k // m$
	> - $s = J_{k - d} - n\%m$

-	$k < m$时，使用减1法计算
	-	m很大时，以$k < m$作为调用减1法很容易使得递归超出
		最大值
	-	另外$m < k <= d * m$时，每轮也不过消去$d$个人，而
		递推式复杂许多、需要递归调用
	-	所以具体代码实现中应该选择较大的$d$值，如5

####	算法

```c
Joseph_factor(n, m):
	// 减常因子法求解Joseph问题
	// 输入：人数n、消去间隔m
	// 输出：最后留下者编号
	if n < 5 * m:
		j_n = 0
		for k from 2 to n
			j_n = (j_n + m) % k
		return j_n

	s = Joseph(n-n/m, m) - k % m
	if s < 0:
		retrun s + n
	else:
		return s + s // (m-1)
	return j_n
```

####	特点

-	算法效率
	-	时间效率$\in O(log n) + m$

-	特别的，对$m=2$时
	-	$n=2k$为偶数时，$J(2k)=2J(k)-1$
	-	$n=2k+1$为奇数时，$J(2k+1)=2J(k)+1$

###	任意第k个

-	考虑报数不重置，则第k个被消去的人报数为$k * m - 1$

-	对报数为$p = k * m + a, 0 \leq a < m$的人

	-	此时已经有k个人被消去，剩余n-k个人

	-	则经过$n - k$个剩余人报数之后，该人才继续报数，则
		其下次报数为$q = p + n - k = n + k*(m-1) + a$

-	若该人报数$p$时未被消去，则$a \neq m-1$，则可以得到
	$p = (q - n) // (m-1) * m +  (q-n) \% (m-1)$

####	算法

```c
Joseph_k(n, m, k):
	// 计算Joseph问题中第k个被消去人编号
	// 输入：人数n、间隔m、被消去次序k
	// 输出：第k个被消去人编号
	j_k = k*m - 1
	while j_k >= n:
		j_k = (j_k-n) // (m-1) * m - (j_k-n)%(m-1)
	return j_k
```

####	算法特点

-	算法效率
	-	时间效率$\in O(log n)$

-	特别的，m=2时对n做一次**向左循环移位**就是最后者编号

##	双人游戏

-	双人游戏中往往涉及两个概念
	-	*state*：状态，当前游戏状态、数据
	-	*move*：走子，游戏中所有可能发生的状态改变
-	状态、走子彼此之间相互“调用”
	-	状态调用走子**转化**为下个状态
	-	走子调用状态**评价**当前状态

```c
make_move(state, move):
	switch move:
		case move_1:
			state = move_1(state)
			evaluate_state(state)
		...other cases...

evaluate_state(state):
	switch state:
		case state_1:
			make_move(state, move_1)
		...other cases...
	end game
```

###	拈游戏

同样局面，每个玩家都有同样可选走法，每种步数有限的走法都能
形成游戏的一个较小实例，最后能移动的玩家就是胜者。

-	拈游戏（单堆版）：只有一堆棋子n个，两个玩家轮流拿走最少
	1个，最多m个棋子
-	拈游戏（多堆版）：有I堆棋子，每堆棋子个数分别为
	${n_1,\cdots,n_I}$，可以从任意一堆棋子中拿走任意允许数量
	棋子，甚至拿走全部一堆

####	减可变规模算法

#####	算法

（单堆）从较小的n开始考虑胜负（标准流程）

-	n=0：下个人失败
-	1<=n<=m：下个人胜利（可以拿走全部）
-	n=m+1：下个人失败（无论拿走几个，对方符合1<=n<=m
	胜利条件）
-	数学归纳法可以证明：n=k(m+1)时为败局，其余为胜局

```c
// 两个函数轮流递归调用
find_good_move(coins):
	// 判断当前是否有成功步骤
	// 输入：棋子数目
	// 输出：成功策略或没有成功策略
	for taken=1 to limit do
		if(is_bad_position(coins-taken))
			// 对手没有成功策略
			return taken
	return NO_GOOD_MOVE

is_bad_position(coins):
	// 判断当前是否是good position
	// 输入：棋子数量
	// 输出：是否有成功策略
	if (coins == 0)
		return true
	return find_good_move(coins) == NO_GOOD_MOVE
		// 没有成功策略
```

####	特点

-	堆为2时，需要对两堆是否相同分别考虑

-	对更一般的I堆时
	-	对每堆数量的位串计算*二进制数位和*
	-	结果中包含至少一个1则对下个人为胜局，全为0则为负局
	-	则玩家下步要拿走的棋子数量要使得位串二进制数位和全0
		，则对方陷入负局

	-	#todo又是二进制？？？和约瑟夫斯问题一样了
	-	但是这里没有涉及最多能拿几个啊，不一定能够成功拿到
		使拈和全为0啊

> - 二进制数位和（拈和）：每位求和并忽略进位（奇或）


