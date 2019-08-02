---
title: 代数运算
tags:
  - 代数
  - 抽象代数
categories:
  - 代数
  - 抽象代数
date: 2019-07-21 00:46:35
updated: 2019-07-21 00:46:35
toc: true
mathjax: true
comments: true
description: 代数运算
---

##	运算

> - *partial order*：偏序，集合X上自反的、反对称的、传递的
	关系
> - 全序：若R是集合上的偏序关系，若对每个$x, y \in X$，必有
	$xRy$或$yRx$，则称R是集合X上全序关系

-	偏序指集合中只有部分成员之间可比较
-	全序指集合全体成员之间均可比较

###	线性同余法

线性同余法：产生伪随机数最常用方法

$$\left \{ \begin{array}{l}
a_0 = & d \\
a_n = & (ba_{n-1} + c) % m, & n=1,2,\cdots
\end{array} \right.$$

> - $d \leq m$：随机序列种子
> - $b \geq 0, c \geq 0, m \geq 0$：关系到产生随机序列的随机
	性能
> > -	$m$：应该取得充分大
> > -	$gcd(m ,b)=1$：可以取b为素数

##	常用定理

###	Lucas定理

$$
C(n, m) \% p = (C(n//p, m//p) * C(n\%p, m\%p)) \% p
$$

> - $p < 10^5$：必须为素数





