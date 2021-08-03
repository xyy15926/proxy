---
title: Random
categories:
  - Algorithm
  - Specification
tags:
  - Algorithm
  - Specification
  - Random
date: 2021-05-13 17:13:29
updated: 2021-05-13 17:13:29
toc: true
mathjax: true
description: 
---

##	线性同余法

线性同余法：产生伪随机数最常用方法

$$\left \{ \begin{array}{l}
a_0 = & d \\
a_n = & (ba_{n-1} + c) % m, & n=1,2,\cdots
\end{array} \right.$$

> - $d \leq m$：随机序列种子
> - $b \geq 0, c \geq 0, m \geq 0$：关系到产生随机序列的随机性能
> > -	$m$：应该取得充分大
> > -	$gcd(m ,b)=1$：可以取b为素数

