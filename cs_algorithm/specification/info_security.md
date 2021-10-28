---
title: Infomation Security
categories:
  - Algorithm
  - Specification
tags:
  - Algorithm
  - Specification
  - Hashing
  - Digest
  - Encrypt
date: 2019-05-25 19:55:48
updated: 2021-10-22 11:21:26
toc: true
mathjax: true
comments: true
description: Infomation Security
---

##	*Hash* 摘要方法

-	*Hash* 摘要方法核心：将任意长度消息映射到 $n$ 位哈希值（摘要）映射
	-	不同算法基本结构一致，在摘要长度、循环次数之间有差异
		-	消息补位
		-	消息分块
		-	逐个处理消息块

###	*MD4*

-	附加填充比特：在末尾对消息填充，使得消息 $M$ 长度满足 $len(M) mod 512 = 448$
	-	填充最高位位为 1、其余为 0

-	分块：将填充后消息 512bits 分块为 $M_1,M_2,\cdots,M_K$

-	初始化 *MD4* 缓存值
	-	*MD4* 使用 128bits 缓存存储哈希函数中间、最终摘要
	-	将其视为 4 个 32bits 寄存器初始化为
		-	`A = 0x67452301`
		-	`B = 0xefcbab89`
		-	`C = 0x98badcfe`
		-	`D = 0x10325476`

-	使用压缩函数迭代计算 $K$ 个消息分块、上次计算结果
	-	$H_{i+1} = C(H_i, M_i)$
	-	最终 $H_K$ 即为 *MD4* 摘要值

###	*MD5*

###	*Secure Hash Algorithm*

-	*SHA-1*：
-	*SHA-2*：<https://zhuanlan.zhihu.com/p/94619052>

##	不可逆签名

###	*Digital Signature Algorithm*

-	*DSA* 算法

##	密钥交换算法

###	*Deffie-Helloman* 算法

-	*DH* 算法：允许通信双方再非安全信道中安全交换密钥
	-	密钥交换双方分别构建密钥一部分，交换各自密钥得到完整的密钥
	-	离散对数求解难度保证算法安全性

-	*DH* 算法逻辑
	-	选择两个质数 $a,b,a < b$
	-	对用户 A、B，分别
		-	私下选择 $X_a < p, X_b < p$
		-	计算 $Y_a = a^{X_a} \mod p$、$Y_b = b^{X_b} \mod p$，并（公开）传递给对方
			-	逆运算即离散对数问题，即求解 $a^{X_a} \equiv X_a \pmod p$ 的困难保证算法安全性
		-	计算密钥 $K = Y_b^X_a \mod p = Y_a^X_b \mod p$
	-	二者计算出的相同密钥 $K$ 密钥

> - *TSL*（*Transport Layer Security*）、*IKE*（*Internet Key Exchange*）均以 *DH* 算法作为密钥加密算法

##	可逆加解密

###	*RSA*

-	*RSA* 算法：非对称加密算法
	-	建立在欧拉定理上实现的数据加解密，也决定其效率低下
	-	大数因式分解的难度保证算法破解难度（即使无法证明 *RSA* 算法破解难度等价于大数因式分解）

-	*RSA* 算法逻辑
	-	考虑两个大素数 $p,q$，计算 $N=pq$
		-	$N$ 的欧拉函数为 $\phi(N)=(p-1)(q-1)$
		-	知晓 $N=pq$ 可以方便的计算 $\phi(N)$，否则（按欧拉函数积性）需对 $N$ 作因式分解求解 $\phi(N)$ 困难，保证算法安全
	-	选择 $e$ 使得 $e,\phi(N)$ 互质，求解 $d$ 满足 $ed \equiv 1 (\mod \phi(N))$
		-	$e, \phi(N)$ 互质是模反元素 $d$ 元素有解的充要条件（裴蜀定理）
		-	一般选择 $e$ 为质数，保证 $e,\phi(N)$ 互质
	-	加、解密逻辑：加密消息记为 $m$	
		-	加密信息 $m_{encrypted} = m^e (\mod N)$
		-	解密信息 $m = m_{encrypted}^d = m^{ed} = m^{ed \mod \phi(N)} = m^1 (\mod N)$（欧拉定理）

> - *RSA* for *Ron Rivest*、*Adi Shamir*、*Leonard Adleman*

