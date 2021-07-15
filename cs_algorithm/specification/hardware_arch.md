---
title: 
categories:
  - 
tags:
  - 
date: 2021-04-06 09:00:06
updated: 2021-04-06 09:00:06
toc: true
mathjax: true
description: 
---

##	无符号整形二进制

###	`1`数量

-	目标：统计 `unsinged` 二进制表示中 `1` 数量
-	思路方向
	-	移位遍历
	-	分治统计

####	仅遍历`1`

```cpp
int population(unsigned int bits){
	int result = 0;
	while (bits != 0){
		bits &= bits - 1;
		++result;
	}
	return result;
}
```

-	遍历减少：仅仅遍历无符号整形二进制表示中 `1` 次数
	-	`bits &= bits - 1` 将末尾 `1` 消除

####	分治+查表

```cpp
int * initialization(){
	int * table = new int[256];
	for (int i = 1; i < 256; i++){
		table[i] = (i & 0x1) + table[i >> 1];
	}
	return table;
}
int population(int bits, int * table){
	return table[bits & 0xff] +
		table[(bit >> 8) & 0xff] +
		table[(bit >> 16) & 0xff] +
		table[(bit >> 24) & 0xff]
}
```

-	思路
	-	建表：为 8bits 数建立 256 长度的 `1` 数量表
		-	递推建立：`f(n) = f(n >> 1) + last_bit(n)`
	-	分治：将无符号整型分4块查表、加和结果

####	分治

```cpp
int population(unsigned int bits){
	// 分组计算
	bits = (bits & 0x55555555) + ((bits >> 1) & 0x55555555);
	bits = (bits & 0x33333333) + ((bits >> 2) & 0x33333333);
	bits = (bits & 0x0f0f0f0f) + ((bits >> 4) & 0x0f0f0f0f);
	bits = (bits & 0x00ff00ff) + ((bits >> 8) & 0x00ff00ff);
	bits = (bits & 0x0000ffff) + ((bits >> 16) & 0x0000ffff);
	return bits;
}
```

![unsigned_population__division](imgs/unsigned_population_division.png)


-	分治统计：依次将相邻 2bits、 4bits 分组，计算组内 `1`数量
	-	移位、求并：将组内无效 bits 置 `0`、并对齐
	-	`+`：计算组内 `1` 数量
		-	`0x0 + 0x0 = 0x0`
		-	`0x0 + 0x1 = 0x1`
		-	`0x1 + 0x1 = 0x10`：进位，符合逻辑
-	改进方式：考虑避免不必要的 `&`
	-	根据进位特性替换 2bits 组对应语句
	-	考虑组内空位是否可容纳 `+` 后结果
		-	8bits 组：有效半组 4bits 足够存储中的最大 `1` 数量 8，但
			无法存储 16 或更大， 需要及时置空无效bit
		-	16bits 组及之后：有效半组足够存储最大 `1` 数量 32，可以
			计算完之后再取值

```cpp
int population(unsigned int bits){
	// 等于上述首行
	bits = bits - ((bits >> 1) & 0x55555555);
	bits = (bits & 0x33333333) + ((bits >> 2) & 0x33333333);
	// 4bits 足够存储组内 8bits 中 `1` 最大数量 8
	bits = (bits + (bits >> 4)) & 0x0f0f0f0f;
	// 8bits 足够存储全部 32bits 中 `1` 最大数量 32
	bits = bits + (bits >> 8);
	bits = bits + (bits >> 16);
	return bits & 0x3f
}
```

###	奇偶性

-	目标：判断 `unsigned` 二进制表示中 `1` 数量奇偶性
-	思路方向
	-	移位遍历，注意语言特性
		-	逻辑移位和算术移位
		-	求余结果
	-	分治统计

####	分治统计

```cpp
// 原始版本
unsigned char parity(unsigned int i){
	// 相邻2bit一组异或确定奇偶
	i = i ^ (i >> 1);
	// 相邻4bit一组，依赖以上结果确定奇偶
	i = i ^ (i >> 2);
	i = i ^ (i >> 4);
	i = i ^ (i >> 8);
	i = i ^ (i >> 16);
	// 奇偶性保存在最后1bit，取出
	return i & 0x1;
}
```

-	分治统计：依次将相邻 2bits、 4bits 分组，统计组内奇偶性
	-	分组方式顺序调换仅影响中间结果中存放奇偶性统计结果 bits 位置
		-	奇偶性统计结果存放在组内最后 bit
		-	其中每次分组统计事实上都是统计 2bits 的奇偶性
-	改进方式
	-	调整分组顺序，将存储奇偶性 bits 位置移至最后
	-	计算奇偶性 bits 对应奇偶性表，查表得到结果
		-	一般可以设置长度为 `0x0f` 长度的数组，其中取值为索引奇偶性
		-	`0x6996` 即为对应奇偶性表，其中各位序 bit 取值为位序值对应
			的奇偶性

```cpp
// 改进版本
unsigned char parity_k(unsigned int i){
	// 将存储奇偶性 bits 移至最后
	i = i ^ (i >> 4);
	i = i ^ (i >> 8);
	i = i ^ (i >> 16);
	// 查表得到结果
	return (0x6996 >> (i & 0x0f )) & 0x01;
}
```

###	奇偶性填充

-	目标：将 `unsigned char` 中最高位作为校验位，保证整体的二进制标表示的奇偶性
-	思路方向
	-	求`1`数量并设置标志位
	-	按位乘积后取模

####	取模

```cpp
unsigned char even(unsigned char i){
	return ((i * 0x10204081) & 0x888888ff) % 1920;
}
unsigned char odd(unsigned char i){
	return ((i * 0x00204081) | 0x3DB6DB00) % 1152;
}
```

-	各数字二进制含义（设 `i` 的二进制表示为 `abcdefg`）
	-	`0x10204081 * i` 得到 `i` 二进制重复 5 次（溢出被截断）
	-	`0x888888ff &` 抽取所需 bits `d000a000e000b000f000c000gabcdefg`
	-	`1920 = 15 * 128`：对其取模即得到`[X]abcdefg`
		（将被除数表示为 16 进制分块可证）

###	位元反序

-	目标：返回 6bits 长二进制的反序

```cpp
unsigned char revert(unsigned char i){
	return ((i * 0x00082082) & 0x01122408) % 255;
}
```

-	各数字二进制含义（设 `i` 的二进制表示为 `abcdef`）
	-	`0x00082082 * i` 得到 `i` 二进制重复 4 次
	-	`0x01122408 &` 抽取所需 bits `0000000a000e00b000f00c000000d000`
	-	对 `255` 取模即得到反序*（将被除数表示为 256 进制分块可证）

###	前导 `0`

-	目标：获取 `unsigned` 的二进制表示中前导 `0` 数量
-	思路方向
	-	移位遍历
	-	区间映射：将原始 `unsigned` 映射为较小范围的取值

####	区间映射

```cpp
unsigned char nlz(unsigned int i){
	static unsigned char table[64] = {
		32, 31, 'u', 16, 'u', 30, 3, 'u', 15, 'u', 'u', 'u', 29,
		10, 2, 'u', 'u', 'u', 12, 14, 21, 'u', 19, 'u', 'u', 28,
		'u', 25, 'u', 9, 1, 'u', 17, 'u', 4, 'u', 'u', 'u', 11,
		'u', 13, 22, 20, 'u', 26, 'u', 'u', 18, 5, 'u', 'u', 23,
		'u', 27, 'u', 6, 'u', 24, 7, 'u', 8, 'u', 0, 'u'
	}

	i = i | (i >> 1);
	i = i | (i >> 2);
	i = i | (i >> 4);
	i = i | (i >> 8);
	i = i | (i >> 16);
	i = i * 0x06eb14f9;
	return table[i >> 26];
}
```

-	区间映射
	-	移位取或：将最高位 `1` 传播至所有低位，原始值映射至 33 种取值
	-	`0x06eb14f9`：将 33 种值映射为低 6 位取值均不同值
		-	此类数的共同特点是因子均为 $2^k \pm 1$
			（此类数乘法容易通过移位操作实现）
		-	最小的此性质的数为 `0x45bced1 = 17 * 65 * 129 * 513`

##	速算

###	无符号整形除法

-	目标：大部分计算机架构中除法耗时，寻找方法将特定除数的除法转换为
	其他指令
-	思路：除数为常数时，用移位、乘法、加法替代除法运算
	-	常数除法（有符号或无符号）基本都有对应的乘法版本
	-	注意类型溢出

####	除数为3

```cpp
unsigned div3(unsigned int i){
	// 在更高级别优化过程实际就会转化为类似指令
	// 但此语句可能会导致类型溢出
	return (i * 2863311531) >> 33;
}
```

$$\begin{align*}
2863311531 &= \frac {2^{33} + 1} 3 \\
i/3 &= \lfloor i * \frac {2^{33} + 1} 3 * \frac 1 {2^{33}} \rfloor \\
&= \lfloor \frac i 3 + \frac i {3 * 2^{33}} \rfloor = \frac i 3
\end{align*}$$

###	快速求平方根倒数

```c
float i_sqrt(float a){
	union {
		int ii;
		float i;
	};
	i = a;
	float ihalf = 0.5f * i;

	// 得到较好的初始估算值
	ii = 0x5f000000 - (ii >> 1);

	// 牛顿法迭代，可以重复以下语句多次提高精度
	i = i * (1.5f - ihalf * i * i);

	return i;
}
```

-	移位获取初始值
	-	考虑（规格化）单精度浮点数 $a$ 的二进制表示

		$$\begin{align*}
		a &= 2^{E-127} * (1+F) \\
		\frac 1 {\sqrt a} &= 2^{\frac {127 - E} 2 + 127} * (1+F)^{-\frac 1 2} \\
		&= 2^{190.5 - \frac E 2} * (1+F)^{-\frac 1 2}
		\end{align*}$$

	-	则 `0x5f000000 - (ii >> 1)` 可以满足指数部分得到近似结果
		$190 - \frac E 2$
	-	其他细节使得存在比 `0x5f000000` 实践更优值，如：`0x5f375a86`
		-	规格化值的底数接近 1
		-	移位运算指数最后位部分移至尾数
		-	减法运算尾数部分向指数借位

-	牛顿法：$x__{n+1} = x__n - \frac {f(x_n)} {f^{'}(x__n)}$
	-	求 $\frac 1 {\sqrt x}$，即求 $f(x) = x^{-2} - a$ 零点
	-	则迭代为 $x__{n+1} = x__n(1.5 - 0.5 a x__n^2)$

> - <http://www.lomont.org/papers/2003/InvSqrt.pdf>
> - <https://zhuanlan.zhihu.com/p/33543750>



