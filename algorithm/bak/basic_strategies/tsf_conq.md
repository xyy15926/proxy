#	变治法

##	简介

变治法分成两个阶段工作

-	“变”：出于某种原因，把问题实例变得容易求解
-	“治”：对实例问题进行求解

根据对问题的变换方式，可以分为3类

-	instance simplification：实例化简，变换为同样问题的
	更简单、更方便的实例
-	representation change：改变表现，变换为同样实例不同表现
-	problem reduction：问题化简，变换为算法已知的另一个问题
	的实例


##	预排序

如果列表是有序的，许多关于列表的问题会更容易求解

###	检验数组唯一性

####	算法

```c
PresortElementUniqueness(A[0..n-1])
	// 先对数组排序，求解元素唯一性问题
	// 输入：n个可排序元素构成数[0..n-1]
	// 输出：A中有相等元素，返回true，否则false
	对数组排序
	for i=0 to n-2 do
		if A[i] = A[i+1]
			return false
	return true
```

####	特点

-	而蛮力法的最差效率$\in \Theta(n^2)$

-	算法效率
	-	排序至少需要$nlogn$次比较，判断是否重复至多n-1次，
		则排序部分算法决定了算法总效率
	-	$T(n)=T_{sort}(n)+T_{scan}(n) \in \Theta(nlogn)$

###	众数计算

寻找给定数字列表中最经常出现的数值

####	算法

```c
PresortMode(A[0..n-1])
	// 对数组预排序来计算其模式（众数）
	// 输入：可排序数组A[0..n-1]
	// 输出：数组模式
	对数组A排序
	i = 0
	modefrequency = 0
	while i <= n-1 do
		runlength = 1
		runvalue = A[i]
		while i + runlength <= n-1 and A[i+runlength] == runvalue
			// 相等数值邻接，只需要求出邻接次数最大即可
			runlength = runlength+1
		if runlength > modefrequency
			modefrequency = runlength
			modevalue = runvalue
		i = i+runlength

	return modevalue
```

####	特点

-	蛮力法：扫描列表，使用辅助列表存储已经遇到的值、出现的
	频率，比较所有不同值出现频率
	-	当所有元素均不相同时蛮力法效率最差，比较次数为
		$\frac {n(n-1)} 2$

-	算法效率：算法运行时间同样取决于排序时间

###	查找问题

####	算法

```c
PreorderSearch(A[0..n-1])
	// 对数组预排序然后查找
	// 输入：可排序数组A[0..n-1]
	// 输出：元素在数组中的位置
	对B[(数组元素, 索引)]进行预排序
	使用折半查找寻找二元组
	返回二元组中索引
```

####	特点

-	蛮力法：最差情况下需要比较n次
-	预排序：算法运行效率同样取决于排序算法
	-	查找算法在最差情况下总运行时间$\in \Theta(nlogn)$
	-	如果需要在统一列表上进行多次查找，预排序才值得

##	高斯消元法

假设方程组系数矩阵为n阶方阵，且解唯一

###	前向消去法

####	算法

```c
ForwardElimination(A[1..n, 1..n], b[1..n])
	// 对方程组扩展矩阵[A|b]使用高斯消元法
	// 输入：矩阵A[1..n, 1..n]，向量b[1..n]
	// 输出：扩展的上三角矩阵
	for i = 1 to n do
		A[i, n+1] = b[i]
		// 得到扩展矩阵
	for i = 1 to n-1 do
		for j = i+1 to n do
			for k = n+1 downto i do
				A[j, k] = A[j, k] - A[i, k]*A[j, i] / A[i, i]
```

####	算法特点

-	前向消去法不一定正确
	-	如果A[i, i]==0，不能以其作为除数，此时需要交换行
		（解唯一时总是存在非0行）
	-	A[i, i]非常小，导致比例因子A[j, i] / A[i, i]非常大，
		产生大的舍入误差

-	最内层循环效率低

###	部分选主元法

####	算法

```c
BetterForwardElimination(A[1..n, 1..n], b[1..n])
	// 用部分选主元法实现高斯消去
	// 输入：矩阵A[1..n, 1..n]，向量b[1..n]
	// 输出：扩展的上三角矩阵
	for i = 1 to n do
		A[i, n+1] = b[i]
	for i = 1 to n-1 do
		pivotrow = i
		for j = i+1 to n do
			if |A[j, i]| > A[pivot, i]
				pivotrow = j
				// 选择第i列系数最大的行作为第i次迭代基点
				// 保证比例因子绝对值不会大于1
			for k = i to n+1 do
				swap(A[i, k], A[pivot, k])
			for j = j+1 to n do
				temp = A[j, i] / A[i, i]
				// 这样只需要计算依次比例因子
				for k = i to n+1 do
					A[j, k] = A[j, k] - A[i, k] * temp
```

####	特点

-	部分选主元法克服了前向消去法弊端
	-	最内层乘法（加法）执行次数为
		$\frac {n(n-1)(2n+5) 6 \approx \frac n^3 3 \in \Theta(n^3)$
	-	始终能保证比例因子绝对值不大于1

###	反向替换法

在得到上三角系数矩阵中

-	从最后一个方程中可以立刻求出$x_n$
-	将$x_n$带入倒数第二个方程求出$x_{n-1}$
-	逐次递推得到所以解

####	特点

-	算法效率$\in \Theta(n^2)$

###	高斯消去应用

-	矩阵（可逆矩阵）中应用
	-	LU分解（Doolittle分解）
	-	Cholesky分解（要求矩阵正定）
	-	求逆
	-	求行列式
-	高斯消元法整个算法效率取决于消去部分，是立方级
	-	事实上此方法在计算机上求解大规模方程组很难，因为舍入
		误差在计算过程中会不断累积

##	数值计算

###	霍纳法则

霍纳法则：改变表现技术的例子

####	算法

-	不断将x作为公因子提取出来，合并降次后的项

```c
Horner(P[0..n], x)
	// 用霍纳法则求多项式在给定点的值
	// 输入：多项式系数数组P[0..n]、数字x
	// 输出：多项式在x点的值
	p = P[n]
	for i = n-1 downto 0 do
		p = x*p + P[i]
	return p
```

####	特点

-	算法效率
	-	效率始终为n，只相当于直接计算中$a_n x^n$的乘法数量

###	二进制（计算）幂

将幂次转换为二进制位串，利用二进制位串简化计算

####	从左至右二进制幂

-	对位串应用霍纳法则

```c
LeftRightBinaryExponentiation(a, B[0..n-1])
	// 从左至右二进制幂算法计算a^n
	// 输入：数字a、表示幂次的二级制位串B[0..n-1]
	// 输出：a^n的值
	product = a
	for i = n-1 downto 0 do
		product = product * product
		if B[i] == 1:
			prduct = product * a
	return product
```

####	从右至左二进制幂

-	累次计算二进制位串中为1部分值，将其累乘

```c
RightLeftBinaryExponentiation(a, B[0..n-1])
	// 从右至左二进制幂算法
	// 输入：数字a、表示幂次的二级制位串B[0..n-1]
	// 输出：a^n的值
	term = a
	if B[0] == 1
		product = a
	else
		product = 1
		// 保存累乘值
	for i = i to n do
		term *= 2
		// 保存二进制位为1部分值
		if B[i] = 1
			product = product * term
	return product
```

####	特点

-	算法效率
	-	两个算法效率取决于位串长度，是对数级的

###	欧几里得算法

计算最大公约数、最大公倍数

####	最大公约数

$$gcd(m, n) = gcd(n, m mod n)$$

-	n为0，返回m作为结果结束
-	将m处以n的余数赋给r
-	将n付给m，r赋给n，返回第一步

```
Euclid(m, n)
	while n != 0 do
		r = m mod n
		m = n
		n = r
	return m
```

####	最大公倍数

$$lcm(m, n) = \frac {m * n} {gcd(m, n)}$$

-	利用最大公约数计算最小公倍数

###	极值问题

-	最大值、最小值相互转换
-	求极值转换为求导数为0的临界点

###	线性规划

###	整数规划

###	化简为图

	把问题简化为标准图问题













