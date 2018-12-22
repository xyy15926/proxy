
##	排序


##	查找

###	二叉树遍历

将空树高度定义为-1

####	二叉树高度

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

-	检查树是否为空是这个算法中最频繁的操作
	-	树的节点数为n，则根据加法操作次数满足递推式
		$A(n(T))=A(n(T_{left})) + A(n(T_{right})) + 1$，
		得到$A(n(T)) = n$
	-	考虑为树中每个节点的**空子树**添加**外部节点**得到
		扩展树，则外部节点数量x满足$x=n+1$，而检查树是否为空
		次数即为扩展树节点数目$C(n(T))=n+x=2x+1$

-	不是所有关于二叉树的算法都需要遍历两棵子树，如：查找、
	插入、删除只需要遍历两颗子树中的一棵，所以这些操作属于
	减可变规模（减治法）

##	数值计算

###	大整数乘法

####	算法

考虑a、b两个n位整数，n为偶数

-	从中间把数字分段，得到$a_1, a_0, b_1, b_0$

-	则有
	$$\begin{align}
	c & = a * b = (a_1 10^{n/2} + a_0) * (b_1 10^{n/2} + b_0) \\
	& = (a_1 * b_1)10^n + (a_1 * b_0 + a_0 * b_1) 10^{n/2} + (a_0 + b_0) \\
	& = c_2 10^n + c_1 10^{n/2} + c_0
	\end{align}$$
	-	$c_2 = a_1 * b_1$
	-	$c_0 = a_0 * b_0$
	-	$c_1 = (a_1 + a_0) * (b_1 + b_0) - (c_2 + c_0)

-	若n/2也是偶数，可以使用相同的方法计算得到三个乘法表达式
	-	若n为2的幂次，就可以得到计算n位数积的递归乘法
	-	n迭代到足够小时，递归就可以停止

####	特点

-	算法效率
	-	乘法次数递推式：$M(n)=3M(n/2), M(1)=1$，则
		$M(n) = n^(log_2 3) \approx n^{1.585}$
	-	加法次数递推式：$A(n)=3A(n/2) + cn, A(1)=1$，则
		$A(n) \in \Theta(n^{log_2 3})$

-	算法有渐进效率优势，实际性能依赖于计算机系统、算法实现
	质量，在某些情况下
	-	计算8位乘法时，分治算法速度快于传统方法
	-	计算超过100位时，速度是传统算法2倍

####	应用

-	在密码技术中，需要对超过100位十进制整数进行乘法运算，而
	计算机往往不能直接运算

###	Strassen矩阵乘法

$$\begin{align}
\begin{bmatrix}
C_{00} & C_{01} \\
C_{10} & C_{11}
\end{bmatrix}
	& =
\begin{bmatrix}
A_{00} & A_{01} \\
A_{10} & A_{11}
\end{bmatrix}
\begin{bmatrix}
B_{00} & B_{01} \\
B_{10} & B_{11}
\end{bmatrix} \\
	& =
\begin{bmatrix}
M_1+M_2-M_5+M_7 & M_3+M_5 \\
M_2+M_4 & M_1+M_3-M_2+M_6
\end{bmatrix} \\

M_1 & = (A_{00} + A_{11}) · (B_{00} + B_{11}) \\
M_2 & = (A_{10} + A_{11}) · B_{00} \\
M_3 & = A_{00} · (B_{01} - B_{11}) \\
M_4 & = A_{11} · (B_{10} - B_{00}) \\
M_5 & = (A_{00} + A_{01}) · B_{11} \\
M_6 & = (A_{10} - A_{00}) · (B_{00} + B_{01}) \\
M_7 & = (A_{01} + A_{11}) · (B_{10} + B_{11}) \\
\end{align}$$

####	算法

若A、B是两个n阶方阵（若n不是2幂次，考虑填充0）

-	将A、B、C均分块为4个n/2子矩阵
-	递归使用Strassen方程中定义的矩阵M进行计算计算C各个子阵

####	算法特点

-	对2 * 2分块计算，Strassen算法执行了7次乘法、18次加减法，
	蛮力算法需要执行8次乘法、4次加法

-	算法效率
	-	乘法次数递推式：$M(n) = 7M(n/2), M(1) = 1$，则
		$M(n) = 7^{log_2 n} = n^{log_2 7} \approx n_{2.807}$
	-	加法次数递推式：$A(n) = 7A(n/2) + 18(n/2)^2, A(1)=0$
		，则$A(n) \in \Theta(n^{log_2 7})$
	-	矩阵趋于无穷大时，算法表现出的渐进效率卓越

-	还有一些算法能运行时间$\in \Theta(n^\alpha)$，最小能达到
	2.376，但是这些算法乘法常量很大、算法复杂，没有实用价值

-	矩阵乘法效率下界为$n^2$，目前得到的最优效率和其还有很大
	距离

##	几何

###	最近对问题

-	点数量n不大3时，可以通过蛮力算法求解的
-	假设集合中每个点均不相同、点按其x坐标升序排列
-	另外使用算法得到点按照y坐标升序排列的列表Q

####	算法

-	在点集在x轴方向中位数m作垂线，将点集分成大小为
	$\lceiling n/2 \rceiling, \lfloor n/2 \rfloor$两个子集
	$P_l, P_r$，然后递归求解子问题$P_l, P_r$得到最近点问题解

-	定义$d=min{d_l, d_r}$
	-	d不一定是所有点对最小距离
	-	最小距离点对可能分别位于分界线两侧，在合并子问题的
		解时需要考虑

-	只需要考虑关于m对称的2d垂直带中的点，记S为来自Q、位于
	分隔带中的点列表
	-	S同样升序排列
	-	扫描S，遇到距离更近的点对时更新最小距离$d_{min}=d$
	-	对于S中点P，只需考虑**在其后、y坐标差小于$d_min$**
		的矩形范围内点（因为S有序，P前的点已经考虑过）
	-	该矩形范围内点数目不超过6个（包括P），所以考虑下个点
		前，至多考虑5个点

```c
EfficientClosestPair(P, Q)
	// 分治法解决最近点问题
	// 输入：P存储平面上n个点，按x轴坐标升序排列
			Q存储和P相同的n个点，按y坐标升序排列
	/// 输出：最近点直接欧几里得距离
	if n <= 3
		return 蛮力法最小距离
	else
		将P前ceiling(n/2)个点复制到P_l
		将Q相应的ceiling(n/2)点复制到Q_l
		将P余下floor(n/2)个点复制到P_r
		将Q余下floor(n/2)个点复制到Q_r

		d_l = EfficientClosestPair(P_l, Q_l)
		d_r = EfficientClosestPair(P_r, Q_r)
		d = min{d_l, d_r}

		m = P[ceiling(n/2) - 1].x
		将Q中所有|x-m|<d的点复制到数组S[0..num-1]

		dminsq = d^2
		for i=0 to num-2 do
			k = i+1
			while k <= num-1 and (S[k].y - S[i].y)^2 < dminsq
				dminsq = min((S[k].x - S[i].x)^2 + (S[k].y - S[i].y)^2, dminsq)
				k = k+1
	return sqrt(dminsq)
```

####	算法特点

-	算法效率
	-	将问题划分为规模相同子问题、合并子问题解，算法都只
		需要线性时间
	-	运行时间递推式$T(n) = 2T(n/2) + f(n)$，其中
		$f(n) \in \Theta(n)$，则$T(n) \in \Theta(nlogn)$
	-	已经证明在对算法可以执行的操作没有特殊假设情况下，
		这是可能得到的最好效率

###	凸包问题

-	集合S中点按照x坐标升序排列（x坐标相同，按y坐标升序）

####	快包算法

-	$p_1$、$p_$n显然是凸包顶点，且$\overrightarrow{p_1p_n}$
	将点集分为左右两部分$S_1$、$S_2$
	-	其上的点不是凸包顶点，之后不必考虑
	-	S的凸包也被划分为upper hull、lower hull，可以使用
		相同的方法构造

-	若$S_1$为空，则上包就是线段$p_1p_n$；否则寻找距离
	$p_1p_n$最大点$p_{max}$，若有多个，则选择使得夹角
	$\angle p_{max}p_1p_n$最大点
	-	$p_max$是上包顶点
	-	包含在$\triangle p_1p_{max}p_2$中的点不是上包顶点，
		之后不必考虑
	-	不存在同时位于$\overrightarrow{p_1p_{max}}$、
		$\overrightarrow{p_{max}p_n}$左侧的点

-	对$\overrightarrow{p_1p_{max}}$及其左侧点构成的集合
	$S_{1,1}$、$\overrightarrow{p_{max}p_n}$及其左侧的点构成
	集合$S_{1,2}$，重复以上即可继续得到上包顶点

-	类似的可以对$S_2$寻找下包顶点

>	向量左侧、距离计算参考线代

####	算法特点

快包和快排很类似

-	最差效率$\Theta(n)$，平均效率好得多
-	也一般会把问题平均的分成两个较小子问题，提高效率
-	对于均匀分布在某些凸区域（园、矩形）的点，快包平均效率
	可以达到线性

