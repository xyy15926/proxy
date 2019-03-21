#	Numpy函数笔记 

##	特殊NDA

```python
NDA = np.ones(
	shape = (d0, d1, ...),
	dtype = np.float64/str/int,
	order = "C"/"F"
)

NDA = np.zeros(
	shape = (d0, d1, ...)
	dtype = np.float64/str/int,
	order = "C"/"F"
)

NDA = np.eyes(
	N(int),
	M = None/int,
	k = 0/int,
	dtype = np.float64,
	oder = "C"/"F"
)
	# 返回对角阵

	# `N`：矩阵行数
	# 'M`：矩阵列数，默认为`N`
	# `k`：对角线index，大于0为上对角线，小于0为下对角线
```

##	`random`

```python
import np.random as npr
```

###	分布

####	0~1均匀分布

```python
ND(f64) = npr.random/npr.random_sample(
	size = None(1)/int/tuple)

	# 返回0~1间的随机数

ND（f64) = npr.rand(
	[d0](1/int),
	[d1](int),
	...
)
	# 无参数返回单个f64
```

####	正态分布

```python
ND(f64) = npr.normal(
	loc = 0.0,
	scale = 1.0,
	size = None(1)/int/(int)
)
	# 返回正态分布随机数
	# `loc`：正太分布均值
	# `scale`：正太分布方差

ND(f64) = npr.randn(
	[d0](int),
	[d1](int),
	...
)
	# 标准正态分布随机数ND
```

####	二项分布

```python
ND(i64) = npr.binomial(
	n(int/float),
	p(float/[float_64]),
	size(int/(int))
}
	# 返回二项分布随机数ND
	# `n`：二项分布实验次数，给出float会被trunate
	# `p`：二项分布概率
```

###	自定范围

####	排列、组合

#####	`randint`

```python
ND(i64) = npr.randint(
	low(int),
	high = None,
	size = None(1)/int/tuple,
	dtype = "l"/int/np.int/"int"/...
)

	# 返回`low`~`high`中随机int ND（左闭右开）
	# `high`：为None时，`low`表示上限，0为下限
```

#####	`shuffle`

```python
None = npr.shuffle(
	x([]/1Darray)
)
	# shuffle `x`元素，直接修改原对象（所有参数不能为tuple）
```

#####	`choice`

```python
ND = npr.choice(
	a(int/1Darray),
	size = None(1)/int/tuple,
	replace = True,
	p = None/array-like
)

	# 从`a`从产生样本

	# `a`：为int时相当于`np.arange(a)`
	# `replace`：默认允许重复挑选
	# `p`：个元素被选择概率
```

####	`permutation`

```python
ND = npr.permutation(
	x(int/array)
)

	# 随机permute `x`

	# `x`为int时，相当于`np.arange(x)`
	# `x`为array时，只会在最高维轴shuffle
```

### 生成器

```python
RandomState = npr.RandomState(
	seed = None/int/array-like
)
	# 返回指定种子值的`mtrand.Random`对象
	# 对象可以传递种子值
RandomState.randint(9)
	# 可以直接调用random系方法

None = npr.seed(
	seed = None/int,
)
	# 设置全局种子
```

##	Matrix

```python
np.matrix
```

-	numpy中矩阵类型就是二维
	-	为了区分行向量、列向量，所以矩阵类型都是2维，无论
		如何取值，得到都是新的二维矩阵
	-	即`mtx[0]`永远都是表示矩阵第一行




