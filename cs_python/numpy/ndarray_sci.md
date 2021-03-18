---
title: 
categories:
  - 
tags:
  - 
date: 2021-03-14 17:52:56
updated: 2021-03-14 17:52:56
toc: true
mathjax: true
description: 
---

##	NumPy Numeric

###	矩阵、向量乘积

|Function|Desc|
|-----|-----|
|`dot(a,b[,out])`|`a`最后轴与`b`倒数第二轴的点积，即shape满足线代要求|
|`inner(a,b[,out])`|`a`最后轴与`b`最后轴的点积|
|`vdot(a,b)`|向量点积，多维将被展平|
|`outer(a,b[,out])`|向量外积，多维将被展平|
|`matmul(x1,x2,/[,out,casting,order,...])`|矩阵乘积|
|`tensordot(a,b[,axes])`|沿指定轴计算张量积|
|`einsum(subscripts,*operands[,out,dtype,...])`|Einstein求和约定|
|`einsum_path(subscripts,*operands[,optimize])`|考虑中间数组情况下评估计算表达式最小代价|
|`linalg.matrix_power(a,n)`|方阵幂|
|`kron(a,b)`|Kronecker积（矩阵外积，分块）|
|`trace(a[,offset,axis1,axis2,dtype,out])`|迹|

-	Einstein求和约定：简化求和式中的求和符号

	```python
	a = np.arange(0,15).reshape(3,5)
	b = np.arange(1,16).reshape(3,5)
	# Transpose
	np.einsum("ij->ji", a)
	# Sum all
	np.einsum("ij->", a)
	# Sum along given axis
	np.einsum("ij->i", a)
	np.einsum("ij->j", a)
	# Multiply
	np.einsum("ij,ij->",a,b)
	# Inner product
	np.einsum("ik,jk->",a,b)
	```

-	`np.tensordot`：张量积，类似普通内积，仅有结构
	-	`axes`为整形
		-	`axes>0`：`a`末尾`axes`维度、`b`开头`axes`维度
			内积
		-	`axes=0`：Kronecker积
	-	`axes`为2-Tuple：分别指定`a`、`b`内积的轴

###	其他

|Function|Desc|
|`np.i0(X)`|第1类修改的Bessel函数，0阶|

##	`np.linalg`

-	NumPy的线代基于*BLAS*、*LAPACK*提供高效的标准底层实现
	-	依赖库可以是NumPy提供的C版本子集
	-	也可是针对特定平台优化的库（更好）
		-	*OpenBLAS*
		-	*MKL*
		-	*ATLAS*

###	`np.linalg`

|Function|Desc|
|-----|-----|
|`multi_dot(arrays)`|自动选择最快的计算顺序计算内积|
|`cholesky(a)`|cholesky分解|
|`det(a)`|行列式|
|`eig(a)`|特征值、特征向量（右乘）|
|`eigh(a[,UPLO])`|Hermitian（共轭对称）或实对称矩阵特征值、特征向量|
|`eigvals(a)`|特征值|
|`eigvalsh(a[,UPLO])`|Hermitian（共轭对称）或实对称矩阵特征值|
|`inv(a)`|矩阵逆|
|`lstsq(a,b[,rcond])`|最小二乘解|
|`norm(x[,ord,axis,keepdims])`|矩阵、向量范数|
|`pinv(a[,rcond,hermitian])`|Moore-Penrose伪逆|
|`solve(a,b)`|线程方程组求解|
|`tensorsolve(a,b[,axes])`|张量方程组求解|
|`tensorrinv(a[,ind])`|张量逆|
|`svd(a[,full_matrices,compute_uv,hermitian])`|奇异值分解|
|`qr(a[,mode])`|QR分解|
|`matrix_rank(M[,tol,hermitian])`|使用SVD方法计算矩阵秩|
|`slogdet(a)`|行列式的符号、自然对数|

-	部分线代函数支持传入高维数组、数组序列，同时计算结果
	-	对高维数组，要求数组最后2、1维度满足计算要求

##	（快速）傅里叶变换`np.fft`

###	Standard FFTs

|Function|Desc|
|-----|-----|
|`fft(a[,n,axis,norm])`|1维离散傅里叶变换|
|`fft2(a[,n,axes,norm])`|2维离散FFT|
|`fftn(a[,n,axes,norm])`|N维离散FFT|
|`ifft(a[,n,axis,norm])`|1维离散逆FFT|
|`ifft2(a[,n,axes,norm])`|2维离散逆FFT|
|`ifftn(a[,n,axes,norm])`|N维离散逆FFT|

###	Real FFTs

|Function|Desc|
|-----|-----|
|`rfft(a[,n,axis,norm])`|1维离散傅里叶变换|
|`rfft2(a[,n,axes,norm])`|2维离散FFT|
|`rfftn(a[,n,axes,norm])`|N维离散FFT|
|`irfft(a[,n,axis,norm])`|1维逆离散FFT|
|`irfft2(a[,n,axes,norm])`|2维离散逆FFT|
|`irfftn(a[,n,axes,norm])`|N维离散逆FFT|

###	Hermitian FFTs

|Function|Desc|
|-----|-----|
|`hfft(a[,n,axis,norm])`|Hermitian对称（实谱）的信号的FFT|
|`ihfft(a[,n,axis,norm])`|Hermitian对称（实谱）的信号的逆FFT|

###	其他

|Function|Desc|
|-----|-----|
|`fftfreq(n[,d])`|离散FFT样本频率|
|`rfftfreq(n[,d])`||
|`fftshift(x[,axes])`|平移0频成分到频谱中间|
|`ifftshift(x[,axes])`||

##	`np.lib.scimath`

-	`np.lib.scimath`中包含一些顶层命名空间的同名函数
	-	相较于顶层空间，其定义域被扩展，相应其值域也扩展到
		复数域

		```python
		np.emath.log(-np.e) == 1 + np.pi * 1j
		```

> - `np.emath`是`np.lib.scimath`模块的推荐别名

