---
title: 数据预处理
tags:
  - Machine Learning
  - Data Preprocessing
categories:
  - ML Technique
  - Data Preprocessing
date: 2019-07-21 00:46:35
updated: 2019-07-21 00:46:35
toc: true
mathjax: true
comments: true
description: 数据预处理
---

##	数据说明

###	数据模式

-	结构化数据：行数据，可用二维表逻辑表达数据逻辑、存储在数据库中
	-	可以看作是关系型数据库中一张表
	-	行：记录、元组，表示一个样本信息
	-	列：字段、属性，有清晰定义

-	非结构化数据：相对于结构化数据而言，不方便用二维逻辑表达的数据
	-	包含信息无法用简单数值表示
		-	没有清晰列表定义
		-	每个数据大小不相同
	-	研究方向
		-	社交网络数据
		-	文本数据
		-	图像、音视频
		-	数据流
	-	针对不同类型数据、具体研究方面有不同的具体分析方法，不存在普适、可以解决所有具体数据的方法

-	半结构化数据：介于完全结构化数据、完全无结构数据之间的数据
	-	一般是自描述的，数据结构和内容混合、没有明显区分
	-	树、图（*XML*、*HTML* 文档也可以归为半结构化数据）

> - 结构化数据：先有结构、再有数据
> - 半结构化数据：先有数据、再有结构

##	数据问题

###	稀疏特征

-	产生原因
	-	数据缺失
	-	统计数据频繁 0 值
	-	特征工程技术，如：独热编码

###	缺失值

####	产生原因

-	信息暂时无法获取、成本高
-	信息被遗漏
-	属性不存在

####	缺失值影响

-	建模将丢失大量有用信息
-	模型不确定性更加显著、蕴含规则更难把握
-	包含空值可能使得建模陷入混乱，导致不可靠输出

####	缺失利用

-	直接使用含有缺失值特征：有些方法可以完全处理、不在意缺失值
	-	分类型变量可以将缺失值、异常值单独作为特征的一种取值
	-	数值型变量也可以离散化，类似分类变量将缺失值单独分箱

-	删除含有缺失值特征
	-	一般仅在特征缺失率比较高时才考虑采用，如缺失率达到 90%、95%

####	插值补全

-	非模型补全缺失值
	-	均值、中位数、众数
	-	同类/前后均值、中位数、众数
	-	固定值
	-	矩阵补全
	-	最近邻补全：寻找与样本最接近样本相应特征补全

-	手动补全：根据对所在领域理解，手动对缺失值进行插补
	-	需要对问题领域有很高认识理解
	-	缺失较多时费时、费力

-	建模预测：回归、决策树模型预测
	-	若其他特征和缺失特征无关，预测结果无意义
	-	若预测结果相当准确，缺失属性也没有必要纳入数据集

-	多重插补：认为待插补值是随机的
	-	通常估计处待插补值
	-	再加上**不同噪声**形成多组可选插补值
	-	依据某准则，选取最合适的插补值

-	高维映射：*one-hot* 编码增加维度表示某特征缺失
	-	保留所有信息、未人为增加额外信息
	-	可能会增加数据维度、增加计算量
	-	需要样本量较大时效果才较好

-	压缩感知：利用信号本身具有的**稀疏性**，从部分观测样本中恢复原信号
	-	感知测量阶段：对原始信号进行处理以获得稀疏样本表示
		-	傅里叶变换
		-	小波变换
		-	字典学习
		-	稀疏编码
	-	重构恢复阶段：基于稀疏性从少量观测中恢复信号

###	异常值

> - 异常值/离群点：样本中数值明显偏离其余观测值的个别值

异常值分析：检验数据是否有录入错误、含有不合常理的数据

####	非模型异常值检测

-	简单统计
	-	观察数据统计型描述、散点图
	-	箱线图：利用箱线图四分位距对异常值进行检测

-	$3\sigma$ 原则：取值超过均值 3 倍标准差，可以视为异常值
	-	依据小概率事件发生可能性“不存在”
	-	数据最好近似正态分布

####	模型异常值检测

-	基于模型预测：构建概率分布模型，计算对象符合模型的概率，将低概率对象视为异常点
	-	分类模型：异常点为不属于任何类的对象
	-	回归模型：异常点为原理预测值对象
	-	特点
		-	基于统计学理论基础，有充分数据和所用的检验类型知识时，检验可能非常有效
		-	对多元数据，可用选择少，维度较高时，检测效果不好

-	基于近邻度的离群点检测：对象离群点得分由其距离 *k-NN* 的距离确定
	-	*k* 取值会影响离群点得分，取 *k-NN* 平均距离更稳健
	-	特点
		-	简单，但时间复杂度高 $\in O(m^2)$，不适合大数据集
		-	方法对参数 *k* 取值敏感
		-	使用全局阈值，无法处理具有不同密度区域的数据集

-	基于密度的离群点检测
	-	定义密度方法
		-	*k-NN* 分类：*k* 个最近邻的平均距离的倒数
		-	*DSSCAN* 聚类中密度：对象指定距离 *d* 内对象个数
	-	特点
		-	给出定量度量，即使数据具有不同区域也能很好处理
		-	时间复杂度 $\in O^(m^2)$，对低维数据使用特点数据结构可以达到 $\in O(mlogm)$
		-	参数难以确定，需要确定阈值

-	基于聚类的离群点检测：不属于任何类别簇的对象为离群点
	-	特点
		-	（接近）线性的聚类技术检测离群点高度有效
		-	簇、离群点互为补集，可以同时探测
		-	聚类算法本身对离群点敏感，类结构不一定有效，可以考虑：对象聚类、删除离群点再聚类
		-	检测出的离群点依赖类别数量、产生簇的质量

-	*One-class SVM*

-	*Isolation Forest*

####	异常值处理

-	删除样本
	-	简单易行
	-	观测值很少时，可能导致样本量不足、改变分布

-	视为缺失值处理
	-	作为缺失值不做处理
	-	利用现有变量信息，对异常值进行填补
	-	全体/同类/前后均值、中位数、众数修正
	-	将缺失值、异常值单独作为特征的一种取值

> - 很多情况下，要先分析异常值出现的可能原因，判断异常值是否为**真异常值**

###	类别不平衡问题

####	创造新样本

-	对数据集重采样
	-	尝试随机采样、非随机采样
	-	对各类别尝试不同采样比例，不必保持 1:1 违反现实情况
	-	同时使用过采样、欠采样

-	属性值随机采样
	-	从类中样本每个特征随机取值组成新样本
	-	基于经验对属性值随机采样
	-	类似朴素贝叶斯方法：假设各属性之间相互独立进行采样，但是无法保证属性之前的线性关系

-	对模型进行惩罚
	-	类似 *AdaBoosting*：对分类器小类样本数据增加权值
	-	类似 *Bayesian*分类：增加小类样本错分代价，如：*penalized-SVM*、*penalized-LDA*
	-	需要根据具体任务尝试不同惩罚矩阵

####	新角度理解问题

-	将小类样本视为异常点：问题变为异常点检测、变化趋势检测
	-	尝试不同分类算法
	-	使用 *one-class* 分类器

-	对问题进行分析，将问题划分为多个小问题
	-	大类压缩为小类
	-	使用集成模型训练多个分类器、组合

> - 需要具体问题具体分析

####	模型评价

-	尝试其他评价指标：准确度在不平衡数据中不能反映实际情况
	-	混淆矩阵
	-	精确度
	-	召回率
	-	*F1* 得分
	-	*ROC* 曲线
	-	*Kappa*

###	数据量缺少

####	图片数据扩充

*Data Agumentation*：根据先验知识，在保留特点信息的前提下，对原始数据进行适当变换以达到扩充数据集的效果

-	对原始图片做变换处理
	-	一定程度内随机旋转、平移、缩放、裁剪、填充、左右翻转，这些变换对应目标在不同角度观察效果
	-	对图像中元素添加噪声扰动：椒盐噪声、高斯白噪声
	-	颜色变换
	-	改变图像亮度、清晰度、对比度、锐度

-	先对图像进行特征提取，在特征空间进行变换，利用通用数据
	扩充、上采样方法
	-	*SMOTE*

-	*Fine-Tuning* 微调：直接接用在大数据集上预训练好的模型，在小数据集上进行微调
	-	简单的迁移学习
	-	可以快速寻外效果不错针对目标类别的新模型

##	特征缩放

> - 正则化：**针对单个样本**，将每个样本缩放到单位范数
> - 归一化：针对单个属性，需要用到所有样本在该属性上值

###	*Normalizaion*

归一化/标准化：将特征/数据缩放到指定大致相同的数值区间

-	某些算法要求数据、特征数值具有零均值、单位方差
-	消除样本数据、特征之间的量纲/数量级影响
	-	量级较大属性占主导地位
	-	降低迭代收敛速度：梯度下降时，梯度方向会偏离最小值，学习率必须非常下，否则容易引起**宽幅震荡**
	-	依赖样本距离的算法对数据量机敏感

> - 决策树模型不需要归一化，归一化不会改变信息增益（比），*Gini* 指数变化

####	*Min-Max Scaling*

线性函数归一化：对原始数据进行线性变换，映射到 $[0, 1]$ 范围

$$
X_{norm} = \frac {X - X_{min}} {X_{max} - X_{min}}
$$

> - 训练集、验证集、测试集都使用训练集归一化参数

####	*Z-Score Scaling*

零均值归一化：将原始数据映射到均值为 0，标准差为 1 的分布上

$$
Z = \frac {X - \mu} {\sigma}
$$

###	*Regularization*

正则化：将样本/特征**某个范数**缩放到单位 1

$$\begin{align*}
\overrightarrow x_i & = (
	\frac {x_i^{(1)}} {L_p(\overrightarrow x_i)},
	\frac {x_i^{(2)}} {L_p(\overrightarrow x_i)}, \cdots,
	\frac {x_i^{(d)}} {L_p(\overrightarrow x_i)})^T \\
L_p(\overrightarrow x_i) & = (|x_i^{(1)}|^p + |x_i^{(2)}|^p + 
	\cdots + |x_i^{(d)}|^p)^{1/p}
\end{align*}$$

> - $L_p$：样本的 $L_p$ 范数

-	使用内积、二次型、核方法计算样本之间相似性时，正则化很有用
