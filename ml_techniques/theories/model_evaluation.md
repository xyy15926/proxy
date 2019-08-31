---
title: 模型评估
tags:
  - 机器学习
categories:
  - 机器学习
date: 2019-08-02 23:17:39
updated: 2019-08-02 23:17:39
toc: true
mathjax: true
comments: true
description: 模型评估
---

##	评估方向

###	模型误差

给定损失函数时，基于损失函数的误差显然评估学习方法的标准

-	回归预测模型：模型误差主要使用*MSE*
-	分类预测模型：模型误差主要是分类错误率*ERR=1-ACC*

> - 模型训练时采用损失函数不一定是评估时使用的

####	*Training Error*

训练误差：模型在训练集上的误差，损失函数$L(Y, F(X)$
（随机变量）均值


$$
e_{train} = R_{emp}(\hat f) = \frac 1 N \sum_{i=1}^N
	L(y_i, \hat {f(x_i)})
$$

> - $\hat f$：学习到的模型
> - $N$：训练样本容量

-	训练时采用的损失函数和评估时一致时，训练误差等于经验风险

-	训练误差对盘对给定问题是否容易学习是有意义的，但是本质上
	不重要

-	模型训练本身就以最小化误差为标准，如：最小化MSE、最大化
	预测准确率，一般偏低，不能作为模型预测误差的估计

-	训练误差随模型复杂度增加单调下降（不考虑模型中随机因素）

####	*Test Error*

测试误差：模型在测试集上的误差，损失函数$L(Y, f(X))$
（随机变量）均值

$$
e_{test} = \frac 1 {N^{'}} \sum_{i=1}^{N^{'}}
	L(y_i,\hat {f(x_i)})
$$

> - $\hat f$：学习到的模型
> - $N$：测试样本容量

-	测试误差反映了学习方法对未知测试数据集的预测能力，是模型
	*generalization ability*的度量，可以作为模型误差估计

-	测试误差随模型复杂度增加呈U型

	-	偏差降低程度大于方差增加程度，测试误差降低
	-	偏差降低程度小于方差增加程度，测试误差增大

-	训练误差小但测试误差大表明模型过拟合，使测试误差最小的
	模型为理想模型

###	模型复杂度

> - *approximation error*：近似误差，模型偏差，代表模型对
	训练集的拟合程度
> - *estimation error*：估计误差，模型方差，代表模型对训练集
	波动的稳健性

-	模型复杂度越高

	-	低偏差：对训练集的拟合充分

	-	高方差：模型紧跟特定数据点，受其影响较大，预测结果
		不稳定

	-	远离真实关系，模型在来自同系统中其他尚未观测的数据
		集上预测误差大

-	而训练集、测试集往往不完全相同

	-	复杂度较高的模型（过拟合）在测试集上往往由于其高方差
		效果不好，而建立模型最终目的是用于预测未知数据

	-	所以要兼顾偏差和方差，通过不同建模策略，找到恰当
		模型，其复杂度不太大且误差在可接受的水平

	-	使得模型更贴近真实关系，泛化能力较好

> - 简单模型：低方差高偏差
> - 复杂模型：低偏差高方差

> - 模型复杂度衡量参*data_science/loss*

####	*Over-Fitting*

过拟合：学习时选择的所包含的模型复杂度大（参数过多），导致
模型对已知数据预测很好，对未知数据预测效果很差

-	若在假设空间中存在“真模型”，则选择的模型应该逼近真模型
	（参数个数相近）

-	一味追求对训练集的预测能力，复杂度往往会比“真模型”更高

####	解决方法

-	减少预测变量数量

	-	最优子集回归：选择合适评价函数（带罚）选择最优模型

	-	验证集挑选模型：将训练集使用*抽样技术*分出部分作为
		*validation set*，使用额外验证集挑选使得损失最小的
		模型

		> - 抽样技术参见*sampling*

	-	正则化（罚、结构化风险最小策略）

		-	岭回归：平方损失，$L_2$范数
		-	LASSO：绝对值损失，$L_1$范数
		-	Elastic Net

-	减弱变量特化程度：仅适合迭代求参数的方法

	-	*EarlyStop*：提前终止模型训练
	-	*Dropout*：每次训练部分神经元

###	模型信息来源

-	训练数据包含信息
-	模型形成过程中提供的先验信息
	-	模型：采用特定内在结构（如深度学习不同网络结构）、
		条件假设、其他约束条件（正则项）
	-	数据：调整、变换、扩展训练数据，让其展现更多、更有用
		的信息

##	*Classification*/*Tagging*

-	分类问题：输出变量$Y$为有限个离散变量
-	标注问题：输入$X^{(1)}, X^{(2)}, \cdots, X^{(n)}$、输出
	$Y^{(1)}, Y^{(2)}, \cdots, Y^{(n)}$**均为变量序列**

> - 经验损失、结构损失总是能用作评价模型，但是意义不明确

###	混淆矩阵

-	对比实际类别值、预测类别值，编制混淆矩阵
-	基于混淆矩阵，计算各类错判率、总错判率（总错判率会
	受到数据不平衡性的影响）

|真实情况\预测结果|正例|反例|
|------|------|------|
|正例|*TP*（真正例）|*FN*（假反例）|
|反例|*FP*（假正例）|*TN*（真反例）|

![confusion_matrix](imgs/confusion_matrix.png)

###	*F-Measure*

F-测度：准率率和召回率综合值，越大越好


$$
F-measure = \frac {(\beta^2 + 1) * P * R} {\beta^2 * P + R}
$$

> - $P = \frac {TP} {TP+FP}$：查准率、精确率
> - $R = \frac {TP} {TP+FN}$：查全率、召回率、覆盖率

####	F1值

F1值：$\beta=1$时的F测度

$$
\frac {1} {F_{1}} = \frac {1} {2}
	\left( \frac {1} {P} + \frac {1} {R} \right) \\
\Rightarrow F_{1} = \frac {2 * P * R} {P + R}
	= \frac {2 * TP} {样例总数 + TP - TN}
$$

###	Recevier Operating Characteristic Curve

ROC曲线：不同**正样本概率**划分阈值下TPR、FPR绘制的折线/曲线

$$
TPR = \frac {TP} {TP+FN} \\
FPR = \frac {FP} {FP+TN}
$$

-	TPR越高越好，FPR越低越好，但是这两个指标相互制约，两者
	同时增加、减小
	-	模型倾向于将样本**判定为**为正例，则TP、FP同时增加、
		TPR、FPR同时变大
	-	即模型取不同阈值，会产生正相关的TPR、FPR的点列

-	ROC曲线即以FPR为横坐标、TPR为正坐标绘制曲线
	-	FPR接近1时，TPR也接近1，这是不可避免的
	-	而FPR接近0时，TPR越大越好
	-	所以模型ROC曲线下方面积越大，模型判断正确效果越好

-	理解
	-	将正负样本的正样本概率值分别绘制在`x=1`、`x=-1`两条
		直线上
	-	阈值即为`y=threshold`直线
	-	TPR、FPR则为`x=1`、`x=-1`两条直线在阈值直线上方点
		数量，与各直线上所有点数量比值

###	*Area Under Curve*

AUC值：ROC曲线下方面积，越大越好

-	AUC值实际上为：随机抽取一对正、负样本，模型对其中正样本
	的正样本预测概率值、大于负样本的正样本预测概率值的概率

	-	*=1*：完美预测，存在一个阈值可以让模型TPR为1，FPR为0
	-	*0.5~1*：优于随机预测，至少存在某个阈值，模型TPR>FPR
	-	*=0.5*：同随机预测，无价值
	-	*0~0.5*：差于随机预测，但是可以反向取预测值

####	AUC计算

-	绘制ROC曲线，计算曲线下面积
	-	给定一系列阈值（一般为样本数量），分别计算TPR、FPR
	-	根据TPR、FPR计算AUC

-	正负样本分别配对，计算正样本预测概率大于负样本比例

	$$\begin{align*}
	auc & = \frac {\sum I(P_P > P_N)} {M * N} \\
	I(P_P, P_N) & = \left \{ \begin{array}{l}
		1, & P_P > P_N, \\
		0.5, & P_P = P_N, \\
		0, & P_P < P_N
	\end{array} \right.
	\end{align*}$$

	> - $M, N$：正、负样本数量

-	Mann-Witney U检验（即分别配对简化）

	$$
	auc = \frac {\sum_{i \in Pos} rank(i) - 
		\frac {M * (M+1)} 2} {M * N}
	$$

	> - $Pos$：正样本集合
	> - $rank(i)$：样本$i$的按正样本概率排序的秩
		（对正样本概率值相同样本，应将秩加和求平均保证
		其秩相等）

####	加权AUC

WAUC：给**每个样本**赋权，计算统计量时考虑样本权重

-	FPR、TPR绘图

	$$\begin{align*}
	WTPR & = \frac {\sum_{i \in Pos} w_i I(\hat y_i=1)}
		{\sum_{i \in Pos} w_i} \\
	WFPR & = \frac {\sum_{j \in Neg} w_j I(\hat y_j=1)}
		{\sum_{j \in Neg} w_j}
	\end{align*}$$

	> - $WTPR, WFPR$：加权TPR、加权FPR
	> - $\hat y_i$：样本预测类别
	> - $w_i$：样本权重

-	Mann-Witney U检验：考虑其意义，带入权重即可得

	$$\begin{align*}
	auc = \frac {\sum_{i \in Pos} w_i * rank(i) -
		\sum_{i \in Pos} w_i * rank_{pos}(i)}
		{\sum_{i \in Pos} w_i * \sum_{j \in Neg} w_j}
	\end{align*}$$

	> - $rank_{pos}(i)$：正样本内部排序，样本$i$秩
	> - $Neg$：负样本集合

####	多分类AUC

-	*micro*：所有类别统一考虑，将每个类别均视为样本标签
	-	将n个样本的m个分类器共n * m个得分展平
	-	将n个样本的m维one-hot标签展平，即其中有n个正样本、
		n * (m-1)个负样本
	-	使用以上预测得分、标签计算auc

	```python
	# one-vs-rest分类器得分
	y_score = classifer.transform(X_test)
	# 展平后计算fpr、tpr
	fpr_micro, tpr_micro, threshhold_micro = \
		skilearn.metrics.roc_curve(y_test.ravel(), y_score.ravel())
	# 利用fpr、tpr计算auc
	auc_micro = skilearn.metrics.auc(fpr_micro, tpr_micro)

	# 等价于直接调用
	auc_micro = skilearn.metrics.roc_auc_score(y_test, y_score,
												average="micro")
	```

-	*macro*：对各类别，分别以计算roc曲线（即fpr、tpr），计算
	平均roc曲线得到auc
	-	对各类别分别计算fpr、tpr，共m组fpr、tpr
	-	平均合并fpr、tpr，计算auc
		-	方法1：合并fpr、去除重复值，使用m组fpr、tpr分别
			求合并后fpr插值

			```python
			# 分别计算各类别fpr、tpr
			fprs, tprs = [0] * n_classes, [0] * n_classes
			for idx in range(n_classes):
				fprs[idx], tprs[idx], _ = sklearn.metrics.ruc_curve(
					y_test[:, i], y_score[:, i])
			# 合并fpr
			all_fpr = np.unique(np.concatenate(fprs))
			mean_tpr = np.zeros_like(all_fpr)
			# 计算合并后fpr插值
			for idx in range(n_classes):
				mean_tpr += scipy.interp(all_fpr, fpr[idx], tpr[idx])
			mean_tpr /= n_classes
			auc_macro = sklearn.metrics.auc(all_fpr, mean_tpr)

			# 但是和以下结果不同
			auc_macro = sklearn.metrics.roc_auc_score(fprs)
			```

> - 以上分类器均为*one-vs-rest*分类器，m个类别则m个分类器、
	每个样本m个得分

###	*Accuracy*

准确率、误分率：评价分类器性能一般指标

$$\begin{align*}
acc & = \frac 1 N sign(y_i = \hat y_i) \\
& = \frac {TP+TN} N \\
mis & = 1 - acc
\end{align*}$$

> - $y_i$：第$i$样本实际类别
> - $\hat y_i$：第$i$样本预测类别
> - $N$：样本数量

-	对给定测试集，分类器正确分类样本数与总样本数比值
-	0-1损失函数时经验风险

##	*Regression*

-	回归问题

###	方差

回归树中输出变量取值异质性测度

$$
R(t) = \frac 1 {N - 1} \sum_{i=1}^N (y_i(t) - \bar{y}(t)^2) \\
\delta R(t) = R(t) - (\frac {N_r} N R(t_r) + \frac {N_l} N (R(t_l))
$$

###	均方误差（偏差）

$$
MSE = \frac {1} {n} \sum_{i=1}^{n} (y_{i} - \hat{y_{i}})^{2}
$$

基于最小化MSE原则

$$
\begin{align*}
E(Y - \hat{Y}) &= E[f(x) + \epsilon - \hat{f}(X)]^{2} \\
	&= E([f(X) - \hat{f}(X)])^{2} + Var(\epsilon) \\
	&= E_{\tau}[\hat{y}_{0} - E_{\tau}(\hat{y})]^{2} +
		[E_{\tau}(\hat{y}_{0}) - f(x_{0})]^{2} +
		Var(\epsilon) \\
	&= Var_{\tau}(\hat{y}_{0})+ Bias^{2}(\hat{y}_{0}) + Var(\epsilon)\\
\end{align*}
$$

###	$R^2$

$$\begin{align*}
R^2 & = \frac {SSR} {SST}\\
R^2_{adj} & = 1 - \frac {1 - R^2} {n - p - 1}
\end{align*}$$

> - $n, p$：样本量、特征数量
> - $SSR$：回归平方和、组内平方和
> - $SST$：离差平方和
> - $R^2_{adj}$：调整的$R^2$

###	*Akaike Information Criterion*

AIC：赤池信息准则

$$\begin{align*}
AIC & = -2log(L(\hat \theta, x)) + 2p \\
& = nln(SSE/n) + 2p
\end{align*}$$

> - $n, p$：样本量、特征数量
> - $\theta$：带估参数
> - $L(\theta, x)$：似然函数
> - $SSE$：残差平方和

###	*Bayesian Information Criterion*

BIC：贝叶斯信息准则

$$\begin{align*}
BIC & = -2log(L(\hat \theta, x)) + ln(n)p \\
& = nln(SSE/n) + ln(n)p
\end{align*}$$

###	$C_p$

$$\begin{align*}
C_p & = \frac {SSE} {\hat {\sigma^2}} - n + 2p
& = (n - m - 1) \frac {SSE_p} {SSE_m} - n + 2p
\end{align*}$$

> - $p$：选模型特征子集中特征数量
> - $m$：所有特征数量
> - $SSE_p$：选模型中残差平方和
> - $SSE_m$：全模型中残差平方和


