#	Emsemble Learning

##	集成学习

> - 集成学习：训练多个基模型，并将其组合起来，以达到更好的
	预测能力、泛化能力、稳健性
> - *base learner*：基模型，基于**独立样本**建立的、一组
	**具有相同形式**的模型中的一个
> - 组合预测模型：由基模型组合，即集成学习最终习得模型

-	源于样本均值抽样分布思路
	-	$var(\bar{X}) = \sigma^2 / n$
	-	基于独立样本，建立一组具有相同形式的基模型
	-	预测由这组模型共同参与
	-	组合预测模型稳健性更高，类似于样本均值抽样分布方差
		更小

-	关键在于
	-	获得多个独立样本的方法
	-	组合多个模型的方法

-	目前主要包括三个框架
	-	*bagging*
	-	*boosting*
	-	*stacking*

###	原理

*probably approximately correct*：概率近似正确，在概率近似
正确学习的框架中

> - *strongly learnable*：强可学习，一个概念（类），如果存在
	一个多项式的学习算法能够学习它，并且**正确率很高**，那么
	就称为这个概念是强可学习的
> - *weakly learnable*：弱可学习，一个概念（类），如果存在
	一个多项式的学习算法能够学习它，学习的正确率仅比随机猜测
	略好，称此概念为弱可学习的
> - *Schapire*证明：在PCA框架下强可学习和弱可学习是等价的

###	具体措施

弱学习算法要比强学习算法更容易寻找，所以具体实施提升就是
需要解决的问题

-	**改变训练数据权值、概率分布的方法**
	-	提高分类错误样本权值、降低分类正确样本权值

-	**将弱学习器组合成强学习器的方法**
	-	*competeing*
	-	*simple majority voting*
	-	*weighted majority voting*
	-	*confidence-based weighting*

###	学习器组合方式

> - 很多模型无法直接组合，只能组合预测结果

-	*simple majority voting*/*simple average*：简单平均
	$$
	h = \frac 1 K \sum_{k=1}_K h_k
	$$

	> - $h_k$：第k个预测

-	*weighted majority voting*/*weighted average*：加权平均
	$$
	h = \frac {\sum_{k=1}^K w_k h_k} {\sum_{k=1}^K w_k}
	$$

	> - $w_k$：第k个预测权重，对分类器可以是准确率

-	*competing voting*/*largest*：使用效果最优者

-	*confidence based weighted*：基于置信度加权
	$$\begin{align*}
	h = \arg\max_{y \in Y} \sum_{k=1}^K ln(\frac {1 - e_k}
		{e_k}) h_k
	\end{align*}
	$$

	> - $e_k$：第k个模型损失


