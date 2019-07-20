#	Model Enhancement

##	Emsemble Learning

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

###	分类

-	*homogenous ensemble*：同源集成，基学习器属于同一类型
		-	*bagging*
		-	*boosting*

-	*heterogenous ensemble*：异源集成，基学习器不一定属于同
	一类型
	-	*stacking*

||Target|Data|parallel|Classifier|Aggregation|
|-----|-----|-----|-----|-----|
|Bagging|减少方差|基于boostrap随机抽样，抗异常值、噪声|模型间并行|同源不相关基学习器，一般是树|分类：投票、回归：平均|
|Boosting|减少偏差|基于误分分步|模型间串行|同源若学习器|加权投票|
|Stacking|减少方差、偏差|K折交叉验证数据、基学习器输出|层内模型并行、层间串行|异质强学习器|元学习器|

> - 以上都是指原始版本、主要用途

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

##	Meta Learning

元学习：自动学习关于关于机器学习的元数据的机器学习子领域

-	元学习主要目标：使用学习到元数据解释，自动学习如何
	*flexible*的解决学习问题，借此提升现有学习算法性能、
	学习新的学习算法，即学习学习

-	学习算法灵活性即可迁移性，非常重要
	-	学习算法往往基于某个具体、假象的数据集，有偏
	-	学习问题、学习算法有效性之间的关系没有完全明白，对
		学习算法的应用有极大限制

###	要素

-	元学习系统必须包含子学习系统
-	学习经验通过提取元知识获得经验，元知识可以在先前单个
	数据集，或不同的领域中获得
-	学习*bias*（影响用于模型选择的前提）必须动态选择
	-	*declarative bias*：声明性偏见，确定假设空间的形式
		，影响搜索空间的大小
		-	如：只允许线性模型
	-	*procedural bias*：过程性偏见，确定模型的优先级
		-	如：简单模型更好

###	*Recurrent Neural networks*

RNN：*self-referential* RNN理论上可以通过反向传播学习到，
和反向传播完全不同的权值调整算法

###	*Meta Reinforcement Learning*

MetaRL：RL智能体目标是最大化奖励，其通过不断提升自己的学习
算法来加速获取奖励，这也涉及到自我指涉

