#	时间序列分析

时间序列数据：在不同时间点收集到的数据，反映某事物、现象随
实际变化状态、程度

##	描述性时序分析

通过直观的数据比较、绘图观测，寻找序列中蕴含的发展规律

-	操作简单、直观有效，是时序分析的第一步
-	但是只能展示非常明显的规律性
-	最早的时序分析方法，所有时序分析的基础
-	帮助人们找到自然规律
	-	尼罗河的泛滥
	-	范蠡稳定粮价
	-	小麦价格指数序列
	-	太阳黑子运动规律

##	确定性时序分析

根据序列的观察特征，先构想一个序列运行的理论，默认序列按照
此理论确定性运作

-	侧重于确定性信息的提取
-	通常不能通过分析误差自行修正模型，只能通过新的模型假定，
	推翻旧模型实现分析方法的改进
-	假定条件决定了序列的拟合精度，如果确定性的假定条件不对，
	误差将很大，因此限制其使用范围

###	确定性时域分析方法

-	原理：事件的发展通常具有一定的**惯性**，用统计语言描述
	就是序列值之间存在一定的相关关系，即某种统计规律

-	目的：寻找序列值之间的相关关系的统计规律，并拟合适当数学
	模型描述，进而用于预测

-	特点
	-	理论基础扎实
	-	操作步骤规范
	-	分析结果易于解释

####	常用领域

-	宏观经济领域的*time series decomposition*

-	确定性趋势预测
	-	趋势预测：线性趋势预测、非线性趋势预测
	-	指数平滑预测：简单、两参、三参指数平滑

###	频域分析方法

假设任何一种无趋势的实现序列，都可以分解成若干不同频率的
周期波动（借助傅里叶变换，用三角函数逼近）

##	随机性时序分析

主要是时域分析方法，是目前时序分析的主流方法

##	时域分析发展

###	启蒙阶段

-	AR模型：George Undy Yule
-	MA模型、Yule-Walker方程：Sir Gilbert Thomas Walker

###	核心阶段

-	ARIMA：Box & Jenkins书中系统的阐述了ARIMA模型的识别
	、估计、检验、预测原理和方法，现在被称为经典时间序列
	分析方法，是时域分析的出核心内容

###	完善阶段

-	异方差场合
	-	ARCH：Robert Fry Engle
	-	GARCH：Bollerslov
	-	GARCH衍生模型
		-	EGARH
		-	IGARCH
		-	GARCH-M
		-	NGARCH
		-	QGARCH
		-	TGARCH

-	多变量场合
	-	ARIMAX：Box & Jenkins
	-	co-intergration and error correction model：
		C.Granger，协整理论
	-	SYSLIN：Klein，宏观经济连理方程组模型
	-	Vector Autoregressive Model：Sims，货币政策及其
		影响

-	非线性场合
	-	Threshold Autoregressive Model
	-	Artificical Neural Network
	-	Hebbian Learning（神经可塑性假说）
	-	Multivariate Adaptive Regression Splines
	-	Linear Classifier
	-	Support Vector Machines

