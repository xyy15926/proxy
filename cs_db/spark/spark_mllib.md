---
title: Spark MLLib
categories:
  - DataBase
  - Spark
tags:
  - Spark
  - Machine Learning
  - Classification
  - Regression
  - Collaborative Filtering
date: 2019-07-11 00:51:41
updated: 2021-07-19 09:01:22
toc: true
mathjax: true
comments: true
description: Spark MLLib
---

##	MLLib

*Spark MLLib*：Spark平台的机器学习库

-	能直接操作RDD数据集，可以和其他BDAS其他组件无缝集成，
	使得在全量数据上进行学习成为可能

-	实现包括以下算法
	-	Classification
	-	Regression
	-	Clustering
	-	Collaborative Filtering
	-	Dimensionality Reduction

-	MLLib是MLBase中的一部分
	-	MLLib
	-	MLI
	-	MLOptimizer
	-	MLRuntime

-	从Spark1.2起被分为两个模块
	-	`spark.mllib`：包含基于RDD的原始算法API
	-	`spark.ml`：包含基于DataFrame的高层次API
		-	可以用于构建机器学习PipLine
		-	ML PipLine API可以方便的进行数据处理、特征转换、
			正则化、联合多个机器算法，构建单一完整的机器学习
			流水线

> - MLLib算法代码可以在`examples`目录下找到，数据则在`data`
	目录下
> - 机器学习算法往往需要多次迭代到收敛为止，Spark内存计算、
	DAG执行引擎象相较MapReduce更理想
> - 由于Spark核心模块的高性能、通用性，Mahout已经放弃
	MapReduce计算模型，选择Spark作为执行引擎

##	`mllib.classification`

###	Classification 

####	Logistic Regression

```python
from pyspark.mllib.classification import \
	LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabledPoint

def parse_point(line):
	value = [float(i) for i line.split(", \r\n\t")

data = sc.textFile("data/mllib/sample_svm_data.txt")
parsed_data = data.map(parse_point)
	# map `parse_point` to all data

model = LogisticRegressionWithLBFGS.train(parsed_data)
labels_and_preds = parsed_data.map(lambda p: (p.label, model.predict(p.features)))
train_err = labels_and_preds \
	.filter(lambda lp: lp[0] != lp[1]) \
	.count() / float(parsed_data.count())

model.save(sc, "model_path")
same_model = LogisticRegressionModel.load(sc, "model.path")
```

-	Decision Tree
-	Random Forest
-	Gradient
-	boosted tree
-	Multilaye Perceptron
-	Support Vector Machine
-	One-vs-Rest Classifier
-	Naive Bayes

###	Clustering

####	K-means

```python
import numpy as np
from pyspark.mllib.clustering import KMeans, KMeansModel

data = sc.textFile("data/mllib/kmeans_data.txt")
parsed_data = data.map(lambda line: np.array([float(i) for i in line.split()]))

cluster_model = KMeans.train(
	parsed_data,
	maxIteration=10,
	initializationMode="random"
)
def error(point):
	center = cluster_model.centers[cluster.predict(point)]
	return np.sqrt(sum([i**2 for i in (point - center)]))
WSSSE = parsed_data \
	.map(lambda point.error(point)) \
	.reduce(lambd x, y: x + y)

cluster_model.save(sc, "model_path")
same_model = KMeansModel.load(sc, "model_path")
```

####	Gaussian Mixture Model(GMM)

-	混合密度模型
	-	有限混合模型：正态分布混合模型可以模拟所有分布
	-	迪利克莱混合模型：类似于泊松过程
-	应用
	-	聚类：检验聚类结果是否合适
	-	预测：
	# todo

```md
import numpy as np
from pyspark.mllib.clustering import GussianMixture, \
	GussianMixtureModel

data = sc.textFile("data/mllib/gmm_data.txt")
parsed_data = data.map(lambda line: np.array[float(i) for i in line.strip()]))

gmm = GaussianMixture.train(parsed_data, 2)
for w, g in zip(gmm.weights, gmm.gaussians):
	print("weight = ", w,
		"mu = ", g.mu,
		"sigma = ", g.sigma.toArray())

gmm.save(sc, "model_path")
same_model = GussainMixtureModel.load(sc, "model_path")
```

####	Latent Dirichlet Allocation(LDA)

```md
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors

data = sc.textFile("data/mllib/sample_lda_data.txt")
parsed_data = data.map(lambda line: Vector.dense([float(i) for i in line.strip()]))

corpus = parsed_data.zipWithIndex() \
	.map(lambda x: [x[1], x[0]).cache()
ldaModel = LDA.train(corpus, k=3)

topics = ldaModel.topicsMatrix()

for word in range(0, ldaModel.vocabSize()):
	for topic in word:
		print(topic)

ldaModel.save(sc, "model_path")
same_model = LDAModel.load("model_path")
```

-	Disecting K-means

###	Regression

####	Linear Regression

-	耗时长、无法计算解析解（无意义）
-	使用MSE作为极小化目标函数，使用SGD算法求解

```python
from pyspark.mllib.regression import LabledPoint, \
	LinearRegressionWithSGD, LinearRegressionModel

def parse_point(line):
	value = [float(i) for i line.split(", \r\n\t")

data = sc.textFile("data/mllib/ridge-data/lpsa.data")
parsed_data = data.map(parse_point)
	# map `parse_point` to all data

model = LinearRegressionWithSGD.train(
	parsed_data,
	iteration=100,
	step=0.00000001
)
values_and_preds = parsed_data.map(lambda p:(p.label, model.predict(p.features)))
MSE = values_and_preds \
	.map(lambda vp: (vp[0] - vp[1]) ** 2) \
	.reduce(lambda x, y: x + y) / values_and_preds.count()

model.save(sc, "model_path")
	# save model
same_model = LinearRegressionModel.load(sc, "model_path")
	# load saved model
```

-	Generalized Linear Regression
-	Decision Tree Regression
-	Random Forest Regression
-	Gradient-boosted Tree Regression
-	Survival Regression
-	Isotonic Regression

###	Collaborative Filtering

