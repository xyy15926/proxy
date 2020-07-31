---
title: 高维数据检索方法
tags:
  - 算法
categories:
  - 算法
date: 2019-06-04 23:11:44
updated: 2019-06-04 23:11:44
toc: true
mathjax: true
comments: true
description: 高维数据检索方法
---

##	相似性检索

相似性检索：从指定目标集合中检索出与给定样本相似的目标

> - *range searches*：范围检索，给定查询点、检索距离阈值
> - *K-neighbor searches*：K近邻检索，给定查询点、检索结果
	数量

-	待检索目标、样本：以指定*feature space*中的高维数据点
	表示

-	相似性检索则在相应*metric space*中搜索样本点最近邻作为
	检索结果

-	关键：对待检索的目标建立有效的**相似性索引**
	-	对待检索目标进行预划分，在对给定样本进行检索时，只需
		对比相似索引中给出的可能相似的目标
	-	减少相似性检索的对比次数、I/O，让相似性检索在大规模
		数据集中应用成为可能

##	*Tree-Based Index*

基于树结构的索引

-	向量维度大于20之后，仍然需要扫描整个向量集合的大部分，
	与线性扫描没有太大差别

-	包括
	-	*kd-tree*
	-	*R-tree*
	-	R\*-tree
	-	X-tree
	-	SS-tree
	-	SR-tree
	-	VP-tree
	-	metric-trees

##	*Hasing-Based Index*

基于哈希的索引技术：利用*LSH*函数简化搜索

-	*locality sensitive hashing*：*LSH*，局部敏感哈希，特征
	向量越接近，哈希后值越可能相同
	-	局部敏感哈希值能够代表代替原始数据比较相似性
	-	支持对原始特征向量进行**非精确匹配**

-	hash技术能从两个方面简化高维数据搜索

	-	提取特征、减小特征维度
		-	在损失信息较小的情况下对数据进行降维
		-	hash函数（特征提取方法）选择依赖于对问题认识
		-	一般都归于特征提取范畴

	-	划分特征空间（哈希桶）、缩小搜索空间
		-	将高维特征映射到1维先进行近似搜索得到候选集，
			然后在候选集中进行精确搜索
		-	hash函数的选择取决于原始特征表示、度量空间
		-	一般LSH都是指此类哈希技术

####	提取特征

-	*average hashing*：*aHash*，平均哈希
-	*perceptual hashing*：*pHash*，感知哈希
-	*differantiate hashing*：*dHash*，差异哈希

####	划分空间

-	*MinHashing*：最小值哈希，基于Jaccard系数
-	基于汉明距离的LSH
-	基于曼哈顿距离的LSH
-	*Exact Euclidean LSH*：*E2LSH*，基于欧式距离

##	*Visual Words Based Inverted Index*

向量化方法：将向量映射为标量，为（图像）特征建立
*visual vocabulary*

-	基于K-means聚类（层级K-means、近似K-means）
-	在图像检索实际问题中取得了一定成功
-	K-means聚类算法的复杂度与图像特征数量、聚类数量有关
	-	图像规模打达到百万级时，索引、匹配时间复杂度依然较高

> - *visual vocabulary*：视觉词库，代表聚类类别整体
> - *visual word*：视觉单词，每个代表一个聚类类别

