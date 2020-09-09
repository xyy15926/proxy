---
title: 评分卡模型
categories:
  - Machine Learning
  - Model
  - Linear Model
tags:
  - ML Model
  - Linear Model
date: 2020-08-10 19:37:47
updated: 2020-08-10 19:37:47
toc: true
mathjax: true
comments: true
description: 信用评分卡模型，最常见的金融风控手段之一，根据
	客户的各种属性和行为数据，利用一定的信用评分模型，对客户
	进行信用评分，据此决定是否给予授信以及授信的额度和利率
---

##	评分卡模型

评分卡模型

![score_card_sketch](imgs/score_card_sketch.png)

-	评分卡模型在不同业务阶段体现的方式、功能不一样，按照借贷
	用户的借贷时间可以分为
	-	申请评分卡*Application Score Card*：贷前申请评分卡，
		A卡
	-	行为评分卡*Behavior Score Card*：贷中行为评分卡，B卡
	-	催收评分卡*Collection Score Card*：贷后催收评分卡，
		C卡

-	用户的总评分等于基于分加上客户各个属性的评分
	-	对用户评分的有效性为1个月左右
	-	同时假设用户短期内属性变化不大，不会跨越箱边界，保证
		模型稳定性

###	评分卡开发流程

![score_card_construction](imgs/score_card_construction.png)


-	数据清洗
	-	删除缺失率超过某阈值的变量
	-	将剩余变量中缺失值、异常值作为一种状态，即将其作为
		变量的一个取值/箱

-	特征分箱
	-	有效处理特征缺失值、异常值
	-	数据和模型更稳定
	-	将所有特征统一为分类型变量
	-	简化逻辑回归模型，降低过拟合风险，提高模型泛化能力
	-	分箱后变量才可以使用标准评分卡格式，对不同分段评分

-	WOE编码

####	特征筛选

-	预测能力
-	特征之间线性相关性
-	简单性：容易生成、使用
-	强壮性：不容易绕过
-	业务上可解释性

#####	单变量筛选

-	基于IV值
-	stepwise变量筛选
-	正则化变量筛选：LASSO
-	特征重要度：RF、GBDT








