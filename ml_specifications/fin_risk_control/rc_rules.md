---
title: 风控规则
categories:
  - ML Specification
  - FinTech
  - Risk Control
tags:
  - Machine Learning
  - FinTech
  - Financial Risk
  - Swap Set Analysis
  - Rule Evaluation
date: 2021-01-11 08:49:48
updated: 2021-07-16 16:28:43
toc: true
mathjax: true
description: 
---

##	风控规则

-	规则的类型
	-	条件判断：决策路径独立
	-	决策表：决策路径有交集、规律
	-	决策树：决策路径不规律、可能重复检查同一数据

> - 规则引擎：接受数据输入，解释业务规则，根据业务规则、使用
	预定义语义做出业务决策

###	制定原则

-	监管、公司政策类
	-	年龄准入
	-	行业准入
		-	有金融属性行业
		-	政策敏感娱乐行业
	-	地域准入
	-	场景准入
-	风控层面
	-	黑名单类
	-	多头类：申请次数
	-	共债类：申请量
	-	反欺诈类
	-	评分拒绝类

###	规则发现

####	规则评分

-	强弱规则
	-	强规则：可作为独立规则，直接指导决策
	-	弱规则：具有一定区分度，但不决定性

-	弱规则可组合使用，通过评分方式衡量弱规则
	-	使用规则评分衡量规则影响力
	-	规则影响力则可以通过命中坏占比、odds变动衡量
	-	设置阈值，命中规则的评分之和超过阈值才触发报警

####	笛卡尔积法

-	步骤
	-	获取变量：定义好坏，关联特征变量
	-	变量筛选：通过IV值等指标快速筛选变量
	-	指标统计：分组统计申请量、放款量、坏账量等指标
	-	透视呈现：分申请率、放款率、坏账率等指标制作交互，如列联表等
	-	规则提取：结合各维度选择满足要求的组别，提取规则逻辑
	-	规则评估：评估跨期稳定性
	-	策略上线

####	决策树法

-	决策树法优势
	-	可根据划分依据自动对人群细分
-	决策树法缺陷
	-	难以调整决策树划分结果
	-	划分结果可能缺乏业务意义
	-	可能出现过拟合现象

###	规则阈值设置

-	阈值设置指标
	-	*Lift* 值
	-	*收益/风险比*

-	阈值设置依据
	-	对分类取值，根据 *Lift* 值、*收益/风险比* 确定是否作为规则
	-	对有序、数值取值，结合不同阈值点计算 *Lift* 值、*收益/风险比*，绘制曲线
		-	曲线平缓变化，则阈值切分收益变化稳定，阈值调整空间比较大
		-	曲线存在明显陡、缓变化，则阈值切分收益在拐点处收益较大，阈值调整空间有限

##	规则评价

-	案件调查
	-	用信前报警调查
	-	逾期后调查
	-	根据不同目标，可以对不同的案件区分重点调查

###	线下 / 离线（标签已知）效果评估

-	自身效果评估
	-	混淆矩阵
		-	*TPR*/*FPR*
		-	准确率/误拒率
	-	提升度
		-	拒绝样本中坏样本*Lift*提升度
		-	通过样本中好样本*Lift*提升度
	-	通过率、拒绝率
	-	加权收益：好数量 * 好收益 + 坏数量 * 坏收益
-	对比/增量效果评估：和其他数据源比较
	-	有效差异率：查得命中 / 其他通过且为坏样本
	-	无效差异率：查得命中 / 其他拒绝

> - 类似名单类数据评估

###	线上 / 在线（标签未知）效果评估

-	规则报警次数、报警率
	-	规则（触发）报警次数：命中规则后账户被拒绝次数
		-	对强规则，即为规则命中次数
		-	对弱规则，小于规则命中次数
	-	规则报警率 = 规则报警次数 / 规则命中次数
	-	规则报警率低、趋势走低表明规则需修正

-	规则调查次数、调查率
	-	规则调查次数 = 对案件调查分析时调查其次数
		（短路调查）
	-	规则调查率 = 规则调查次数 / 规则报警次数
	-	调查率低则因考虑其他规则替代该规则，或`or`合并规则
	-	规则可以为调查提供提示，而过多不能给调查提供提示的
		规则反而浪费时间

-	规则命中次数、命中率
	-	规则命中次数 = 命中触发报警之后被认定为坏样本数
	-	规则命中率 = 规则命中次数 / 规则报警次数

-	综合命中次数
	-	综合命中次数 = 规则命中次数 + 逾期调查认定坏样本数
	-	综合命中率 = 综合命中次数 / 规则报警次数

> - 在线效果效果是无法在体系内自评估的，必须引入外部信息，包括：人工审核、额外数据源、扩招回机制等

###	规则稳定性

####	通过率波动应对

-	寻找通过率变动的时点
-	计算各维度通过率波动程度PSI
	-	定位各策略节点主次影响
	-	分析主要影响策略节点规则、阈值
-	指导决策

####	逾期率波动应对

-	定位逾期率波动客群：存量客户、新增客户
	-	MOD

###	旁路规则

###	*Swap Set Analysis*

> - 新、旧模型可用离线指标比较优劣，但最终要在业务中比较通过率、坏账率，二者正相关，*swap set* 则是反应模型的通过的变化

-	*Swap Set Analysis* 用于分析新、旧模型更替
	-	根据订单在新、旧模型的通过情况，可以分为三类
		-	*Swap-in Population*：旧模型拒绝但新模型接受
		-	*Swap-out Population*：旧模型接受但新模型拒绝
		-	*No Change*：新、旧模型同时接受、拒绝
	-	从 *swap set* 角度评价 “新模型优于旧模型”
		-	*Swap-in Population >= Swap-out Population* 且坏账率不升
		-	*Swap-in Population = Swap-out Population* 、坏账率不变，但用户响应率提升

-	实务中，已上线的旧模型拒绝订单无法获取表现期，只能通过拒绝推断近似得到坏账率
	-	同时间窗 *A/B-Test*：切分流量让旧模型只打分不拒绝
	-	跨时间窗 *A/B-Test*：用旧模型在灰度期坏账率替代

###	扩召回

扩召回：独立召回之外，利用额外模型扩召回部分样本

-	此处召回一般指通过 **成熟** 的规则、模型从全体中获取部分样本
	-	召回一般为历史沉淀、专家经验规则
	-	召回的理由充足，但泛化性较差

-	扩召回和二次排序训练用的样本是相同的，但
	-	二次排序是在召回的样本基础上再次排序
		-	目标：（全局）排序能力
		-	评价标准：*AUC*、头部准召
	-	扩召回一般是独立于召回建立的模型
		-	目标：学习召回样本的规律，完善召回机制、补充召回样本
			-	因此，扩招回也可以用召回样本作为正样本
			-	扩召回也可用于在线验证新、旧规则的有效性
		-	评价标准：额外召回准确率（对召回样本的学习能力）
			-	事实上，若采用召回样本作为正样本，则 *AUC* 为 1 的扩召回是无价值的，只是复现了召回
		-	特征：可能包含一些专供于扩召回使用的特征
		-	扩召回的正样本可能还包括人工举报、隐案等

##	准入规则

-	风控准入规则应为强拒绝规则
	-	不满足任何规则均会被拒绝
	-	规则无需经过复杂的规则衍生
	-	策略理念：验证借款人依法合规未被政策限制
	-	风控流程中首道防线
		-	准入策略已经趋同
		-	但对不同信贷场景仍应采取更适应业务的准入规则

###	基础认证模块

-	风控基础认证模块：验证申请人真实性
	-	身份证信息验证
	-	人脸信息验证
	-	银行卡四要素验证
	-	运营商三要素验证

###	按数据来源分类

-	个人信用类
	-	个人基本信息
		-	年龄准入
		-	地区准入
		-	行业准入
	-	经济能力信息
		-	月收入
		-	流水
	-	社交信息
-	设备信息
	-	短信
	-	APP安装信息
-	外部数据源
	-	征信报告
	-	外部黑名单
-	行为数据
	-	活动轨迹
	-	登录、注册时间
-	评分卡规则

##	黑、白名单

###	白名单

-	白名单：风险相对可知可控的客户构成的内部名单
	-	业务初期：通过白名单控制入口
		-	控制放量节奏
		-	降低风险
		-	通过宽松风控规则提高审批通过率
		-	通过贷前策略规则筛选白名单，协助调整贷前策略
	-	业务中期：部分客户走特殊的贷前审批流程，满足特殊审批
		要求

-	白名单筛选方式：有部分存量数据情况下
	-	联合建模：缺乏特定业务场景预测变量，与外部机构建模
		补充预测变量
	-	内部数据探索：寻找与违约表现相关性较强的特征规则
		-	类似场景、产品
		-	纯粹凭借专家经验规则
	-	引入外部数据匹配

###	黑名单

-	黑名单：还款能力、还款意愿不能满足正常客户标准
	-	通常多个好客户才能覆盖坏客户的本金损失
	-	通过黑名单客户全部拒绝，但是对于导流助贷机构，业务
		核心是流量和客户质量，拒绝全部黑名单客群成本巨大，
		可能会随机、结合评分放过部分

-	黑名单建立
	-	建立黑名单参考维度
		-	还款表现
		-	渠道
		-	利率
		-	失信名单
	-	黑名单主体
		-	身份证
		-	手机号
		-	邮箱
		-	银行卡
		-	IP

####	三方黑名单

-	自建黑名单命中率不高（二次申请概率低），且需要长期
	积累

-	不同三方黑名单往往会有其侧重点
	-	团伙欺诈名单
	-	公安、司法名单
	-	被执行人名单

-	三方黑名单效果也有好有坏，对效果较差、但通过率影响
	不大黑名单也可以考虑保留
	-	黑名单一般是查得收费，外挂较多黑名单不会提升成本
	-	黑名单可视为容错机制，黑名单不一定能所有样本上
		表现优秀，保留其可防止欺诈团伙等集中攻击

-	同样值得注意的是，黑名单的质量需要考核
	-	非公信黑名单定义各家不同
	-	名单没有明确的退出机制
	-	黑名单按查得收费，有些黑名单会掺沙子
	-	有些名单提供商同时作为信贷放贷方，有动力将优质客户
		截留，将其添加进名单






