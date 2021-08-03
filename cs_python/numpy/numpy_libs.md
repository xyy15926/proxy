---
title: Numpy 附加库
categories:
  - Python
  - Numpy
tags:
  - Python
  - Numpy
  - Finance
  - Histogram
  - Set
date: 2021-03-11 09:36:49
updated: 2021-03-11 09:36:49
toc: true
mathjax: true
description: 
---

##	财金

|Function|Desc|
|-----|-----|
|`fv(rate,nper,pmt,pv[,when])`|未来值|
|`pv(rate,nper,pmt[,fv,when])`|现值|
|`npv(rate,values)`|净现值|
|`pmt(rate,nper,pv[,fv,when])`|等额本息，每期付款|
|`ppmt(rate,per,nper,pv[,fv,when])`|等额本息中第`per`期本金|
|`ipmt(rate,per,nper,pv[,fv,when])`|等额本息中第`per`期利息|
|`irr(values)`|内部收益率|
|`mirr(values,finance_rate,reinvest_rate)`|考虑期内再融资成本`finance_rate`、收益再投资收益`reinvest_rate`|
|`nper(rate,pmt,pv[,fv,when])`|每期付款|
|`rate(nper,pmt,pv,fv[,when,guess,tol,...])`|每期间的利率|

-	参数说明
	-	`pv`：现值
	-	`fv`：未来值
	-	`when`：期初或期末付款
		-	`0`/`end`
		-	`1`/`begin`
	-	`pmt`：*Payment*，每期付款
	-	`ppmt`：*Principle of Payment*，每期付款中本金
	-	`ipmt`：*Interest of Payment*，每期付款中利息

-	值说明
	-	正值：收入
	-	负值：支出

##	Histogram

|Function|Desc|
|-----|-----|
|`histogram(a[,bins,range,normed,weights,...])`||
|`histogram2d(x,y[,bins,range,normed,weights,...])`||
|`histogramdd(sample[,bins,range,normed,weights,...])`||
|`bincount(x[,weights,minlength])`||
|`histogram_bin_edges(a[,bin,range,weights])`||
|`digitize(x,bins[,right])`||

##	Set

###	Operation

|Routine|Function Version|
|-----|-----|
|`in1d(ar1,ar2[,assume_unique,invert])`|是否包含，始终返回1维数组|
|`isin(element,test_element[,...])`|保持`element`shape返回|
|`intersect1d(ar1,ar2[,assume_unique,...])`|交集|
|`union1d(ar1,ar2[,assume_unique,...])`|并集|
|`setdiff1d(ar1,ar2[,assume_unique,...])`|`ar1`-`ar2`|
|`setxor1d(ar1,ar2[,assume_unique,...])`|差集|

###	Unique

|Routine|Function Version|
|-----|-----|
|`unique(ar[,return_index,return_inverse,return_counts,axis])`|返回唯一值|







