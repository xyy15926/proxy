---
title: GGPLOT
tags:
  - R语言
categories:
  - R语言
date: 2019-03-21 17:27:15
updated: 2019-02-17 11:57:08
toc: true
mathjax: true
comments: true
description: GGPLOT
---

```R
p <- ggplot(data=mtcars, aes(x=wt, y=mpg))

geom_bar(color, fill, alpha)
geom_histogram(color, fill, alpha, linetype, binwidth)
geom_boxplot(color, fill, alpha, notch, width)
geom_violin(color, fill, alpha, linetype)
geom_density(color, fill, alpha, linetype)

geom_rug(color, side)
geom_smooth(method, formula, color, fill, linetype, size)


geom_hline(color, alpha, linetype, size)
geom_gitter(color, size, alpha, shape)
geom_line(color, alpha, linetype, size)
geom_point(color, alpha, shape, size)
geom_vline(color, alpha, linetype, size)

geom_text()
```

-	`color`：点、线、填充区域着色
-	`fill`：对填充区域着色，如：条形、密度区域
-	`alpha`：颜色透明度，0~1逐渐不透明
-	`linetype`：图案线条
	-	1：实线
	-	2：虚线
	-	3：点
	-	4：点破折号
	-	5：长破折号
	-	6：双破折号
-	`size`：点尺寸、线宽度
-	`shape`：点形状
	-	1：开放的方形
