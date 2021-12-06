---
title: *Vim* 窗口、展示
categories:
  - Linux
  - Tool
  - Vi
tags:
  - Linux
  - Tool
  - Vi
  - Editing
date: 2021-11-10 15:41:10
updated: 2021-11-26 18:58:15
toc: true
mathjax: true
description: 
---

##	*Window*、*Tab*、*Menu*

-	`:vert[ical]`：可用在窗口分割、调整命令之前，表示竖直分隔
	-	`:vert split`/`:vs`：左右分割窗口
	-	`:vert sb[n]`：左右分割窗口载入`buff[n]`
	-	`:vert diffs file_name`：左右分割窗口`diffs`
	-	`:vert res +/-[n]`：调整窗口大小

-	`:botright`：

###	窗口分割

-	相关命令
	-	`:[ROWS]sp[lit] [FILE]`：水平分割当前窗口，展示当前文件
		-	`ROWS` 为新窗口行数，缺省为当前窗口一半
	-	`:vs[plit]`：竖直分割当前窗口，展示当前文件
	-	`:new`：水平分割当前窗口，新文件
	-	`:vnew`：竖直分割当前窗口，新文件
	-	`:sview`：水平分割当前窗口，只读（缺省当前）文件
	-	`:all`：为命令行启动的 `vim` 每个文件参数分别打开窗口

-	相关快捷键
	-	`<C-w-h/j/k/l>`：切换至左/下/上/右窗口
	-	`<C-w-w>`/`<C-w><C-w>`：轮换窗口（切换活动窗口）
	-	`<C-w-t/b/p>`：切换至最左上/右下/前个窗口

-	相关选项
	-	`splitbelow`：新窗口出现在当前窗口下方
	-	`splitright`：新窗口出现在当前窗口右边

###	窗口调整

-	相关命令
	-	`:[n]winc >/<`：水平方侧扩展/收缩
	-	`:[n]winc +/-`：竖直方向扩展/收缩
	-	`:res[ize]+/-[n]`：竖直方向扩展/收缩
	-	`:vertical res+/-[n]`：水平方向扩展/收缩
	-	`:ressize[n]`：高度设置为n单位
	-	`:vertical res[n]`：宽度设置为n单位

-	相关快捷键
	-	`[ROWS]<C-w-+/->`：竖直方向扩展/收缩当前窗口
	-	`<C-w-=>`：恢复当前窗口高度
	-	`[ROWS]<C-w-_>`：设置窗口高度，缺省设为最大
	-	`<C-w->/<>`：水平方向扩展/收缩当前窗口
	-	`<C-w-H/J/K/L>`：移动窗口至左/下/上/右，占据全部高/宽、一半宽/高
	-	`<C-w-T>`：移动窗口至新标签页
	-	`<C-w-r/x>`：交换窗口位置（具体逻辑未知）

-	相关选项
	-	`winheight`：最小的期望窗口高度
	-	`winminheight`：最小“硬性”高度
	-	`winwidth`：最小的期望窗口宽度
	-	`winminwidth`：最小“硬性”宽度
	-	`equalalways`：关闭、打开新窗口时保持相同大小

> - 水平方向：扩张优先左侧变化，收缩右侧变化
> - 竖直方向：扩展收缩均优先下侧变化

###	滚屏

> - 滚屏基本支持计数前缀

-	向下、向上滚屏
	-	`CTRL-e`、`CTRL-y`：窗口向下、向上按行滚动 `COUNT` 行
	-	`CTRL-d`、`CTRL-u`：窗口向下、向上滚动 `scroll` 选项指定的行数
		-	光标尝试同时移动
		-	给出计数前缀 `COUNT`，会先将 `scroll` 选项设置为 `COUNT`
		-	光标位于缓冲区尾行、首行时，命令无效
	-	`<S-Down>`/`<PageDown>`/`CTRL-f`、`<S-Up>`/`<PageUp>`/`CTRL-b`：窗口向下按页滚动
		-	光标尝试同时移动
	-	`[COUNT]z+`、`[COUNT]z^`：重绘，将 `COUNT` 行置于屏幕顶
		-	`COUNT`：缺省下页、上页首行
		-	光标置于行首非空白字符
	-	`[COUNT]z[HEIGHT]<CR>`、`[COUNT]zt`：重绘，将 `COUNT` 位置置于屏幕顶
		-	`COUNT`：缺省光标所在行
		-	`HEIGHT`：将当前窗口高设为 `HEIGHT`
		-	光标置于行首非空白字符、当前列
	-	`[COUNT]z.`、`[COUNT]zz`：重绘，将 `COUNT` 位置置于屏幕中间
		-	`COUNT`：缺省光标所在行
		-	光标置于行首非空白字符、当前列
	-	`[COUNT]z-`、`[COUNT]zb`：重绘，将 `COUNT` 位置置于屏幕底
		-	`COUNT`：缺省光标所在行
		-	光标置于行首非空白字符、当前列

-	左右滚屏：仅在 `wrap` 复位时有效
	-	`z<Right>`/`zl`、`z<Left>`/`zh`：文本视图向右移动 `COUNT` 字符
	-	`zL`、`zH`：文本视图向右移动半个屏幕
	-	`zs`、`ze`：文本视图滚动至光标位于屏幕开始（最左）、结束（最右）

-	同步滚屏
	-	`:syncbind`：强制所有 `scrollbind` 窗口有相同相对偏移，即窗口滚屏到缓冲区顶部时，所有 `scrollbind` 置位窗口滚屏到缓冲区顶部

-	相关选项
	-	`scrollbind`：置位窗口的滚屏时，所有置位窗口会同步滚屏
		-	置位 `scrollbind` 选项的窗口记录其 “相对偏移”，即使窗口无法继续滚屏，也会记录超越滚屏界限的数量
		-	若同时置位 `diff` 选项，则利用两个缓冲区的差异精确同步位置
	-	`scollopt`：控制 `scrollbind` 选项行为
	-	`scrolloff`：指定窗口可编辑范围
		-	`scrolloff` 范围将作为编辑边界，光标至此开始滚动窗口

-	其他命令
	-	`<C-g>`：在命令行处展示光标位置

###	缓冲区

-	放弃修改、重新加载、保存
	-	`:w[rite]`：保存此次修改
	-	`:e!`：放弃本次修改
		-	和 `:q!` 不同，不退出vim
	-	`:bufdo e!`：放弃vim所有已打开文件修改
	-	`:e[dit]`：重新加载文件
	-	`:e#`：当前窗口加上个buffer（反复切换）
	-	`:bufdo e`：重新加载vim所有已打开文件
	-	`:saveas new_file_name`：另存为，不删除原文件
	-	`:sb[n]`：split窗口加载第n个buffer
	-	`:b[n]`：当前窗口加载第n个buffer
	-	`:n[n]`：当前窗口加载下/第n个buffer

###	标签页

-	相关命令
	-	`[COUNT]gt`/`gT`：下、上一个 Tab
		-	`COUNT`：重复 `COUNT` 次，即下、上 `COUNT` 哥个 Tab
	-	`:tabn [COUNT]`：查看下 `COUNT` 个 Tab，缺省下个
	-	`:tabn[ew] [OPT] [CMD] [FILE_NAME]`：打开新 Tab
	-	`:tabc[lose]`：关闭当前 Tab
	-	`:tabo[nly]`：关闭其他 Tab
	-	`:tabs`：查看所有打开的 Tab 信息
	-	`:tabp[re]`：查看前一个 Tab
	-	`:tabfirst`、`:tablast`：第一个、最后一个 Tab

##	折叠

-	折叠开闭命令
	-	`zo`、`zc`、`za`：展开、关闭、切换折叠
		-	支持前缀计数，表示影响的折叠深度
		-	支持可视模式，影响选中区域所有折叠
	-	`zO`、`zC`、`zA`：（循环）打开、关闭、切换光标下全部折叠
	-	`zv`：查看光标所在行，打开足够折叠保证光标所在行不被折叠
	-	`zx`：更新折叠，强制重新计算折叠
	-	`zm`、`zr`：（整个缓冲区）提高、降低折叠水平
		-	即更新 `foldlevel` 选项
	-	`zM`、`zR`：（整个缓冲区）打开、关闭折叠
	-	`zn`、`zN`、`zi`：禁用、启用、反转折叠
		-	即复位、置位、翻转 `foldenble`
	-	`:[RANGE]foldo[pen][!]`：打开 `RANGE` 范围内折叠
		-	`!`：打开所有层级折叠
	-	`:[RANGE]foldo[pen][!]`：关闭 `RANGE` 范围内折叠
		-	`!`：关闭所有层级折叠

-	折叠跳转
	-	`[z`、`]z`：移至当前折叠开始、结束处
		-	若已经在开始处，移动到包含折叠的折叠开始处
		-	支持前缀计数，表示重复命令次数
	-	`zk`、`zj`：移至上个折叠结束、下个折叠开始处
		-	支持 *操作符-动作*
		-	支持前缀计数，表示重复命令次数

-	折叠命令执行
	-	`:[RANGE]foldd[oopen] {CMD}`：在 `RANGE` 范围内所有未折叠行应用 `CMD`
	-	`:[RANGE]folddoc[lose] {CMD}`：在 `RANGE` 范围内所有已折叠行应用 `CMD`

-	折叠相关选项
	-	`foldenable`：允许折叠
	-	`foldmethod`：折叠模式
		-	`manual`：手动折叠选中的行
		-	`indent`：缩进折叠，缩进程度表示折叠水平
		-	`marker`：标记折叠，折叠标记范围内的行
		-	`syntax`：语法折叠
		-	`expr`：表达式折叠
		-	`diff`：折叠未更改标记
	-	`foldcolumn=<NUM>`：用侧边 `NUM` 列展示可折叠状态
	-	`foldlevel=<NUM>`：设置折叠水平
		-	`0`：折叠所有
		-	`99`：折叠至 99 级
	-	`foldlevelstart=<NUM>`：打开文件时折叠水平
		-	`-1`：初始不折叠

###	折叠模式

-	*Indent* 折叠：按缩进折叠

-	*Marker* 折叠：`{{{`、`}}}`标记折叠区域
	-	各花括号之间无空格
	-	`{{{<NUM>` 后可跟数字标记折叠水平

-	`marker`、`manual` 折叠命令
	-	`zf[MOTION]`：创建折叠操作符
		-	`v_zf`：为选中区域创建折叠操作符
		-	`marker` 时，创建折叠操作符
		-	`manual` 时，新建折叠关闭，置位 `foldenable`
		-	示例
			-	`zf[n]G`：创建当前行至第n行折叠
			-	`[n]zf[+]`：创建当前行至后n行折叠
			-	`[n]zf[-]`：创建当前行至前n行折叠
	-	`[COUNT]zF`：同 `zf`，但对 `COUNT` 行创建折叠
	-	`:[RANGE]fo[ld]`：对 `RANGE` 行内创建折叠，同 `zf`
	-	`zd`：删除光标上折叠
		-	`v_zd`：删除选择区域内包含的顶层折叠
	-	`zD`：（循环）删除光标上全部折叠
		-	`v_zD`：删除选择区域内包含的所有折叠
	-	`zE`：删除窗口中所有折叠



