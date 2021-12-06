---
title: *Vim* 模块功能
categories:
  - Linux
  - Tool
  - Vi
tags:
  - Linux
  - Tool
  - Vi
  - Diff
  - Tags
  - Quickfix
date: 2021-11-10 15:46:55
updated: 2021-12-06 15:46:04
toc: true
mathjax: true
description: 
---

##	*QuickFix*

-	快速修复命令：加快 *编辑-编译-编辑* 循环
	-	可通过 `$ vim -q <FILENAME>` 读取保存在文件中的出错信息
	-	快速修复列表
		-	唯一标识：在 *Vim* 会话中保持不变
		-	列表号：在快速修复栈中加入超过 10 个列表时可能改变
		-	由 `:vimgrep`、`:grep`、`:helpgrep`、`:make` 等命令产生
	-	位置列表：窗口局部的快速修复列表
		-	与窗口相关联，每个窗口有独立位置列表，与快速修复列表独立
		-	窗口分割时新窗口得到位置列表的备份
		-	由 `:lvimgrep`、`:lgrep`、`:lhelpgrep`、`:lmake` 等命令产生

###	跳转

-	快速修复命令
	-	`:[NR]cc[!] [NlR]`：跳转编号 `NR` 错误
		-	`NR`：错误编号，缺省首个或当前编号
	-	`:[COUNT]cn[ext][!]`、`:[COUNT]cN[ext][!]`/`:[COUNT]cp[revious][!]`：显示列表中后、前 `COUNT` 个错误
	-	`:[COUNT]cabo[ve]`、`:[COUNT]cbel[ow]`：当前缓冲区当前行前、后 `COUNT` 个错误
	-	`:[COUNT]cbe[fore]`、`:[COUNT]caf[ter]`：当前缓冲区光标前、后 `COUNT` 个错误
	-	`:[COUNT]cnf[ile][!]`、`:[COUNT]cNf[ile][!]`/`:[COUNT]cpf[ile][!]`：若列表包含文件名，显示后、前 `COUNT` 个文件的最后错误；否则后、前 `COUNT` 个错误
	-	`:cr[ewind][!] [NR]`/`:cfir[st][!] [NR]`：显示错误 `NR`
		-	`NR`：缺省显示首个错误
	-	`:cla[st][!] [NR]`：显示错误 `NR`
		-	`NR`：缺省显示末尾错误

-	位置列表命令：将快速修复命令前缀 `c` 换为 `l` 即可， 命令将使用当前窗口位置列表而不是快速修复列表
	-	`:[NR]ll[!] [NR]`：显示错误 `NR`
		-	`NR`：缺省显示相同错误
	-	`:[COUNT]ln[ext][!]`、`:[COUNT]lN[ext][!]`/`:[COUNT]lp[revious][!]`：显示列表中后、前 `COUNT` 个错误
	-	`:[COUNT]labo[ve]`、`:[COUNT]lbel[ow]`：当前缓冲区当前行前、后 `COUNT` 个错误
	-	`:[COUNT]lbe[fore]`、`:[COUNT]laf[ter]`：当前缓冲区光标前、后 `COUNT` 个错误
	-	`:[COUNT]lnf[ile][!]`、`:[COUNT]lNf[ile][!]`/`:[COUNT]lpf[ile][!]`：若列表包含文件名，显示后、前 `COUNT` 个文件的最后错误；否则后、前 `COUNT` 个错误
	-	`:lr[ewind][!] [NR]`/`:lfir[st][!] [NR]`：显示错误 `NR`
		-	`NR`：缺省显示首个错误
	-	`:lla[st][!] [NR]`：显示错误 `NR`
		-	`NR`：缺省显示末尾错误

-	通用选项
	-	`!`：强制跳转缓冲区，即使可能丢失当前缓冲区的修改

###	建立、恢复列表

-	快速修复命令
	-	`:cf[file][!] [ERRORFILE]`、`:cg[etfile][!] [ERRORFILE]`：读入错误文件，跳转、不跳转到首个错误
	-	`:caddf[ile][!] [ERRORFILE]`：读取错误文件，将错误文件里的错误加入列表中
	-	`:cb[uffer][!] [BUFNR]`、`:cgetb[uffer][!] [BUFNR]`：从缓冲区 `BUFNR` 读入错误列表，跳转、不跳转到首个错误
	-	`:cad[dbuffer] [BUFNR]`：从缓冲区 `BUFNR` 读取错误列表加入列表
	-	`:cex[pr][!] <EXPR>`、`:cgete[xpr] <EXPR>`：用 `EXPR` 计算结果建立列表，跳转、不跳转到首个错误
	-	`:cad[dexpr] <EXPR>`：将 `EXPR` 计算结果读入列表
	-	`:cl[ist][!] [<FROM> [TO]] [+<COUNT]`：显示有效错误
		-	`!`：所有错误
		-	`FROM-TO`：指定错误范围
		-	`+COUNT`：当前和之后 `COUNT` 个错误行
	-	`:[N]cq[uit][!] [N]`：以错误码 `N` 退出
		-	可用于另一个程序调用 *Vim* 的场合

-	位置列表命令：将快速修复命令前缀 `c` 换为 `l` 即可， 命令将使用当前窗口位置列表而不是快速修复列表
	-	`:lf[file][!] [ERRORFILE]`、`:lg[etfile][!] [ERRORFILE]`：读入错误文件，跳转、不跳转到首个错误
	-	`:laddf[ile][!] [ERRORFILE]`：读取错误文件，将错误文件里的错误加入列表中
	-	`:lb[uffer][!] [BUFNR]`、`:lgetb[uffer][!] [BUFNR]`：从缓冲区 `BUFNR` 读入错误列表，跳转、不跳转到首个错误
	-	`:lad[dbuffer] [BUFNR]`：从缓冲区 `BUFNR` 读取错误列表加入列表
	-	`:lex[pr][!] <EXPR>`、`:lgete[xpr] <EXPR>`：用 `EXPR` 计算结果建立列表，跳转、不跳转到首个错误
	-	`:lad[dexpr] <EXPR>`：将 `EXPR` 计算结果读入列表
	-	`:ll[ist][!] [<FROM> [TO]] [+<COUNT]`：显示有效错误
		-	`!`：所有错误
		-	`FROM-TO`：指定错误范围
		-	`+COUNT`：当前和之后 `COUNT` 个错误行

-	通用选项
	-	`!`：强制跳转缓冲区，即使可能丢失当前缓冲区的修改
	-	`ERRORFILE`：缺省为 `errorfile` 选项值，若错误文件编码和 `encoding` 选项不同，可用 `makeencoding` 选项指定
	-	`RANGE`：列表范围
	-	`CMD`：*Ex* 命令，可用 `|` 连接多个命令

-	相关选项
	-	`errorformat`：错误信息格式
		-	即使缺省值可能也无法正常读取直接保存的 *Quickfix* 列表
	-	`errorfile`：错误文件缺省值
	-	`makeencoding`：指定错误晚饭我编码格式

###	列表处理

-	快速修复列表缓冲区处理
	-	`:[RANGE]cdo[!] <CMD>`：在列表的每个有效项目上执行 `CMD`
		-	工作方式类似
			```vimscript
			:cfirst
			:<CMD>
			:cnext
			:<CMD>
			```
		-	最后操作缓冲区成为当前缓冲区
	-	`:[RANGE]cfdo[!] <CMD>`：在列表的每个文件上执行 `CMD`
		-	工作方式类似
			```vimscript
			:cfirst
			:<CMD>
			:cnfile
			:<CMD>
			```
	-	`:CFilter[!] /<PTN>/`：从匹配 `PTN` 的选项创建新列表
		-	`!`：不匹配的选项

-	位置列表处理
	-	`:[RANGE]ldo[!] <CMD>`：在列表的每个有效项目上执行 `CMD`
	-	`:[RANGE]lfdo[!] <CMD>`：在列表的每个文件上执行 `CMD`
	-	`:LFilter[!] /<PTN>/`：从匹配 `PTN` 的选项创建新列表

-	通用选项
	-	`!`：强制跳转缓冲区，即使可能丢失当前缓冲区的修改
	-	`RANGE`：列表范围
	-	`CMD`：*Ex* 命令，可用 `|` 连接多个命令

> - 过滤命令位于可选插件 `cfilter` 中，需要手动 `packadd` 中

###	多个列表切换

-	快速修复列表列表
	-	`:col[der] [COUNT]`、`:cnew[er] [COUNT]`：切换至前、后 `COUNT` 个列表
	-	`:[COUNT]chi[story]`：给出列表列表
		-	`COUNT`：第 `COUNT` 个列表成为当前列表

-	位置列表列表
	-	`:lol[der] [COUNT]`、`:lnew[er] [COUNT]`：切换至前、后 `COUNT` 个列表
	-	`:[COUNT]lhi[story]`：给出列表列表
		-	`COUNT`：第 `COUNT` 个列表成为当前列表

###	*Quickfix* 窗口

-	快速修复列表窗口命令
	-	`:cope[n] [HEIGHT]`：打开窗口显示当前列表
	-	`:ccl[ose]`：关闭窗口
	-	`:cw[indow] [HEIGHT]`：存在可识别错误时，打开窗口
		-	若窗口已打开、且没有可识别错误，窗口关闭
	-	`:cbo[ttom]`：光标置于窗口末行
		-	可用于异步加入错误

-	位置列表窗口命令
	-	`:lope[n] [HEIGHT]`：打开窗口显示当前列表
	-	`:lcl[ose]`：关闭窗口
	-	`:lw[indow] [HEIGHT]`：存在可识别错误时，打开窗口
		-	若窗口已打开、且没有可识别错误，窗口关闭
	-	`:lbo[ttom]`：光标置于窗口末行
		-	可用于异步加入错误

-	通用选项
	-	`HEIGHT`：窗口高度，缺省 10 行高

##	*Tags*

-	*Tags*：出现在 *tags* 文件中、用于跳转的标识符
	-	*tags* 文件由 `ctags` 类似的程序生成
	-	标签栈：记录跳转过的标签历史
		-	最多容纳 20 项，较早项目被前移直至移除

###	标签跳转

-	标签、标签栈跳转命令
	-	`:[COUNT]ta[g][!] [NAME]`：根据 *tags* 文件信息，跳转到 `NAME` 定义处
		-	`NAME`：可为正则表达式，将被置于标签栈；缺失则表示在标签栈中跳转
		-	`COUNT`：跳转至第 `COUNT` 个匹配（若有多个），缺省首个
	-	`g<LeftMouse>`/`<C-LeftMouse>`/`CTRL-]`：跳转至贯光标所在关键字定义第 `COUNT` 个匹配
		-	`v_CTRL-]`：跳转至至高亮文本定义
	-	`:tags`：显示标签栈内容
		-	`>`：标识当前激活项目
		-	`T0` 列：标签匹配输目
	-	`:[COUNT]po[p][!]`：跳转至标签栈上第 `COUNT` 个较早项目
	-	`g<RightMouse>`/`<C-RightMouse>`/`CTRL-t`：跳转至标签栈中第 `COUNT` 个较早项目

-	通用命令选项
	-	`!`：强制跳转，即使可能丢失当前缓冲区更改

-	标签跳转相关选项
	-	`tagcase`：标签大小写匹配设置
		-	`followic`：同 `ignorecase` 选项
		-	`followscs`：同 `smartcase` 选项
		-	`match`：严格大小写匹配
		-	`smart`：包含大写字符则不忽略大小写
	-	`tagstack`：标签被压入栈中

###	标签匹配列表

-	标签匹配列表命令
	-	`:ts[elect][!] [NAME]`：列出匹配 `NAME` 的标签
		-	`NAME`：可为正则表达式，缺省使用标签栈上最后的标签
	-	`:sts[elect][!] [NAME]`：`:tselect[!] [NAME]` 并分隔窗口显示选择的标签
	-	`g]`：类似 `CTRL-]`，但使用 `:tselect`
		-	`v_g]`：使用高亮文本激活 `g]`
	-	`:tj[ump][!] [NAME]`：类似 `:tselect`，若只有一个匹配，直接跳转
	-	`:stj[ump][!] [NAME]`：`:tjump[!] [NAME]` 并分隔窗口显示选择的标签
	-	`g CTRL-]`：类似 `CTRL-]`，但使用 `:tjump`
		-	`v_g CTRL-]`：使用高亮文本激活 `g CTRL-]`
	-	`:[COUNT]tn[ext][!]`：跳转至 `COUNT` 个后匹配的标签
	-	`:[COUNT]tp[revious][!]`/`:[COUNT]tN[ext][!]`：跳转至 `COUNT` 个前匹配的标签
	-	`:[COUNT]tr[ewind][!]`/`:[COUNT]tf[irst][!]`：跳转至第 `COUNT` 个匹配标签
	-	`:tl[ast][!]`：跳转至最后匹配标签

-	位置标签匹配列表
	-	`:lt[ag][!] [NAME]`：跳转至 `NAME` 首个匹，并将标签匹配列表添加至窗口位置列表中

-	快速修复标签匹配列表：以上命令前添加前缀 `p` 即将标签匹配结构加入快速修复列表中
	-	`:pts[elect][!] [NAME]`：列出匹配 `NAME` 的标签
	-	`:ptj[ump][!] [NAME]`：类似 `:tselect`，若只有一个匹配，直接跳转
	-	`:stj[ump][!] [NAME]`：`:tjump[!] [NAME]` 并分隔窗口显示选择的标签
	-	`:[COUNT]ptn[ext][!]`：跳转至 `COUNT` 个后匹配的标签
	-	`:[COUNT]ptp[revious][!]`/`:[COUNT]ptN[ext][!]`：跳转至 `COUNT` 个前匹配的标签
	-	`:[COUNT]ptr[ewind][!]`/`:[COUNT]ptf[irst][!]`：跳转至第 `COUNT` 个匹配标签
	-	`:ptl[ast][!]`：跳转至最后匹配标签

-	通用命令选项
	-	`!`：强制跳转，即使可能丢失当前缓冲区更改

##	*Diff*

-	窗口比较
	-	比较基于缓冲区内容、局限于当前 tab 内
	-	每个被比较的文件中，以下选项被设置（编辑其他值时，选项被重设为全局值）
		-	`diff`
		-	`scrollbind`
		-	`cursorbind`
		-	`scrollopt`：包含 "hor"
		-	`wrap`：关闭，`diffopt` 包含 "followrap" 时保持不变
		-	`foldmethod`："diff"
		-	`foldcolumn`：来自 `diffopt` 的值，缺省为 2

-	进入、设置比较模式命令
	-	`:diffs[plit] [FILE]`：上下分割窗口，并设置比较状态
	-	`:difft[his]`：将当前窗口设置为比较状态
	-	`:diffo[ff][!]`：关闭当前窗口比较状态
		-	`!`：关闭所有窗口比较状态
	-	`:diffu[pdate][!]`：更新当前比较窗口，重新生成比较信息（如：修改当前文件后）
		-	`!`：重新载入文件后更新比较窗口（外部修改文件）

-	比较模式中命令
	-	`:diffg[et] [BUF_NO]`：从其他比较窗口中更新差异
		-	`:%diffget`：获取整个文件差异
		-	多个缓冲区比较时，需要指定基准缓冲区
	-	`:diffpu[t] [BUF_NO]`：向其他比较窗口中更新差异
	-	`]c`、`[c`：比较窗口中移至下个不同的为止
	-	`do`、`dp`：向、从其他窗口更新差异

-	配置选项
	-	`diff`：加入窗口至比较窗口组
	-	`diffopt`：比较选项
	-	`diffexpr`：计算文件不同版本差异文件的表达式

> - `$ vimdiff`、`$ vimdiff -d` 可直接在命令行开启比较模式
> - 比较状态窗口自动参与当前 tab 内比较，其左侧多出空白列




