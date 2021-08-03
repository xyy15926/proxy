---
title: NPM 总述
categories:
  - Web
  - NPM
tags:
  - Web
  - NPM
date: 2021-08-03 10:32:45
updated: 2021-08-03 14:59:56
toc: true
mathjax: true
description: NPM 包管理系统总述
---

##	`npm`

-	`npm` 的三个独立组成部分
	-	网站：查找包、设置参数、管理 `npm` 使用体验的主要途径
	-	注册表：存储包信息
	-	*CLI*：终端应用

###	`npm` 包管理

-	`npm` 包可以分为是否全局安装
	-	全局安装：适合安装命令行工具包
		-	位于 `/user/local` 或 *Node.js* 安装目录
	-	局部安装（缺省）：适合安装包依赖，且包通过 *Node.js* 的 `require` 加载
		-	位于当前目录 `node_modules` 目录下
	-	全局安装和局部安装互相独立
		-	若同时需要命令行、包依赖，则应分别安装或使用 `npm` 链接
	-	为避免污染全局环境，以下方式可以用于局部安装命令行
		-	`npx` 包（命令）：查找 `noode_modules` 中局部安装包
		-	`alias` 设置别名：添加 `PATH=<bin-dir>:$PATH <cmd>` 别名，即每次为命令执行设置环境变量

> - <https://docs.npmjs.com/cli/v7/commands/npm>
> - <https://www.npmjs.cn/cli/npm/>

####	输入命令

-	`install`：安装
	-	`-g`：全局安装
	-	`--save`：维护 `package.json` 中依赖项
	-	`--save-dev`：维护 `package.json` 中开发依赖项

-	`uninstall`：卸载
	-	`-g`：卸载全局安装包
	-	`--save`：维护 `package.json` 中依赖项
	-	`--save-dev`：维护 `package.json` 中开发依赖项

-	`update`：更新
	-	`-g`：更新全局安装包

-	`outdated`：检查版本
	-	`-g`：检查全局安装包
	-	`--depth=<num>`：检查深度

####	输出命令

-	`whoami`：
-	`publish`：发布包

###	`npm` 配置

-	`config`：更新、修改用户或全局 `npmrc` 文件

##	`npm` 配置文件

###	`npm` 用户配置文件

####	`.npmrc`

```cnf
repository=<repo-URL>
init.author.email=
init.author.name=
init.license=
```

-	`.npmrc`：`npm` 用户配置文件，缺省为 `~/.npmrc`
	-	指定 `npm` 本身配置：包仓库地址、用户信息

> - <https://www.npmjs.cn/files/npmrc/>

####	`.npm-init.js`

```js
// 直接设置键值对
module.exports = {
	"<custom-field>": "<field-value>",
	"<custom-field>": "<field-value>",
}
// 通过 `prompts` 函数交互式设置键值对
module.exports = prompts("<Question 1>, "<Field>")
)
```

-	`.npm-init.js`：用户包初始化配置文件，缺省为 `~/.npm-init.js`
	-	设置 `package.json` 生成内容

###	环境变量

-	`NPM_CONFIG_PREFIX`：全局包安装地址

##	`npm` 包配置文件

###	包配置文件

####	`package.json`

```json
{
	"name": "<package-name>",
	"version": "<semantic-version>",
	"description"："",
	"main": "index.js",
	"scripts": {
		"tests": "echo \"Error:\" && exit 1"
	},
	"repository": {
		"type": "git",
		"url": "<URL>"
	},
	"keywords": [ ],
	"author": "<author-name>",
	"license": "<license-name>",
	"bugs": {
		"url": "<URL>"
	},
	"homepage": "<URL>"
	// 以上为 `npm init` 默认生成内容
	"dependencies": {
		"<dep-named>": "<dep-version>",
		"<dep-named>": "<dep-version>"
	},
	"devDependecies": {
		"<dev-only-dep-named>": "<dep-version>",
		"<dev-only-dep-named>": "<dep-version>"
	}
}
```

-	`package.json`：局部包管理文件，位于当前包目录
	-	列出包依赖
	-	**语义化** 管理包版本
	-	方便迁移

-	创建：`npm init [--yes]` 初始化包即默认生成 `package.json`
	-	包含字段可通过 `.npm-init.js` 设置
	-	字段值大部分为空，除非可从 `npm` 用户配置文件 `init` 字段中获取

-	字段说明
	-	`name`：可用 `@<scope>/` 为包设置域名，方便组织相关包
	-	`version`：应遵守语义化版本规则
	-	`dependencies`：包依赖，安装、卸载时 `--save` 标志会自动维护
	-	`devDependencies`：开发时包依赖，安装、卸载时 `--save-dev` 标志会自动维护














