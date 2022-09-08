---
title: 
categories:
  - 
tags:
  - 
date: 2022-07-13 12:45:03
updated: 2022-09-08 10:58:43
toc: true
mathjax: true
description: 
---

##	*Clash*

-	*Clash*：*Go* 开发的多平台代理工具
	-	协议支持
		-	*SS/SSR*
		-	*v2ray*
		-	*Trojan*
	-	平台支持
		-	*Windows*：<https://github.com/Fndroid/clash_for_windows_pkg>
		-	*ClashX*：<https://github.com/yichengchen/clashX/releases>
		-	*ClashForAndroid*：<https://github.com/Kr328/ClashForAndroid/releases/>

-	*Clash* 使用方式
	-	仅能通过读取配置文件配置、启动
	-	通过 *RESTful API* 管理、控制
		-	各节点情况
		-	节点选择

###	*Clash Yaml* 配置文件

```yaml
port:						# HTTP 端口
socks-port:					# SOCK5 端口
redir-port:					# redir 端口
allow-lan:
mode: Rule
proxies:
-
  name:
proxy-groups:
-
  name:
  type:
  proxies:
  -
Rule:
- 
```

-	关键字段说明
	-	`mode`：代理规则
	-	`external-controller`：*RESTful API* 控制地址
	-	`proxies`：代理节点列表
		-	其中元素字段
			-	`name`：代理节点名称，将作为 `proxy-groups` 引用键
		-	一般，不直接用于设置代理
	-	`proxy-groups`：代理组列表
		-	用于设置代理
		-	其中元素字段
			-	`name`：代理组名称，将作为 `Rule`、其他代理组引用键
			-	`type`：代理组类型，包括 `url-test`、`select` 等
				-	`url-test`：根据延迟自动选择节点
				-	`select`：手动选择节点，缺省（未通过 *RESTful API* 选择）首个
			-	`proxies`：代理，包括代理节点、代理组、`REJECT`、`DIRECT`
		-	`proxy-groups` 元素可以相互引用
	-	`Rule`：`Rule` 模式下的代理规则列表
		-	其中元素遵循 `<RULE_TYPE>, <TARGET>, <RULE>` 结构
			-	`RULE_TYPE`：`DOMAIN`、`DOMAIN-SUBFFIX`、`DOMAIN-KEYWORD`、`IP-CIDR`、`GEOIP`
			-	`TARGET`：代理目标，即 *IP*、域名
			-	`RULE`：采取的规则，即代理组、`REJECT`、`DIRECT` 等
		-	`Rule` 中 `RULE` 字段决定各地址的代理规则
			-	即在配置中包含流量分发职能，而不仅有节点选择职能

-	有些工具可以帮助不同工具间配置文件的转换
	-	*convert2clash*：<https://github.com/waited33/convert2clash>
	-	*subconverter*：<https://github.com/tindy2013/subconverter>
	-	*sub-web*：<https://github.com/CareyWang/sub-web>
	-	*sub-web* 在线服务：<https://sub-web.netlify.app/>


> - *Clash* 官方配置：<https://github.com/Dreamacro/clash/wiki/configuration>
> - *Clash* 配置说明：<https://v2xtls.org/%E6%B7%B1%E5%85%A5%E7%90%86%E8%A7%A3clash%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6/>

###	*Clash RESTful API*

|*URL*|描述|*Query*|*Body*|说明|
|-----|-----|-----|-----|-----|
|`GET /traffic`|获取实时流量| | | |
|`GET /logs`|获取实时日志| | | |
|`GET /proxies`|获取全部代理| | | |
|`GET /proxies/:name`|获取指定代理| | |由地址中 `name` 指定|
|`GET /proxies/:name/delay`|测试代理延迟|`timeout`、`url`| | |
|`PUT /proxies/:name`|切换代理| |`name`|地址中 `name` 一般为 `select` 类代理组|
|`GET /configs`|获取当前基础设置| | | |
|`PATCH /configs`|修改当前设置| |`port`、`socks-port`、...| |
|`PUT /configs`|重新加载配置文件|`force`|`path`|不影响 `external-controller`、`secret` 值|
|`GET /rules`|获取已解析规则| | | |

> - *Clash RESTful API* 官方: <https://github.com/Dreamacro/clash/wiki/external-controller-API-reference>
> - *Clash RESTful API* 说明<https://clash.gitbook.io/doc/restful-api/proxies#qie-huan-selector-zhong-xuan-zhong-de-dai-li>

