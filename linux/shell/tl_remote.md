---
title: Linux 远程工具
categories:
  - Linux
  - Tool
tags:
  - Linux
  - Tool
  - Remote
  - Rsync
  - SSH
date: 2021-07-29 21:32:17
updated: 2021-10-26 19:49:44
toc: true
mathjax: true
description: 
---

##	*OpenSSH*

-	*OpenSSH*：最流行的 *SSH* 实现
	-	`openssh`：客户端包名，可通过包管理器直接安装
	-	`openssh-server`：服务器端包名

> - <https://xdev.in/posts/understanding-of-ssh/>

###	`/usr/bin/ssh`

-	`ssh`：*OpenSSH* 远程客户端程序
	-	`$ ssh <USER>@<HOST>:<PORT> <OP>`：登录远程服务器
		-	`USER`：缺省为当前用户名（`@` 需省略）
		-	`PORT`：端口
		-	`OP`：非交互登录 *SSH* 时指定需要执行的命令
	-	`$ ssh -Q <ALG>`：查看 `ssh` 支持的算法版本，`ALG` 为算法类型
		-	`cipher`：（全部）数据加密算法（对称加密算法）
		-	`cipher-auth`：（已启用）数据加密算法
		-	`mac`：完整性校验算法
		-	`kex`：密钥交换算法
		-	`key`：数据签名、认证算法（非对称加密算法）

-	连接选项参数
	-	`-C`：数据压缩传输
	-	`-c <EALG>`：指定数据加密算法（逗号分隔多个）
	-	`-m <MALG>`：指定数据完整性校验算法
	-	`-i <PRIVATE>`：指定使用的私钥
	-	`-l <USER>`：指定的登录用户名（等同于 `<USER>@<HOST>` 写法）
	-	`-p <PORT>`：指定端口，缺省 22
	-	`-o "<KEY> <VALUE>"`/`-o <KEY>=<VALUE>`：指定参数
		-	`-o "User root"`/`-o User=root`：等价于 `-l root`
	-	`-f`：后台运行 *SSH* 连接
	-	`-t`：*SSH* 执行远程命令时，提供交互 Shell
	-	`-q`：安静模式，不输出警告信息
	-	`-v`：显示详细信息
		-	可以类似 `-vv`/`-v -v` 重复多次，表示信息的详细程度
	-	`-T`：测试连接
	-	`-F <CONFIG>`：指定配置文件，缺省 `~/.ssh/config`
		-	为覆盖其余配置，可以指定 `-F /dev/null`
	-	`-1`、`-2`：指定使用 *SSH1*、*SSH2* 协议
	-	`-4`、`-6`：指定使用 *IPv4*、*IPv6* 协议

-	转发选项参数
	-	`-L <SELF_ADDR>:<SELF_PORT>:<TGT_ADDR>:<TGT_PORT>`：本地端口转发
	-	`-R <SSHD_ADDR>:<SSHD_PORT>:<TGT_ADDR>:<TGT_PORT>`：远程端口转发
	-	`-R <SELF_PORT>`：动态端口转发
	-	`-X`：打开图像窗口转发
	-	`-N`：*SSH* 连接仅用于端口转发，不能远程执行命令

####	密钥认证

-	*SSH* 密钥认证
	-	前置依赖
		-	客户端已生成密钥（服务器端必然有密钥）
		-	客户端公钥已经被服务器端承认
	-	*SSH* 密钥可以通过 `ssh-agent` 管理，避免每次使用均需输入密钥口令

-	*SSH* 密钥认证流程
	-	客户端向服务器发起公钥认证请求，发送 *RSA* 公钥的指纹作为标识符
	-	服务器端检查（`authorized_keys` 中）是否存在请求账号的公钥、访问权限
		-	若公钥不存在则断开连接
		-	存在，服务器端使用公钥对随机串加密，发送给客户端
	-	客户端收到数据后
		-	用私钥解密
		-	结合 session-id 生成 *MD5* 摘要（避免重放攻击），发送给服务器端
	-	服务器端同样方式生成摘要，与客户端返回结果比较

####	证书认证

-	*SSH* 证书认证
	-	前置依赖
		-	客户端、服务器端均已安装 *可信任CA* 证书（公钥）
			-	客户端安装 *CA* 签发服务器证书的公钥：在 `ssh_known_hosts` 中指定
			-	服务器端安装 *CA* 签发客户端证书的公钥：`/etc/ssh/sshd_config` 中 `TrustedUserCAKeys` 指定
		-	客户端、服务器端用于 *SSH* 通信公钥均已用受信任证书签名得到证书、并被安装
			-	客户端安装自身公钥证书：`<KEY_NAME>-cert.pub` 与密钥置于同一个目录即可
			-	服务器端安装自身公钥证书：`/etc/ssh/sshd_config` 中 `HostCertificate` 指定

-	*SSH* 证书认证失效
	-	客户端失效 *CA* （证书）公钥：删除 `known_hosts` 中对应 *CA* 证书行
	-	服务器端失效客户端证书、公钥：在 *Key Revocation List*/*KRL* 文件中指定
		-	*KRL* 由 `/etc/ssh/ssh_config` 中 `RevokedKeys` 指定，缺省为 `/etc/ssh/revoked_keys` 
		-	*KRL* 内容可为公钥、*KRL* 限定项（摘要、密钥 ID、证书序列号等）
		-	可通过 `$ ssh-keygen -k` 选项生成 *KRL* 文件

		```cnf
		$ ssh-keygen -kf /etc/ssh/revoked_keys -z 1 user_key.pub
		```

> - 没有为 *SSH* 颁发证书的 *CA*，所以一般都是使用自签发证书

####	端口转发

-	本地端口转发：将本地端口 `SELF_PORT` 通过 *SSH* 服务器端转发到目标主机 `TGT_ADDR:TGT_PORT`
	-	要求 *SSH* 服务器可与目标主机联通
	-	本机地址可省略，表示监听本机所有地址

-	远程端口转发：将 *SSH* 服务器（远程） `SSHD_ADDR` 上 `SSHD_PORT` 端口通过本机转发到目标主机 `TGT_ADDR:TGT_PORT`
	-	配置要求
		-	`sshd_config` 中 `AllowTcpForwarding` 选项需打开
		-	默认转发到远程主机的端口绑定在 `127.0.0.1`，绑定至 `0.0.0.0` 需要打开 `sshd_config` 中 `GatewayPorts`选项，或者通过本地端口转发绑定至 `0.0.0.0`
	-	要求本机可以目标主机联通
	-	目标主机地址可省略

-	动态端口转发：将本地端口的 `SELF_PORT` 通过 *SSH* 服务器转发至任意目标主机
	-	即将 *SSH* 服务器作为 *SOCKS* 代理

> - <https://jeremyxu2010.github.io/2018/12/ssh%E7%9A%84%E4%B8%89%E7%A7%8D%E7%AB%AF%E5%8F%A3%E8%BD%AC%E5%8F%91/>

###	`/usr/bin/ssh-keygen`

-	`ssh-keygen`：生成 *SSH* 密钥工具
	-	`$ ssh-keygen <OPS>`：生成密钥
		-	可以设置口令保护、指定密钥储存位置
		-	口令保护的私钥每次使用都需要输入密码，可用 `ssh-agent` 可以用于统一管理私钥密码
	-	`$ ssh-keygen -I <CERT_ID> -s <CA_KEY> <OPS> <PUB_KEY>`：根据 `PUB_KEY` 签发证书
	-	`$ ssh-keygen -L [-f <KEY_FILE>]`：查看证书信息

####	创建密钥

```shell
$ ssh-keygen -f /etc/ssh/ssh_host_rsa_key -b 4096 -t rsa
```

-	创建密钥选项参数
	-	`-t [dsa|ecdsa|ecdsa-sk|ed25519|ed25519-sk|ras]`：密钥算法
	-	`-b <INT>`：密钥强度（位数）
	-	`-C <STR>`：指定密钥注释
	-	`-f <PATH>`：指定生成的密钥文件名（公钥加上 `.pub`）
	-	`-N <PWD>`：私钥口令
	-	`-p`：（交互式）更改私钥口令
	-	`-F <HOST>`：检查主机 `HOST` 是否在 `known_hosts` 内
	-	`-R <HOST>`：将主机 `HOST` 公钥指纹移除出 `know_hosts`

####	签发证书

```sh
 # 签发服务器端证书
$ ssh-keygen -s host_ca -I host.example.com -h -n host.example.com -V +52w ssh_host_rsa_key.pub
 # 查看证书公开信息
$ ssh-keygen -L -f ssh_host_rsa_key-cert.pub
 # 生成、更新 KRL 文件
$ ssh-keygen -kf /etc/ssh/revoked_keys -z 1 user_key.pub
$ ssh-keygen -ukf /etc/ssh/revoked_keys -z 1 user_key.pub
```

-	*Certificate Authority* 签发证书选项
	-	`-I <CERT_ID>`：证书身份字符串，用于区分、撤销证书
	-	`-s <CA_KEY>`：*CA* 签发证书所用私钥
	-	`-h`：标识证书为服务器端证书
	-	`-n <HOST>`：指定服务器
	-	`-V <TIME>`：证书有效期
		-	`+52w`：52 周
		-	`+1d`：1 天
	-	`-L`：查看证书公开信息
	-	`-f`：指定证书文件
	-	`-z <SERIAL_NO>`：签发证书的序列号，方便之后管理证书
		-	缺省为 0
		-	可用 `+<SERIAL_NO>` 指定同一条命令中序列号递增

-	客户端通过 *Key Revocation List* 失效证书选项
	-	`-k`：操作（默认重新生成） *KRL* 文件
	-	`-f`：指定 *KRL* 文件名，缺省为 `/etc/ssh/revoked_keys`
	-	`-u`：更新指定 *KRL* 文件
	-	`-s <CA_KEY>`：*CA* 签发证书对应 **公钥**
	-	`-z <VERSION_NO>`：指定失效版本号，即位于 *KRL* 文件的行号

###	`/usr/bin/ssh-copy-id`

-	`ssh-copy-id`：添加公钥到远程服务器的 `~/.ssh/authorized_keys` 文件中
	-	`$ ssh-copy-id -i <FILE> <USER>@<HOST>`：将 `FILE` 添加至远程服务器中
	-	务必保证 `authorized_keys` 以换行结尾，否则公钥将连在一起

-	选项参数
	-	`-i`：指定公钥文件

###	`/usr/bin/ssh-add`、`/usr/bin/ssh-agent`

-	`ssh-add`：管理 `ssh-agent` 中代理密钥
	-	`$ ssh-add <KEY_FILE>`：将私钥添加至 `ssh-agent`，缺省添加 `~/.ssh/id_rsa`
	-	`ssh-add` 选项参数
		-	`-d <KEY_FILE>`：删除指定密钥
		-	`-D`：删除所有已添加密钥
		-	`-l`：列出已添加密钥
		-	`-L`：列出代理中私钥对应公钥
		-	`-x/-X`：锁定/解锁 `ssh-agent`（需要设置密码）
		-	`-t <SECONDS>`：设置密钥时限

-	`ssh-agent`：（在一次 Session 中）控制、保存公钥验证所使用的私钥
	-	`ssh-agent` 每次使用均需启动、添加私钥，适合需将频繁使用私钥场合
		-	`$ ssh-agent <SHELL>`：创建子 Shell 并在其中启动 `ssh-agent`
		-	<code>$ eval `ssh-agent`</code>：在当前 Shell 启动 `ssh-agent` 服务
		-	`ssh-add` 添加私钥需要输入私钥密码（无密码私钥无管理必要）
	-	选项参数
		-	`-k`：关闭当前`ssh-agent`进程

###	`/usr/bin/scp`

-	`scp`：*secure copy*，在主机间加密安全传输文件
	-	`$ scp <SRC> <DEST>`：将 `SRC` 复制到 `DEST` 中
		-	`SRC`、`DEST` 中可以包含形如 `user@host` 的主机名，与文件名之间 `:` 分隔，省略时表示当前用户、当前主机
		-	本机到远程：`$ scp /path/to/file user_name@host:/path/to/dest`
		-	远程到本机：`$ scp user_name@host:/path/to/file /path/to/dest`

-	参数选项
	-	`-r`：递归复制（目录）
	-	`-c <CIPHER>`：指定加密算法
	-	`-C`：压缩数据
	-	`-F <CONF>`：指定 *SSH* 连接配置文件
	-	`-i`：指定密钥
	-	`-l <Kbit/sec>`：限制传输数据带宽
	-	`-p`：保留文件元信息，如：修改时间、访问时间、文件状态
	-	`-P`：指定远程主机 *SSH* 端口
	-	`-q`：关闭进度条
	-	`-v`：详细输出

###	`/usr/bin/sshd`

-	`sshd`：*SSH* 服务器端启动程序
	-	直接启动 `/usr/bin/sshd`：（应用要求）必须使用完整路径启动
	-	通过 `systemctl`（或老式 `init.d`）设置服务

-	参数选项
	-	`-t`：检查配置文件正确性
	-	`-d`：显示 *Debug* 信息
	-	`-D`：`sshd` 不作为后台守护进程运行
	-	`-e`：将 `sshd` 写入系统日志的内容导向标准错误
	-	`-f <FILE>`：指定配置文件位置
	-	`-h <KEY_FILE>`：指定服务器端口令交换密钥
	-	`-o "<KEY> <VALUE>"`：指定的配置项键值对（支持 `=`、空格分隔），可多次使用指定多个配置项
	-	`-p`：指定监听端口

###	*SSH* 客户端配置文件

-	客户端全局配置文件
	-	`/etc/ssh/ssh_config`：全局配置文件
	-	`/etc/ssh/ssh_known_hosts`：全局已认可服务器公钥指纹
	-	`/etc/ssh/revoked_keys`：存储不被信任的用户公钥

-	客户端用户配置文件 `~/.ssh`
	-	`~/.ssh/config`：用户配置文件（优先级高于全局）
	-	`~/.ssh/id_ecdsa`、`~/.ssh/id_ecdsa.pub`：用户 *ECDSA* 私钥、公钥
	-	`~/.ssh/id_rsa`、`~/.ssh/id_rsa.pub`：用户 *SSH-2 RSA* 私钥、公钥
	-	`~/.ssh/identity`、`~/.ssh/indentity.pub`：用户 *SSH-1 RSA* 私钥、公钥
	-	`~/.ssh/known_hosts`：已认可 *SSH* 服务器公钥指纹

####	`~/.ssh/config`

```conf
Host <LINK_NAME>
	HostName <HOST>
	User <USER_NAME>
	Port <PORT>
	IdentityFile <PRIVATE_KEY>
	IdentitiesOnly = yes
```

-	内容说明
	-	`Host` 开启 *SSH* 连接配置块（至下个 `Host`），其后 `<LINK_NAME>` 表示连接名称
		-	连接名称可为非域名的别名，在命令行中替代整块配置，简化命令
		-	连接名称中可包含通配符 `*`，为满足表达式的主机（不是连接名）设置缺省值
	-	格式
		-	可用 `=` 或空格分隔选项键值对
		-	配置中所有缩进均不必须，仅表示视觉缩进

-	连接配置选项
	-	`HostName`：`Host` 后为别名时指定主机地址
	-	`User`
	-	`Port`
	-	`AddressFamily [inet|inet6]`：使用的 *IP* 协议
	-	`BindAddress`：指定使用的本机 *IP* 地址
	-	`CheckHostIP [yes|no]`：检查 *SSH* 服务服务器 *IP* 地址是否和公钥数据库吻合
	-	`Compression [yes|no]`：是否压缩传输信号
	-	`ConnectionAttempts <INT>`：最大尝试连接次数
	-	`ConnectionTimeout <INT>`：连接超时秒数
	-	`ServerAliveInterval <INT>`：建立连接 `INT` 秒后未收到服务器端消息则发送 `keepalive` 信号
	-	`ServerAliveCountMax <INT>`：发送 `keepalive` 信号 `INT` 次后无回应则断开连接
	-	`TCPKeepAlive [yes|no]`：是否定期发送 `keepalive` 信号
	-	`Protocol [1|2]`：支持的 *SSH* 协议版本（逗号分隔多个）
	-	`SendEnv <ENV>`：向服务器端发送环境变量，变量值从当前客户端环境中拷贝（空格分隔多个）
	-	`LogLevel [QUIET|DEBUG]`：日志详细程度

-	密钥、算法配置
	-	`IdentityFile`：私钥文件
	-	`HostCertificate`：服务器端 *CA* 证书文件
	-	`TrustedUserCAKeys`：信任的 *CA* 公钥
	-	`Ciphers [blowfish|3des|]`：指定加密算法
	-	`HostKeyAlgorithms [ssh-dss|ssh-rsa]`：密钥交换算法
	-	`MACs [hmac-sha1|hmac-md5]`：数据校验算法
	-	`PreferredAuthentications [publickey|hostbased|password]`：登录方法优先级
	-	`NumberOfPasswordAuthentication [no|yes]`：支持密码登录
	-	`PubKeyAuthentication [no|yes]`：支持密钥登录
	-	`UserKnownHostsFile`：当前用户公钥（指纹）存储文件
	-	`GlobalKnownHostsFile`：全局公钥（指纹）存储文件
	-	`VerifyHostKeyDNS [yes|no]`：检查 *SSH* 服务器 DNS 记录，确认指纹是否与存储记录一致
	-	`StrictHostKeyChecking [no|yes|ask]`：严格检查密钥
		-	`yes`：服务器公钥未知或改变则拒绝连接
		-	`no`：服务器公钥未知则加入，改变则输出警告后不改变公钥数据库并继续连接
		-	`ask`：询问

-	端口、转发、监听设置选项
	-	`LocalForward <SELF_ADDR>:<SELF_PORT> <TGT_ADDR>:<TGT_PORT>`：本地端口转发
	-	`RemoteForward <SSHD_ADDR>:<SSHD_PORT> <TGT_ADDR>:<TGT_PORT>`：远程端口转发
	-	`DynamicForward <SELF_PORT>`：动态端口转发

> - 此文件对所有使用 *OpenSSH* 的场合，包括 *Git*、`scp` 等

####	`id_rsa`、`id_rsa.pub`

```cnf
 # public key
 # KEY_ALG：密钥算法，常用 `ssh-rsa`、`ecdsa-sha2-nistp256`
<KEY_ALG> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX <USERNAME>@<HOST>

```

-	`~/.ssh/id_rsa`：*RSA* 密钥算法私钥默认地址
	-	`KEY_ALG`：密钥算法类型
	-	`USERNAME@HOST`：标识密钥属于不同主机、用户，可省略注释

```cnf
 # private key
-----BEGIN OPENSSH PRIVATE KEY-----
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
-----END OPENSSH PRIVATE KEY-----
```

-	`~/.ssh/id_rsa.pub`：*RSA* 密钥算法公钥默认地址

####	`known_hosts`

```cnf
 # 已认可 *CA* （证书）公钥
 # IP/ADDR：允许的 ip、域名，可用通配符
 # KEY_ALG：密钥算法，常用 `ssh-rsa`、`ecdsa-sha2-nistp256`
@cert-authority <IP/ADDR> <KEY_ALG> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 # 已认可服务器端公钥
<IP/ADDR> <HOST> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

-	`known_hosts`：存储已认可公钥内容（服务器端、*CA* 证书）
	-	*CA* （证书）公钥往往要自行维护（自颁发）
	-	一般的服务器端公钥在每次尝试登录服务器后都会自动记录，并在其改变时警告

###	*SSH* 服务器端配置文件

-	服务器端配置文件
	-	`/etc/ssh/sshd_config`：`sshd`（*SSH* 服务器端）配置文件
	-	`/etc/ssh/ssh_host_ecdsa_key`、`/etc/ssh/ssh_host_ecdsa_key.pub`：服务器端 *ECDSA* 私钥、公钥
	-	`/etc/ssh/ssh_host_rsa_key`、`/etc/ssh/ssh_host_rsa_key.pub`：*SSH 2* 协议版本服务器端 *RSA* 私钥、公钥
	-	`/etc/ssh/rsa_host_rsa_key-cert.pub`：服务器端（默认 *RSA*）公钥对应证书

	> - 服务器端密钥地址可在 `/etc/ssh/sshd_config` 中更改

-	服务器端用户配置文件
	-	`~/.ssh/authorized_keys`：已认可 *SSH* 客户端公钥
		-	权限需设置为 `644`，否则 *SSH* 服务器端将拒绝读取

####	`/etc/ssh/sshd_config`

```shell
HostKey /etc/ssh/ssh_host_rsa_key
PubkeyAuthentication yes
AuthorizedKeyFile .ssh/authorized_keys
```

-	格式说明
	-	每个配置项一行，`=`、空格分隔的键值对
	-	配置项大小写不敏感
	-	注释必须另起一行，不能直接更在配置项后

-	常规设置
	-	`AcceptEnv`：允许客户端通过 `SendEnv` 设置的环境变量，空格分隔
	-	`AllowGroups`：允许登录的用户组，空格分隔多个，缺省所有
	-	`DenyGroups`
	-	`AllowUsers`：允许登录的用户，空格分隔多个，缺省所有
		-	用户名支持通配符
		-	支持使用 `user@host` 限定主机
	-	`DenyUsers`
	-	`Compression`：数据压缩传输，默认 `yes`
	-	`Protocol`：指定使用的 *SSH* 协议，`1,2` 表示同事支持两个版本协议
	-	`X11Forwarding`：打开图形界面转发，缺省为 `no`

-	信息设置
	-	`Banner`：用户登陆后 `sshd` 展示的信息文件
	-	`PrintMotd`：用户登录展示系统 `/etc/motd` 文件
	-	`LogLevel`：日志粒度
	-	`PrintLastLog`：打印上次用户登录时间，缺省 `yes`
	-	`SyslogFacility`：*Syslog* 如何处理 `sshd` 日志，缺省为 `Auth`

-	监听、端口、转发设置
	-	`LoginGraceTime`：客户端登陆时无操作最长时间
	-	`MaxStartups`：允许的最大并发数量
		-	可设置为 `A:B:C` 格式，前 `A` 个连接正常，之后 `B%` 概率连接被拒绝，直至最多 `C` 个连接
	-	`ClientAliveCountMax`：建立连接后客户端失去响应后，服务器尝试连接上限
	-	`ClientAliveInterval`：客户端静默时间上限
	-	`TCPKeepAlive`：是否定期发送 `keepalive` 信号
	-	`AllowTcpForwarding`：允许端口转发
		-	`yes`：允许本地、远程端口转发
		-	`local`：允许本地端口转发
		-	`remote`：允许远程端口转发
	-	`ListenAddress`：`sshd` 监听的本机 *IP* 地址
		-	默认为 `0.0.0.0`，监听所有地址
		-	多行监听多个地址
	-	`Port`：`sshd` 监听端口，多行监听多个端口

-	密钥、算法设置选项
	-	`HostKey`：*SSH* 服务器端密钥
	-	`PasswordAuthentication`：允许密码登录，缺省 `yes`
	-	`PubKeyAuthentication`：允许公钥登录，缺省为 `yes`
	-	`RSAAuthentication`：允许 *RSA* 口令交换，缺省为 `yes`
	-	`PermitEmptyPasswords`：允许空密码登录，缺省 `no`
	-	`PermitRootLogin`：允许 root 用户登录，缺省 `yes`
		-	`prohibit-password`：root 用户仅允许密钥登录
	-	`PermitUserEnvironment`：是否加载 `~/.ssh/environment`、`~/.ssh/authorized_keys` 中 `environment` 环境变量设置
	-	`AuthorizedKeysFile`：存储用户公钥的目录，默认为 `.ssh/authorized_keys`
	-	`ChallengeResponseAuthentication`：使用“键盘交互”身份验证
		-	理论上可向用户询问多重问题，但实务中仅询问用户密码
	-	`Ciphers`：可接受的信息加密算法，逗号分隔多个算法，缺省 `3des-cbc`
	-	`MACs`：可接受的数据校验算法，逗号分隔多个算法，缺省 `hmac-sha1`
	-	`StrictModes`：要求用户 *SSH* 配置文件、密钥文件所在目录的所有者必须为 root、用户本人，组、其他写权限需关闭，缺省为 `yes`
	-	`UserLogin`：用户认证内部使用 `/usr/bin/login` 替代 *SSH*，缺省为 `no`

-	*SSH1* 配置项
	-	`KeyRegenerationInterval`：*SSH1* 版本密钥重新生成的时间间隔
	-	`QuietModes`：日志仅输出致命错误信息
	-	`ServerKeyBits`：*SSH 1* 密钥重新生成时的位数，缺省为 `768`

-	*SSH2* 配置项
	-	`VerboseMode`：日志输出详细 *Debug* 信息，缺省为 `yes`

####	`~/.ssh/authorized_keys`

```cnf
 # 已认可 *CA* (证书）公钥
 # USER_NAME：允许登录用户名
 # KEY_ALG：密钥算法，一般为 `ssh-rsa`
@cert-authority principals="<USER_NAME>" <KEY_ALG> XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 # 已认可客户端公钥
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

-	`~/.ssh/authorized_keys`：存放被认可的公钥内容（客户端、*CA* 证书）
	-	缺省为此文件，可在 `/etc/.ssh/sshd_config` 中配置
	-	文件须设置为仅 root 用户有权更改 `644`，否则报错
	-	被认可主机可使用私钥免口令登录

##	其他远程命令

###	`rsync`

-	`$ rsync <SRC> <DEST>`：将 `SRC` 拷贝至 `DEST` 目录，实现本地主机、远程主机的文本双向同步
	-	`SRC` 后跟 `/` 表示将其中内容复制到 `DEST` 内
	-	当 `SRC`、`DEST` 均为本机地址时，即类似 `cp`
	-	同步过程分两步
		-	确定源目录中需同步文件文件
			-	默认情况下，`rsync` 仅检查源文件、目标文件大小、*mtime* 决定是否需同步
			-	可以通过设置选项参数改变检查模式
		-	确定目标目录中文件行为
			-	删除目标主机上比源主机多的文件
			-	先备份已经存在的目标文件
			-	追踪链接文件

> - <https://wangdoc.com/ssh/rsync.html>
> - <https://www.cnblogs.com/f-ck-need-u/p/7220009.html>

####	选项参数

-	基本选项参数
	-	`-r`/`--recursize`：递归同步，类似 `cp` 命令
	-	`-a`/`--archive`：归档模式，递归传输并保持文件元信息
		-	软链接也会被同步
	-	`-n`/`--dry-run`：模拟执行结果，配合 `-v` 输出将同步内容
	-	`--delete`：删除目标目录多余内容，确保目标目录与源目录内容完全相同
	-	`-e <OPS>`/`--rsh=`：指定用于连接的远程 Shell，缺省为 `ssh`
		-	`-e ssh`：使用 `ssh` 进行远程同步
		-	`-e ssh -p 2344`：指定 `ssh` 并使用 2344 端口
	-	`--link-dest`：指定基准目录，仅备份与基准目录增量部分
		-	首次同步时，在基准目录处同步全部数据
		-	每次同步，在目标目录处仅修改过文件生成新副本，未修改文件为指向基准目录处文件硬链接
		-	适合需要同时维持多个备份的场合，避免备份文件过大
	-	`--remove-source-file`：传输完成后，删除发送方文件
	-	`--port`：使用 `rsync` 协议同步时请求端口号，缺省为 873
	-	`--password-file=<FILE>`：使用 `rsync` 协议同时指定读取密码的文件
		-	或者设置 `RSYNC_PASSWORD` 环境变量

-	传输文件控制
	-	`-c`/`--checksum`：检查文件内容摘要决定是否重新传输
		-	默认只检查文件大小、修改日期
	-	`--max-size=<MAX>`、`--min-size=<MIN>`：设置文件大小上、下限
	-	`--size-only`：只同步大小有变化的文件
	-	`-u`/`--update`：仅同步源目录中时间戳更新的文件
	-	`--exclude=<PTN>`/`--exclude <PTN>`：排除文件、目录
		-	排除多个模式可多次设置参数，或者使用 *Bash* 的大括号模式扩展
	-	`--exclude-from=<FILE>`：从文件中读取排除模式，每行一个模式
	-	`--include=<PTN>`：指定必须同步文件
		-	优先级高于 `--exclude`，往往配合使用
	-	`ignore-existing`：跳过目标目录中已存在文件
	-	`--existing`/`--ignore-non-existing`：不同步目标目录中不存在的文件和目录
	-	`-m`：不同步空目录
	-	`-d`/`--dirs`：仅同步目录本身（不同步目录中内容）

-	目标设置
	-	`-b`/`--backup`：备份目标上已经存在文件
		-	备份文件名作为添加 `--suffix` 后缀，缺省为 `~`
	-	`--backup-dir`：目标中已存在文件的保存路径，不指定则为同一目录
	-	`-t`/`--times`：保持 *mtime* 属性
		-	建议任何时候都使用，否则目标文件 *mtime* 设置为系统时间
	-	`-o`/`--owner`：保持属主
	-	`-g`/`--group`：保持group
	-	`-p`/`--perms`：保持权限（不包括特殊权限）
	-	`-D`/`--device --specials`：拷贝设备文件、特殊文件
	-	`-l`/`--links`：如果目标是软链接文，拷贝链接而不是文件
	-	`-R`/`--relative`：使用相对路径，即在目标中创建源中指定的相对路径

		```shell
		$ rsync -R -r /var/./log/anaconda /tmp
			# 将会在目标创建`/tmp/log/anaconda`
			# `.`是将绝对路径转换为相对路径，其后的路径才会创建
		```

-	传输设置
	-	`-z`：同步时压缩数据
	-	`--bwlimit=<KB/s>`：传输带宽上限
	-	`--append`：断点续传
	-	`--append-verify`：传输完成后对文件校验，校验失败则重新传输
	-	`-w`/`--whole-file`：使用全量传输
		-	网络带宽高于磁盘带宽时，此选项更高效
	-	`--partial`：允许恢复中断传输，中断文件将保留在目标目录，下次传输时恢复
		-	否则删除传输中断文件
		-	一般需要配合 `--append`、`--append-verify` 使用
	-	`--partial-dir=<DIR>`：保存中断文件的临时目录
	-	`--progess`：显示进度
	-	`-P`：等价于 `--partial --progress`
	-	`-i`：输出源目录、目标目录中文件差异的详细情况
	-	`-v`、`-vv`、`-vvv`：显示详细、更详细、最详细信息

####	（远程）同步协议

```shell
$ rsync <SRC> <DEST>
```

-	`rsync` 默认使用 *SSH* 进行远程登录、数据传输
	-	`SRC`、`DEST` 中可包含形如 `user@host` 的主机名，与文件名之间 `:` 分隔，省略时表示当前用户、当前主机
	-	若需指定 `ssh` 参数，需使用 `-e` 选项指定

-	若另一端启动有 `rsync` 守护进程，可以使用使用 `rsync://` 协议
	-	`SRC`、`DEST` 为形如 `[USER@]<IP/ADDR>::<MODULE>/<DEST>`、或 `rsync://<IP/ADDR>/<MODULE>/<DEST>` 形式的地址
	-	`MODULE` 是 `rsync` 守护进程指定的资源名，由管理员分配，不是实际地址
	-	可通过 `$ rsync rsync://<IP/ADDR>` 查看所有分配的资源
	-	若同时通过 `-e`/`--rsh` 选项指定 Shell，则会在指定 Shell 中启动新的 `rsync` 守护进程进行同步


