#	Vim安装问题

##	Vim配置问题

	./configure --with-features=huge --enable-multibyte \
	--enable-python3interp --with-python3-config-dir=/usr/lib64/python3.4/config-3.4m/ \
	--enable-pythoninterp --with-python-config-dir=/usr/lib64/python2.7/config/ \
	--prefix=/usr/local --enable-cscope

按照以上命令配置，编译出的Vim版本中是**动态**支持
`+python/dyn`和 `+python3/dyn`，此时Vim看似有python支持，
但是在Vim内部`:echo has("python")`和`:echo has("python3")`
都返回`0`

之后无意中尝试去掉对`python`的支持，编译出来的Vim就是可用
的`python3`，不直到为啥
