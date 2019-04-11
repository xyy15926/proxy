#	Heap

##	*Heap*

堆/大根堆：每个节点包含一个键、且大于等于其子女键、基本完备
的二叉树

-	认为非叶子节点自动满足大于其子女键值
-	根到某个叶子节点路径上，键值序列递减（允许键值相等则是
	非递增）
-	键值之间没有从左到右的次序，即树的同一层节点间没有任何
	关系，或者说**左右子树**之间没有任何关系

###	堆特性

-	完全二叉树特性参见*完全二叉树*
-	堆的根总是包含了堆的最大元素
-	堆的节点及该节点子孙也是一个堆

> - 堆这种数据结构经常被用于实现优先队列

###	最小堆

*min-heap*：（最大）堆的镜像

-	堆的主要特性最小堆也满足，但
	-	最小堆根节点包含最小元素
-	事实上可以通过给元素取反构造堆得到最小堆

##	堆算法

###	Bottom-Up Heap Construction

自底向上堆构造

####	算法

-	按照给定顺序放置键初始化一棵完全二叉树
-	对二叉树进行堆化，从最后父母节点开始、到根为止，算法检
	查节点键是否满足大于等于其子女键
	-	若不满足，交换节点键K和其子女最大键值
	-	在新位置上检查是否满足大于子女键值，直到满足为止
-	对以当前节点为根的子树满足堆化条件后，算法对节点直接前趋
	进行检查，直到树根满足堆化

```c
HeapBottomUp(H[1..n])
	// 用自底向上算法，从给定数组元素中构造一个堆
	// 输入：可排序数组H[1..n]
	// 输出：堆H[1..]
	for i=floor(n/2) downto 1 do
		k = i
		v = H[k]
		heap = false
		while not heap 2*k <=n do
			j = 2*k
			if j < n
				if H[j] < H[j+1]
				// 这里两个`if`组合保证不越界、拿到最大子女
					j = j + 1
			if v >= H[j]
				heap = True
			else
				H[k] = H[j]
				k = j
				// 检查新位置是否满足堆化
	H[k] = v
```

####	算法特点

-	算法效率
	-	最差情况下，每个位于树第i层节点会移到叶子层h中，每次
		移动比较两次，则总键值比较次数
		$C_{worst}=2(n-log_2(n+1))

###	Top-Down Heap Construction

自顶向下堆构造算法：把新键连续插入已经构造好的堆

####	算法

-	把包含键K的新节点附加在当前堆最后一个叶子节点
-	将K与其父母进行比较
	-	若后者大于K，算法停止
	-	否则交换这两个键，并比较K和其新父母，直到K不大于其
		父母或是达到了树根

####	算法特点

-	算法效率
	-	操作所需键值比较次数小于堆高度$h \approx log_2 n$，
		则总比较次数$\in \Theta(nlogn)$

###	删除堆根节点

####	算法

-	根键和堆中最后一个键K交换
-	堆规模减1（删除原根键）
-	按照自底向上堆构造算法，把K沿着树向下筛选使得新树满足
	堆化

####	算法特点

-	算法效率
	-	效率取决于树堆化所需比较次数，不可能超过树高度两倍，
		即$\in O(logn)$

###	堆排序算法

####	算法

-	为给定数组构造堆
-	删除最大键，即对堆应用n-1此根删除操作

####	算法特点

-	算法效率
	-	根删除阶段算法所需比较次数
		$$\begin{align}
		C(n) & \leq 2\sum_{i=1}^{n-1} \lfloor log_2i \rfloor \\
			& \leq 2 \sum_{i=1}^{n-1} log_2(n-1) \\
			& = 2(n-1)log_2(n-1)
		\end{align}$$

	-	两阶段总效率$\in O(nlogn)$
	-	详细分析证明，无论最差情况、平均情况下，时间效率
		$\in \Theta(nlogn)$

-	堆排序时间效率和归并排序、快排时间效率属于同一类，但是
	堆排序时在位的（不需要额外存储空间）

-	随机实验表明：堆排序比快排慢、和归并排序相比有竞争力


