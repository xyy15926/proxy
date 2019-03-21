#	TensorFlow介绍

##	基本使用

###	Approach

-	assemble a graph
-	use a session to execute operation in the graph

###	Session



####	`tf.InteractiveSession`

交互式会话代替`tf.Session`

```python
sess = tf.InteractiveSession()
	# 启动交互式会话
x = tf.Variable([1.0, 2.0])
a = tf.Variable([3.0, 3.0])
x.initializer.run()
	# 替代`Session.run()`，避免使用变量持有对话
sub = tf.subtract(x, a)
print(sub.eval())
	# 同`.run`
```

-	TensorFlow程序通常组织成构建阶段、执行阶段
	-	构建阶段：创建图表示计算任务
	-	执行阶段：使用会话执行图中OP


####	变量

-	变量维护图执行过程中状态信息

```python
state = tf.Variable(0, name="counter")
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init_op = tf.initialize_all_variables()
	# 启动图后，初始化op
	# 会报错？？？

withe tf.Session() as sess:
	sess.run(init_op)
	print(sess.run(state))
	for _ in range(3):
		sess.run(update)
		print(sess.run(state))
```

####	Fetch

Fetch机制允许，在使用Session对象`run()`执行图时，传入需要
**取回**的结果，取回操作输出内容

```python
input1 = tf.costant(3.0)
input2 = tf.contant(2.0)
input3 = tf.contant(5.0)
intermed = tf.add(input1, input2)
mul = tf.mul(input1, inputmed)

with tf.Session():
	result = sess.run([mul, intermed])
		# 在`Session.run`时传入需要取回的结果`intermed`
	print(result)
```

####	Feed

Feed机制允许使用新tensor值替代图中任意操作的tensor

-	可以提供feed数据作为`.run`调用的参数
-	feed只在调用它的方法内有效，方法结束feed消失
-	常将某些特殊OP指定为*feed*OP，即使用`tf.placeholder()`
	为这些操作创建占位符

```python
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
	# 为feed OP创建占位符
output = tf.multiply(input1, input2)

with tf.Session() as sess:
	print(sess.run([output], feed_dict={
		input1: [7.],
		input2: [2.]
	}))
```


