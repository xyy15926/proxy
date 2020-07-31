---
title: 函数式编程
tags:
  - Python
categories:
  - Python
date: 2019-04-22 00:33:40
updated: 2019-04-22 00:33:40
toc: true
mathjax: true
comments: true
description: 函数式编程
---

##	`functools`

###	`total_ordering`

`total_ordering`：允许类只定义`__eq__`和其他中的一个，其他
富比较方法由装饰器自动填充

```python
from functools import total_ordering

class Room:
	def __init__(self, name, length, width):
		self.name = name
		self.length = length
		self.width = width
		self.square_feet = self.length * self.width

@total_ordering
class House:
	def __init__(self, name, style):
		self.name = name
		self.style = style
		self.rooms = list()

	@property
	def living_space_footage(self):
		return sum(r.square_feet for r in self.rooms)

	def add_room(self, room):
		self.rooms.append(room)

	def __str__(str):
		return "{}: {} squre foot {}".format(
			self.name,
			self.living_space_footage,
			self.style)

	def __eq__(self, other):
		return self.living_space_footage == other.living_space_footage

	def __lt__(self, other):
		return self.living_space_footage < other.living_space_footage
```


