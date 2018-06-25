#	Crate、Mod、可见性、文件结构 

##	文件结构规则

###	拆分文件

Rust一般库中**文件名/文件夹**都表示mod
（测试文件规则比较特殊）

-	mod`foo`在其父mod`bar`中声明 
-	如果mod`foo`没有子mod，将其实现放在`foo.rs`文件中
-	若mod`foo`有子mod，创建文件夹`foo`，将其实现放在
	`foo/mod.rs`中

以上是文件拆分规则，也可以不拆分文件

###	库Crate

库crate中`lib.rs`相当于该crate顶层mod（根mod）

-	所有的mod直接或间接（祖先mod）声明于此，否则不能识别
-	从引用库crate的外部crate角度来看，其名称和库crate同名
	`extern crate crate_name;`的同时就`use crate_name;`，
	此时可将引用其的mod视为根mod的父mod

###	库、二进制Crate

crate中可以同时有lib.rs和main.rs，此时库crate和二进制
crate应该看作相互**独立**
		
-	在两处都使用`mod`关键字**声明定义**mod（不能在`main.rs`
	中使用`use`**声明使用**mod）

-	在`main.rs`中使用`extern crate crate_name`引入
	“外部”库crate

##	可见性规则

###	Mod默认私有

-	默认仅crate内部可见
	-	父mod处直接可用
	-	兄弟mod、子mod可以通过“回溯“声明使用
-	`pub`声明为公用后，对外部crate也可见

###	Fn默认私有

-	默认仅mod“内部”可见（包括后代mod）
	-	当前mod内直接可用
	-	子mod可以通过“回溯”声明可用
-	`pub`声明为公用后，对外部mod也可见

###	说明
	
-	项（mod、fn）的声明使用路径都是相对于当前项，即默认调用
	其后代项（mod、fn），通过以下“回溯”方式调用非直接后代项
	-	`super`直接父mod路径起始：`super::child_mod`
	-	`::`根mod起始：`::child_mod`
-	fn和mod的可见规则相似的，只是注意：fn是否可见只与mod有关
	，mod是否可见只有crate有关。从这个意义上说，crate不能
	看作是“大号“的mod

##	相关关键字

好像都是单一用途（意义），罕见

-	`extern`：**引入**外部crate（同时包含`use crate_name;`）
-	`crate`：标记外部crate
-	`mod`/`fn`：**声明定义（注册）**mod/fn（同crate内仅一次
	，位于其父mod处）
-	`use`：**声明使用**项（mod、fn），用于缩略代码
