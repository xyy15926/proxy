#	VSCode基础

##	Settings

-	VSCode配置文件分为两种（其中项目会重复）
	-	User Settings：用户对所有项目的配置
	-	Workspace Settings：针对当前项目的设置

-	配置方式分为两种完全相同
	-	UI：提示丰富
	-	JSON：配置快捷，JSON字典格式

###	Command Palette

####	Python

-	选择默认python环境

####	Terminal

-	选择默认terminal，可能的terminal包括cmd、powershell、
	WSL（若启用Linux子系统）

##	Original Setting

###	Terminal

```json
{
	"terminal.integrated.shell.windows"："/path/to/shell",
		// 默认shell
	"terminal.integrated.shellArgs.windows": [ ],
	"terminal.integrated.shellArgs.linux": [ ],
	"terminal.integrated.shellArgs.osx": [ ],
		// VSCode创建terminal启动参数（3系统分别配置）
}
```

-	VSCode terminal分为很integrated、external

-	VSCode应该是兼容所有terminal shell，如
	-	`C:/Windows/System32/cmd.exe`
	-	`C:/Windows/System32/powershell.exe`
	-	`C:/Windows/System32/wsl.exe`：WSL启用
	-	"/path/to/git/bin/bash.exe"：Git中bash等

-	VSCode terminal虽然兼容多种shell，但只能创建默认shell
	-	需要多种shell只能切换默认shell再创建
	-	python shell等shell是特殊shell，无法默认创建，必须要
		在命令面板中创建（虽然在普通shell中打开python环境，
		但是VSCode不认可）

###	Python

```json
{
	"python.condaPath": "/path/to/conda/Scripts",
		// conda安装目录Scripts文件夹
	"python.venvPath": "/path/to/conda/envs",
		// 虚拟环境目录，VSCode会在其中查找虚拟环境，作为
		// Command Palette中的备选项
	"python.pythonPath": "/path/to/python.exe",
		// 默认python解释器路径
	"python.terminal.activateEnvironment": true,
		// 创建python shell时，尝试中激活虚拟环境
}
```

-	`python.terminal.activateEnviroment`激活的虚拟环境由
	`python.pythonPath`决定

	-	VSCode会尝试执行`python.pythonPath`同级中
		`Scripts/activate.bat`激活虚拟环境
	
	-	因此虚拟环境需要安装`conda`，否则没有
		`Scripts/ativate.bat`无法正常激活默认虚拟环境

###	CPPC

####	配置文件

-	`.vscode/c_cpp_properties.json`

	```json
	{
		"configurations": [
			{
				"name": "WSL",
				"includePath": [
					"${workspaceFolder}/**"
				],
				"defines": [
					"LOCAL",
					"_DEBUG",
					"UNICODE",
					"_UNICODE"
				],
				"compilerPath": "/usr/bin/gcc",
				"cStandard": "c11",
				"cppStandard": "c++14",
				"intelliSenseMode": "gcc-x64"
			}
		],
		"version": 4
	}
	```

	-	`C/C++`项目基本配置

-	`.vscode/tasks.json`：利用VSCode的Tasks功能调用WSL的
	GCC/G++编译器

	```json
	{
		// tasks.json
		// See https://go.microsoft.com/fwlink/?LinkId=733558
		// for the documentation about the tasks.json format

		"version": "2.0.0",
		"tasks": [
			{
				"label": "Build",
				"command": "g++",
				"args": [
					"-g",
					"-Wall",
					"-std=c++14",
					"/mnt/c/Users/xyy15926/Code/cppc/${fileBasename}",
					"-o",
					"/mnt/c/Users/xyy15926/Code/cppc/a.out",
					"-D",
					"LOCAL"
				],
				"problemMatcher": {
					"owner": "cpp",
					"fileLocation": [
						"relative",
						"${workspaceRoot}"
					],
					"pattern": {
						"regexp": "^(.*):(\\d+):(\\d+):\\s+(warining|error):\\s+(.*)$",
						"file": 1,
						"line": 2,
						"column": 3,
						"severity": 4,
						"message": 5
					}
				},
				"type": "shell",
				"group": {
					"kind": "build",
					"isDefault": true
				},
				"presentation": {
					"echo": true,
					"reveal": "silent",
					"focus": true,
					"panel": "shared"
				}
			},
			{
				"label": "Run",
				"command": "/mnt/c/Users/xyy15926/Code/cppc/a.out",
				"type": "shell",
				"dependsOn": "Build",
				"group": {
					"kind": "test",
					"isDefault": true
				},
				"presentation":{
					"echo": true,
					"reveal": "always",
					"focus": true,
					"panel": "shared",
					"showReuseMessage": true
				}
			}
		]
	}
	```

	-	这里为方便将运行程序任务同`> Task: Run Test Task`
		任务关联，可以在命令面板执行此指令

-	`.vscode/launch.json`：gdb调试配置
	
	```json
	{
		// launch.json
		// Use IntelliSense to learn about possible attributes.
		// Hover to view descriptions of existing attributes.
		// For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
		"version": "0.2.0",
		"configurations": [
			{
				"name": "(gdb) Bash on Windows Launch",
				"type": "cppdbg",
				"request": "launch",
				"program": "/mnt/c/Users/xyy15926/Code/cppc/a.out",
				"args": ["-f", "Threading"],
				"stopAtEntry": false,
				"cwd": "/mnt/c/Users/xyy15926/Code/cppc/",
				"environment": [],
				"externalConsole": true,
				"MIMode": "gdb",
				"pipeTransport": {
					"debuggerPath": "/usr/bin/gdb",
					"pipeProgram": "C:\\windows\\system32\\bash.exe",
					"pipeArgs": ["-c"],
					"pipeCwd": ""
				},
				"setupCommands": [
					{
						"description": "Enable pretty-printing for gdb",
						"text": "-enable-pretty-printing",
						"ignoreFailures": false
					}
				],
				"sourceFileMap": {
					"/mnt/c": "c:\\",
					"/mnt/d": "d:\\"
				},
				"preLaunchTask": "Build"
			},
		]
	}
	```

###	Git

```json
{
	"git.ignore.MissingGitWarning": true,
	"git.path": "/path/to/xxxgit.exe"
}
```

-	"git.path"既可以是windows下Git，也可以是“伪装”Git，使用
	工具[wslgit](https://github.com/andy-5/wslgit)，让VSCode
	直接使用WSL内的Git

##	KeyMapper

-	`<c-s-\`>`：新建默认terminal绘画
-	`<c-s-p>`：command palette





