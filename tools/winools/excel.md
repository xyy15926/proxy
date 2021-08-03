---
title: Excel技巧
categories:
  - Tool
  -	Windows
tags:
  - Tool
  - Windows
  - Office
date: 2020-09-01 16:10:05
updated: 2020-09-01 16:10:05
toc: true
mathjax: true
description: Excel操作技巧
---

##	EXCEL密码破解

###	VBAProject密码

-	`vbaProject.bin`中密码对应字段
	-	`DPB`：当前已存在密码，后跟加密密码值

-	`xls`：97-03格式表格是RAR格式压缩文件
	-	解压之后再次压缩一般无法正常打开，应该是有特殊的压缩
		规则
	-	可以直接用编辑器打开、修改整个文件，但要注意以二进制
		格式打开

-	`xlsm`：07之后带宏表格是ZIP压缩文件
	-	解压之后再次zip打包也无法正常打开，但是zip格式可以
		直接替换其中文件，所以可以直接修改单个文件
	-	vim对zip格式文件处理类型文件夹，所以使用vim二进制打开
		可以修改其中单个文件

####	修改二进制文件

-	处理方法：将密码对应字段“无效”
	-	需要保持文件前后大小不改变
	-	密码字段无效化

-	“删除”密码字段
	-	将`DPB`字段替换为**等长**其他名称
		-	密码无法被正常识别
		-	文件大小没有改变，仅字符被替换
	-	再次打开文件，启用宏会报错
		-	不断点击确认之后即可查看vba工程
		-	为vba工程设置新密码，覆盖替换后错误字段，保存
			可得已知密码文件
			-	右键*VBAProject*
			-	工程属性
			-	保护
			-	查看工程属性密码：修改为新密码即可

-	替换密码字段
	-	将`DPB`后密码值替换为其他值
		-	为保证文件大小不变
			-	若新密码较短需要用`0`填充不足部分
			-	较原始密码长则无需额外操作
	-	打开文件则可以使用已知密码查看vba工程

```bash
 # 二进制打开文件，否则vim会做某些处理，损坏文件
$ vim -b <file>

 # 调用xxd转换字符格式为16进制
 # 可看到16进制、字符对应格式
 # 仅修改左侧16进制有效，修改右侧字符表示不影响文件
 # 内容
$ !%xxd

 # 16进制转换回原始表示
$ !%xxd -r
```

####	VBA脚本穷举密码

-	`xls`格式：打开VBA编辑器执行代码

	```vb
	'移除VBA编码保护
	Sub MoveProtect()
		Dim FileName As String
		FileName = Application.GetOpenFilename("Excel文件（*.xls & *.xla）,*.xls;*.xla", , "VBA破解")
		If FileName = CStr(False) Then
		   Exit Sub
		Else
		   VBAPassword FileName, False
		End If
	End Sub
	 
	'设置VBA编码保护
	Sub SetProtect()
		Dim FileName As String
		FileName = Application.GetOpenFilename("Excel文件（*.xls & *.xla）,*.xls;*.xla", , "VBA破解")
		If FileName = CStr(False) Then
		   Exit Sub
		Else
		   VBAPassword FileName, True
		End If
	End Sub
	 
	Private Function VBAPassword(FileName As String, Optional Protect As Boolean = False)
		  If Dir(FileName) = "" Then
			 Exit Function
		  Else
			 FileCopy FileName, FileName & ".bak"
		  End If
	 
		  Dim GetData As String * 5
		  Open FileName For Binary As #1
		  Dim CMGs As Long
		  Dim DPBo As Long
		  For i = 1 To LOF(1)
			  Get #1, i, GetData
			  If GetData = "CMG=""" Then CMGs = i
			  If GetData = "[Host" Then DPBo = i - 2: Exit For
		  Next
		  If CMGs = 0 Then
			 MsgBox "请先对VBA编码设置一个保护密码...", 32, "提示"
			 Exit Function
		  End If
		  If Protect = False Then
			 Dim St As String * 2
			 Dim s20 As String * 1
			 '取得一个0D0A十六进制字串
			 Get #1, CMGs - 2, St
			 '取得一个20十六制字串
			 Get #1, DPBo + 16, s20
			 '替换加密部份机码
			 For i = CMGs To DPBo Step 2
				 Put #1, i, St
			 Next
			 '加入不配对符号
			 If (DPBo - CMGs) Mod 2 <> 0 Then
				Put #1, DPBo + 1, s20
			 End If
			 MsgBox "文件解密成功......", 32, "提示"
		  Else
			 Dim MMs As String * 5
			 MMs = "DPB="""
			 Put #1, CMGs, MMs
			 MsgBox "对文件特殊加密成功......", 32, "提示"
		  End If
		  Close #1
	End Function
	```

###	Sheet保护密码

####	VBA脚本

-	`xlsx`：打开VBA编辑器执行代码

	```vb
	Sub pj()
	Dim sht As Worksheet
	For Each sht In Worksheets
	sht.Protect AllowFiltering:=True
	sht.Unprotect
	Next
	End Sub
	```

-	`xls`：打开VBA编辑器执行代码

	```vb
	Public Sub AllInternalPasswords()
	' Breaks worksheet and workbook structure passwords. Bob McCormick
	' probably originator of base code algorithm modified for coverage
	' of workbook structure / windows passwords and for multiple passwords
	'
	' Norman Harker and JE McGimpsey 27-Dec-2002 (Version 1.1)
	' Modified 2003-Apr-04 by JEM: All msgs to constants, and
	' eliminate one Exit Sub (Version 1.1.1)
	' Reveals hashed passwords NOT original passwords
	Const DBLSPACE As String = vbNewLine & vbNewLine
	Const AUTHORS As String = DBLSPACE & vbNewLine & _
	"Adapted from Bob McCormick base code by" & _
	"Norman Harker and JE McGimpsey"
	Const HEADER As String = "AllInternalPasswords User Message"
	Const VERSION As String = DBLSPACE & "Version 1.1.1 2003-Apr-04"
	Const REPBACK As String = DBLSPACE & "Please report failure " & _
	"to the microsoft.public.excel.programming newsgroup."
	Const ALLCLEAR As String = DBLSPACE & "The workbook should " & _
	"now be free of all password protection, so make sure you:" & _
	DBLSPACE & "SAVE IT NOW!" & DBLSPACE & "and also" & _
	DBLSPACE & "BACKUP!, BACKUP!!, BACKUP!!!" & _
	DBLSPACE & "Also, remember that the password was " & _
	"put there for a reason. Don't stuff up crucial formulas " & _
	"or data." & DBLSPACE & "Access and use of some data " & _
	"may be an offense. If in doubt, don't."
	Const MSGNOPWORDS1 As String = "There were no passwords on " & _
	"sheets, or workbook structure or windows." & AUTHORS & VERSION
	Const MSGNOPWORDS2 As String = "There was no protection to " & _
	"workbook structure or windows." & DBLSPACE & _
	"Proceeding to unprotect sheets." & AUTHORS & VERSION
	Const MSGTAKETIME As String = "After pressing OK button this " & _
	"will take some time." & DBLSPACE & "Amount of time " & _
	"depends on how many different passwords, the " & _
	"passwords, and your computer's specification." & DBLSPACE & _
	"Just be patient! Make me a coffee!" & AUTHORS & VERSION
	Const MSGPWORDFOUND1 As String = "You had a Worksheet " & _
	"Structure or Windows Password set." & DBLSPACE & _
	"The password found was: " & DBLSPACE & "$$" & DBLSPACE & _
	"Note it down for potential future use in other workbooks by " & _
	"the same person who set this password." & DBLSPACE & _
	"Now to check and clear other passwords." & AUTHORS & VERSION
	Const MSGPWORDFOUND2 As String = "You had a Worksheet " & _
	"password set." & DBLSPACE & "The password found was: " & _
	DBLSPACE & "$$" & DBLSPACE & "Note it down for potential " & _
	"future use in other workbooks by same person who " & _
	"set this password." & DBLSPACE & "Now to check and clear " & _
	"other passwords." & AUTHORS & VERSION
	Const MSGONLYONE As String = "Only structure / windows " & _
	"protected with the password that was just found." & _
	ALLCLEAR & AUTHORS & VERSION & REPBACK
	Dim w1 As Worksheet, w2 As Worksheet
	Dim i As Integer, j As Integer, k As Integer, l As Integer
	Dim m As Integer, n As Integer, i1 As Integer, i2 As Integer
	Dim i3 As Integer, i4 As Integer, i5 As Integer, i6 As Integer
	Dim PWord1 As String
	Dim ShTag As Boolean, WinTag As Boolean

	Application.ScreenUpdating = False
	With ActiveWorkbook
	WinTag = .ProtectStructure Or .ProtectWindows
	End With
	ShTag = False
	For Each w1 In Worksheets
	ShTag = ShTag Or w1.ProtectContents
	Next w1
	If Not ShTag And Not WinTag Then
	MsgBox MSGNOPWORDS1, vbInformation, HEADER
	Exit Sub
	End If
	MsgBox MSGTAKETIME, vbInformation, HEADER
	If Not WinTag Then
	MsgBox MSGNOPWORDS2, vbInformation, HEADER
	Else
	On Error Resume Next
	Do 'dummy do loop
	For i = 65 To 66: For j = 65 To 66: For k = 65 To 66
	For l = 65 To 66: For m = 65 To 66: For i1 = 65 To 66
	For i2 = 65 To 66: For i3 = 65 To 66: For i4 = 65 To 66
	For i5 = 65 To 66: For i6 = 65 To 66: For n = 32 To 126
	With ActiveWorkbook
	.Unprotect Chr(i) & Chr(j) & Chr(k) & _
	Chr(l) & Chr(m) & Chr(i1) & Chr(i2) & _
	Chr(i3) & Chr(i4) & Chr(i5) & Chr(i6) & Chr(n)
	If .ProtectStructure = False And _
	.ProtectWindows = False Then
	PWord1 = Chr(i) & Chr(j) & Chr(k) & Chr(l) & _
	Chr(m) & Chr(i1) & Chr(i2) & Chr(i3) & _
	Chr(i4) & Chr(i5) & Chr(i6) & Chr(n)
	MsgBox Application.Substitute(MSGPWORDFOUND1, _
	"$$", PWord1), vbInformation, HEADER
	Exit Do 'Bypass all for...nexts
	End If
	End With
	Next: Next: Next: Next: Next: Next
	Next: Next: Next: Next: Next: Next
	Loop Until True
	On Error GoTo 0
	End If
	If WinTag And Not ShTag Then
	MsgBox MSGONLYONE, vbInformation, HEADER
	Exit Sub
	End If
	On Error Resume Next
	For Each w1 In Worksheets
	'Attempt clearance with PWord1
	w1.Unprotect PWord1
	Next w1
	On Error GoTo 0
	ShTag = False
	For Each w1 In Worksheets
	'Checks for all clear ShTag triggered to 1 if not.
	ShTag = ShTag Or w1.ProtectContents
	Next w1
	If ShTag Then
	For Each w1 In Worksheets
	With w1
	If .ProtectContents Then
	On Error Resume Next
	Do 'Dummy do loop
	For i = 65 To 66: For j = 65 To 66: For k = 65 To 66
	For l = 65 To 66: For m = 65 To 66: For i1 = 65 To 66
	For i2 = 65 To 66: For i3 = 65 To 66: For i4 = 65 To 66
	For i5 = 65 To 66: For i6 = 65 To 66: For n = 32 To 126
	.Unprotect Chr(i) & Chr(j) & Chr(k) & _
	Chr(l) & Chr(m) & Chr(i1) & Chr(i2) & Chr(i3) & _
	Chr(i4) & Chr(i5) & Chr(i6) & Chr(n)
	If Not .ProtectContents Then
	PWord1 = Chr(i) & Chr(j) & Chr(k) & Chr(l) & _
	Chr(m) & Chr(i1) & Chr(i2) & Chr(i3) & _
	Chr(i4) & Chr(i5) & Chr(i6) & Chr(n)
	MsgBox Application.Substitute(MSGPWORDFOUND2, _
	"$$", PWord1), vbInformation, HEADER
	'leverage finding Pword by trying on other sheets
	For Each w2 In Worksheets
	w2.Unprotect PWord1
	Next w2
	Exit Do 'Bypass all for...nexts
	End If
	Next: Next: Next: Next: Next: Next
	Next: Next: Next: Next: Next: Next
	Loop Until True
	On Error GoTo 0
	End If
	End With
	Next w1
	End If
	MsgBox ALLCLEAR & AUTHORS & VERSION & REPBACK, vbInformation, HEADER
	End Sub
	```



















