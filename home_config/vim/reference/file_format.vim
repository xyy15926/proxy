autocmd BufNewFile *.cc,*.sh,*.java,*.cpp,*.h,*.hpp,*.py,*.lua exec ":call SetTitle()"
"新建.cc,.java,.sh,.cpp,.h, .hpp,
"""定义函数SetTitle，自动插入文件头
func SetTitle()
    let filetype_name = strpart(expand("%"), stridx(expand("%"), "."))
    let file_name = strpart(expand("%"), 0, stridx(expand("%"), "."))
    if file_name =~ "\/"
        let file_name = strpart(file_name, strridx(file_name, "/") + 1)
    endif
    let time_value = strftime("%Y-%m-%d %H:%M:%S")
    if filetype_name == ".sh"
        call setline(1, "\#!/bin/bash")
        call append(line("."), "")
        call append(line(".")+1, "\#########################################################################")
        call append(line(".")+2, "\# File Name: ". file_name . filetype_name)
        call append(line(".")+3, "\# Created on: ".time_value)
        call append(line(".")+4, "\# Author: glendy")
        call append(line(".")+5, "\# Last Modified: ".time_value)
        call append(line(".")+6, "\# Description: ")
        call append(line(".")+7, "\#########################################################################")
        call append(line(".")+8, "")
    else
        if filetype_name == ".lua"
            call setline(1, "\--lua")
            call append(line("."), "") 
            call append(line(".")+1, "\--#########################################################################")
            call append(line(".")+2, "\--# File Name: ". file_name . filetype_name)
            call append(line(".")+3, "\--# Created on: ".time_value)
            call append(line(".")+4, "\--# Author: glendy")
            call append(line(".")+5, "\--# Last Modified: ".time_value)
            call append(line(".")+6, "\--# Description: ")
            call append(line(".")+7, "\--#########################################################################")
            call append(line(".")+8, "") 
            call append(line(".")+9, file_name . " = {}")
            call append(line(".")+10, file_name .".__index = ". file_name)
            call append(line(".")+11, "function ". file_name .":new()")
            call append(line(".")+12, "    local o = {}")
            call append(line(".")+13, "    self.__index = self")
            call append(line(".")+14, "    setmetatable(o, self)")
            call append(line(".")+15, "    \-- construct function code here")
            call append(line(".")+16, "    return o")
            call append(line(".")+17, "end")
            call append(line(".")+18, "function ". file_name .":hotfix()")
            call append(line(".")+19, "    setmetatable(self, ". file_name .")")
            call append(line(".")+20, "end")
            call append(line(".")+21, "") 
            call append(line(".")+22, "return ". file_name)
        else
            if filetype_name == ".py"
                call setline(1, "\# -*- coding: utf-8 -*-")
                call append(line("."), "") 
                call append(line(".")+1, "\#########################################################################") 
                call append(line(".")+2, "\# File Name: ". file_name . filetype_name)  
                call append(line(".")+3, "\# Created on : ".time_value)  
                call append(line(".")+4, "\# Author: glendy")
                call append(line(".")+5, "\# Last Modified: ".time_value)
                call append(line(".")+6, "\# Description:")  
                call append(line(".")+7, "\#########################################################################")
                call append(line(".")+8, "")
            else
                call setline(1, "\/*")
                call append(line("."), " * File Name: ". file_name . filetype_name)
                call append(line(".")+1, " * ")
                call append(line(".")+2, " * Created on: ".time_value)
                call append(line(".")+3, " * Author: glendy")
                call append(line(".")+4, " * ")
                call append(line(".")+5, " * Last Modified: ".time_value)
                call append(line(".")+6, " * Description: ")
                call append(line(".")+7, " */")
                call append(line(".")+8, "")
                if filetype_name == ".h"
                    call append(line(".")+9, "#ifndef _". toupper(file_name) . substitute(toupper(filetype_name), ".", "_", "") ."_")
                    call append(line(".")+10, "#define _". toupper(file_name) . substitute(toupper(filetype_name), ".", "_", "") ."_")
                    call append(line(".")+11, "")
                    call append(line(".")+12, "class " . file_name)
                    call append(line(".")+13, "{")
                    call append(line(".")+14, "public:")
                    call append(line(".")+15, "")
                    call append(line(".")+16, "protected:")
                    call append(line(".")+17, "")
                    call append(line(".")+18, "};")
                    call append(line(".")+19, "")
                    call append(line(".")+20, "#endif //". toupper(file_name) . substitute(toupper(filetype_name), ".", "_", "") ."_")
                endif
            endif
        endif
    endif
endfunc
