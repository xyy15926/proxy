"
"----------------------------------------------------------
"   Name: heading_file.vim
"   Author: xyy15926
"   Created at: 2018-05-20 15:30:37
"   Updated at: 2019-03-26 11:31:11
"   Description: vim-scripts for auto-adding file information
"----------------------------------------------------------

autocmd BufNewFile *.py,*.rs,*.c,*.cpp,*.h,*.sh,*.java,*.vim call SetHead()
autocmd BufWrite *.py,*.rs,*.c,*.cpp,*.h,*.sh,*.java,*.vim call UpdateTime(-1)

function SetHead()

	normal! gg9O
		"move the original heading lines downwards, so this
		"function could be called to set heads for existing
		"files

	let file = expand("%:t")
	let time = strftime("%Y-%m-%d %H:%M:%S")
	let author = "xyy15926"

	if &filetype ==# "python"
		call setline(1, "\#!  /usr/bin/env python3")
		call setline(2, "\#----------------------------------------------------------")
		call setline(3, "\#   Name: " . file)
		call setline(4, "\#   Author: " . author)
		call setline(5, "\#   Created at: " . time)
		call setline(6, "\#   Updated at: " . time)
		call setline(7, "\#   Description:")
		call setline(8, "\#----------------------------------------------------------")
	elseif &filetype  ==# "vim"
		call setline(1, "\"")
		call setline(2, "\"----------------------------------------------------------")
		call setline(3, "\"   Name: " . file)
		call setline(4, "\"   Author: " . author)
		call setline(5, "\"   Created at: " . time)
		call setline(6, "\"   Updated at: " . time)
		call setline(7, "\"   Description:")
		call setline(8, "\"----------------------------------------------------------")
	elseif &filetype ==# "rust"
		call setline(1, "//")
		call setline(2, "//----------------------------------------------------------")
		call setline(3, "//  Name: " . file)
		call setline(4, "//  Author: " . author)
		call setline(5, "//  Created at: " . time)
		call setline(6, "//  Updated at: " . time)
		call setline(7, "//  Description:")
		call setline(8, "//----------------------------------------------------------")
	elseif &filetype ==# "cpp" || &filetype ==# "c"
		call setline(1, "/*----------------------------------------------------------")
		call setline(2, " *  Name: " . file)
		call setline(3, " *  Author: " . author)
		call setline(4, " *  Created at: " . time)
		call setline(5, " *  Updated at: " . time)
		call setline(6, " *  Description:")
		call setline(7, " *----------------------------------------------------------")
		call setline(8, " */")
	elseif &filetype ==# "sh"
		call setline(1, "\#!  /usr/bin/env shell")
		call setline(2, "\#----------------------------------------------------------")
		call setline(3, "\#   Name: " . file)
		call setline(4, "\#   Author: " . author)
		call setline(5, "\#   Created at: " . time)
		call setline(6, "\#   Updated at: " . time)
		call setline(7, "\#   Description:")
		call setline(8, "\#----------------------------------------------------------")
	endif
endfunction

function UpdateTime(lineno)
	let update_time = strftime("%Y-%m-%d %H:%M:%S")
	let lineno = 6
		"this depends on your formation
	if a:lineno != -1
		lineno = a:lineno
	endif

	let line = getline(lineno)
	if line[4:] =~ "Updated"
		call setline(lineno, line[:3] . "Updated at: " . update_time)
			"set line with original formation
	endif
endfunction

