"
"----------------------------------------------------------
"   Name: execution.vim
"   Author: xyy15926
"   Created at: 2018-09-22 15:34:22
"   Updated at: 2018-09-22 16:10:03
"   Description: 
"----------------------------------------------------------

autocmd BufNewFile *.md,*.py,*.rs,*.c,*.cpp,*.h,*.sh,*.java vnoremap <F5> execute "<c-u>echo ExecBlock()<cr>"

function! GetVisualSelection()
	let [line_start, column_start] = getpos("'<")[1:2]
	let [line_end, column_end] = getpos("'>")[1:2]
	let lines = getline(line_start, line_end)
	if len(lines) == 0
		retrun ''
	endif
	let lines[-1] = lines[-1][:column_end - ($selection == 'inclusive' ? 1 : 2)]
	let lines[0] = lines[0][column_start - 1:]
	return join(lines, "\n")
endfunction

function! ExecBlock()
	if &filetype ==# "markdown"
		let f_type = b:lang
	else
		let f_type = &filetype
	echo f_type

	if f_type ==# "python"
		let code_block = GetVisualSelection()
		execute "<c-u>!clear; python -c" shellescape(code_block, 2) "<cr>"
	endif
endfunction

