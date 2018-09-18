"
"----------------------------------------------------------
"   Name: tab_switch.vim
"   Author: xyy15926
"   Created at: 2018-08-19 16:03:29
"   Updated at: 2018-08-21 14:59:56
"   Description: 
"----------------------------------------------------------

function! Switch_Tab(num)
	let s:count = a:num
	execute "tabnext"s:count
endfunction

function! Switch_Tab_Init()
	for i in range(1, 9):
		exe "noremap <M-" . i . "> :call Switch_Tab(" . i . ")<CR>"
	endfor
endfunction

" autocmd VimEnter * call Switch_Tab_Init()
