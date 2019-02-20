"
"----------------------------------------------------------
"   Name: python.vim
"   Author: xyy15926
"   Created at: 2018-11-29 22:05:31
"   Updated at: 2019-02-20 23:22:02
"   Description: 
"----------------------------------------------------------

autocmd BufWrite *.py call SetMakeParam()

function! SetMakeParam()
	let b:filename = expand("%:t")
	execute "set makeprg=flake8\\ ".b:filename."\\ &\\ python3\\ ".b:filename
endfunction

