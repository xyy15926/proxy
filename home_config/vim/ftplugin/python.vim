"
"----------------------------------------------------------
"   Name: python.vim
"   Author: xyy15926
"   Created at: 2018-11-29 22:05:31
"   Updated at: 2019-01-28 23:03:06
"   Description: 
"----------------------------------------------------------

autocmd BufWrite *.py call SetMakeParam()

function! SetMakeParam()
	let b:filename = expand("%:t")
	execute "set makeprg=pylint\\ ".b:filename."\\ &\\ python3\\ ".b:filename
endfunction

