"
"----------------------------------------------------------
"   Name: python.vim
"   Author: xyy15926
"   Created at: 2018-11-29 22:05:31
"   Updated at: 2018-11-30 00:37:02
"   Description: 
"----------------------------------------------------------

autocmd BufWrite *.py call SetMakeParam()

function SetMakeParam()
	let b:filename = expand("%:t")
	execute "set makeprg=python3\\ ".b:filename
endfunction

