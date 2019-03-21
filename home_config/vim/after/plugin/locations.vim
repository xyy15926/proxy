"
"----------------------------------------------------------
" Name: locations.vim
" Author: xyy15926
" Created at:
" Updated at:
" Discription: settings about locations in vim
"----------------------------------------------------------

"set highlight search{{{
"
"highlight search cterm=none ctermbg=blue
autocmd cursorhold * set nohls
	"listen cursorhold event, rm hls
noremap n :set hls<cr>n
noremap N :set hls<cr>N
noremap / :set hls<cr>/
noremap ? :set hls<cr>?
noremap * :set hls<cr>*
noremap # :set hls<cr>#
	"set hls whenever search
"
"}}}


"locate cursor{{{
"
set cursorcolumn
set cursorline
"set cursorcolumn* with highlight-group grey
highlight cursorcolumn cterm=none ctermbg=23
	"the number show the color
highlight cursorline cterm=none ctermbg=23
"
"}}}

