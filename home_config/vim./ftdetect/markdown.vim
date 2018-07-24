"
"----------------------------------------------------------
" Name: markdown.vim
" Author: xyy1926
" Create at:
" Update at:
" Description: detect *.markdown and setfiletype
"----------------------------------------------------------

autocmd BufNewFile,BufRead *.md,*.markdown,*.mkd set filetype=markdown
	"`setfiletype markdown`(let vim set filtype first, but
	"failed. It seems that vim will `set filetype=modula2`
	"`help filetype` to get more information about filetype
