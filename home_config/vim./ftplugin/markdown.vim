"
"----------------------------------------------------------
" Name: markdown.vim
" Author: xyy15926
" Created at:
" Updated at:
" Description: settings especially for markdown
"----------------------------------------------------------

if exists("b:did_ftplugin")
	finish
	"stop execute following commands in the vimscripts
endif
let b:did_ftplugin = 1
	"avoid executing the same plugin twice for the same
	"buffer, when using `:edit` command without arguments
	"for example.
	"In fact, ftplugin will be executed twice though open
	"a file with nothing done, so this is necessary

onoremap <buffer> ih :<c-u>execute"normal! ?^==\\+$\r:nohlsearch\rkvg_"<cr>
	"select last title marked with `===`
onoremap <buffer> ah :<c-u>execute "normal! ?^==\\+$\r:nohlsearch\rg_vk0"<cr>
	"select last title marked with `===`, with `===`
nnoremap <buffer> <localleader>cc ^hi-<esc>lli`<esc>f：i`<esc>
	"convert code-block to list
nnoremap <buffer> <localleader>lt :<c-u>ilist /^#\{1,6\}[ \t]\+<cr>
	"list all tiltes marked with `#`
	"`:dlist` also functioned well with `:set define`
nnoremap <buffer> <localleader>e  60\|
	"jump to column 60, which should be the maxium length
if ! exists("b:lang")
	let b:lang = input("about what lang?")
endif
inoremap <buffer> ``` ```<esc>"=b:lang<c-m>po```<esc>O
	"ask user what language is this md-file about, so to
	"input code block easier

iabbrev <buffer> >- > -
	"shortcuts for quoted list
iabbrev <buffer> [] [ ]
	"easiser to input todo list
