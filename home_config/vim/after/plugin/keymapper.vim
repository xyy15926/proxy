"
"----------------------------------------------------------
" Name: keymapper.vim
" Author: xyy15926
" Created at:
" Updated at:
" Description: vim global keyboards shortcuts settings
"----------------------------------------------------------

"set JK to normal
"
inoremap KK <esc>
vnoremap KK <esc>
"nnoremap JK <NUL>
nnoremap K ;
"inoremap <esc> <nop>
	"<nop>: no operation, disable <esc> to normal mode

"DIY key-mapping
"
nnoremap <leader>ev :vsplit $MYVIMRC<cr>
nnoremap <leader>sv :source $MYVIMRC<cr>
nnoremap <leader>em :vsplit ~/.vim/after/plugin/keymapper.vim<cr>
nnoremap <leader>sm :source ~/.vim/after/plugin/keymapper.vim<cr>
	"keyboards shortcuts for editing vim config
nnoremap <leader>" viw<esc>a"<esc>hbi"<esc>lel
	"wrap current word with ""
vnoremap <leader>" :'<I"<esc>:'>I"<esc>
	"wrap the selected area in visual mode with ""
inoremap <c-u> <esc>viwUea
	"convert curent word uppercase
inoremap <s-cr> <esc>o
	"shift-enter to start a new line
	"though this can work only in Gvim
inoremap ;/ <esc>A
inoremap ；/ <esc>A
	" just for convinience
" inoremap <leader>; <esc>mqA;<esc>`qa
	"<leader>; to add a `;` at the end of a line
onoremap in( :<c-u>normal! f(vi(<cr>
	"select content in next parentheses
onoremap il( :<c-u>normal! F)vi(<cr>
	"select content in current parentheses, if in a
	"parentheses, or next
nnoremap <leader><localleader>e  60\|
	"jump to column 60, which should be the maxium length

iabbrev @1 xyy15926@163.com
iabbrev @2 xyy15926@gmail.com
iabbrev xyy xyy15926
	"self-info shortcuts
"iabbrev ---- &mdash
"replace `----` with last word
iabbrev xtime <c-r>=strftime("%Y-%m-%d %H:%M:%S")<cr>
	"insert current time to cursor
