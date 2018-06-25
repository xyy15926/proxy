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
inoremap JK <esc>
vnoremap JK <esc>
nnoremap JK <NUL>
"inoremap <esc> <nop>
	"<nop>: no operation, disable <esc> to normal mode

"DIY key-mapping
"
nnoremap <leader>ev :vsplit $MYVIMRC<cr>
nnoremap <leader>sv :source $MYVIMRC<cr>
nnoremap <leader>em :vsplit ~/.vim/plugin/keymapper.vim<cr>
nnoremap <leader>sm :source ~/.vim/plugin/keymapper.vim<cr>
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
inoremap <leader>; <esc>mqA;<esc>`qa
	"<leader>; to add a `;` at the end of a line
onoremap in( :<c-u>normal! f(vi(<cr>
	"select content in next parentheses
onoremap il( :<c-u>normal! f)vi(<cr>
	"select content in current parentheses, if in a 
	"parentheses, or next

iabbrev @1 xyy15926@163.com
iabbrev @2 xyy15926@gmail.com
iabbrev xyy xyy15926
	"self-info shortcuts
"iabbrev ---- &mdash
	"replace `----` with last word
