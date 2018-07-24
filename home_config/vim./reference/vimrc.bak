"DISPLAY CONFIG
set nocompatible
set ruler		"show cursor status on right buttom
syntax on
"
"set highlight search
"highlight search cterm=none ctermbg=blue
autocmd cursorhold * set nohls
noremap n :set hls<cr>n
noremap N :set hls<cr>N
noremap / :set hls<cr>/
noremap ? :set hls<cr>?
noremap * :set hls<cr>*
noremap # :set hls<cr>#
"
"locate cursor
set cursorcolumn
set cursorline
"set cursorcolumn* with highlight-group grey
"highlight cursorcolumn cterm=none ctermbg=grey
"highlight cursorline cterm=none ctermbg=grey



"FUNCTION CONFIG
"
"set ignorecase default, but if the expression contain uppercase, set case
"sensitive
set ignorecase
set smartcase
"
"set JK to normal  
inoremap JK <esc>
vnoremap JK <esc> 
nnoremap JK <NUL>
"
"get the cursor word
nnoremap <space><space> viwy<esc>



"FORMAT CONFIG
set foldenable
set foldmethod=manual
set number						"show line number on the left
"
"set tab/indent
set autoindent
set smartindent
set showmatch
set tabstop=4					"set how much space a tab display, default 8
set softtabstop=4				"set how much space backspace/tab delete/add when editing, 
								"if not multi-tabstop it will turn into tab+space
set shiftwidth=4				"set indent length
"set expandtab					"transform tab into spaces, default noexpandtab

								"expand("%:e")->file's extension name,
								"expand("%:r")->file name without extension
if expand("%:e") == "py"
	set expandtab
endif
if expand("%:e") == "txt"
	set noexpandtab
endif
"
"set listchars
set list	"show tab,line-end-space,etc
set listchars=tab:»·,trail:-,precedes:?,extends:?,eol:↩︎
										"tab->tab, trail->line-end-space
										"precedes->left text out of view, extends->right text outof view
										"eol->end of line,
										"vim command :digraphs may be needed to find SpecialKeys and their codes,
										"then <ctrl-k> in insert-mode to input the codes of the SpecialKeys you want
"highlight SpecialKey,NonText	"mean to highlight specialkey nontext, but failed
"
"others
set scrolloff=5		"set the position(scrolloff) where vim begin to scroll
set backspace=indent,eol,start		"set what backspace could delete
set wrap		"autowrap(show the text next line, no-add change-line sig) when line extend to the border



"SYS CONFIG
"
set nobackup
set directory=~/.vim/tmp
"set backupdir=~/.vim/bakcup
"
"set encoding
set fileencodings=utf-8,ucs-bom,gbk18020,gbk,gb2312		"decoding sequence
set termencoding=utf-8		"terminal encoding
set encoding=utf-8		"encoding used inner vim, including buffer, menu



"PLUGIN INSTALL CONFIG
filetype off
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
Plugin 'VundleVim/Vundle.vim'
Plugin 'scrooloose/nerdtree'
Plugin 'vim-scripts/taglist.vim'
"Plugin 'flazz/vim-colorschemes'
Plugin 'rust-lang/rust.vim'
"Plugin 'godlygeek/tabular'
"Plugin 'plasticboy/vim-markdown'
call vundle#end()
filetype plugin on
"
"
"taglist setting
let Tlist_Show_One_File=1
let Tlist_Exit_OnlyWindow=1
let Tlist_Use_Right_Window=1
let Tlist_Show_Menu=1
"let Tlist_Auto_Open=1
nmap <F6> :TlistToggle<cr>
"
"
"nerdtree setting
nmap <F5> :NERDTreeToggle<cr>
"autocmd vimenter * NERDTree
"autocmd vimenter * TList
"
"
"cscope setting
if has('cscope')
	set cscopeprg=/usr/local/bin/cscope
	set cscopetagorder=1					"set search tags file first instead of cscope.out
	if filereadable('cscope.out')
		cs add cscope.out
		nmap <C-\>g :cs find g <C-R>=expand("<cword>")<CR><CR>			"functions definition
		nmap <C-\>c :cs find c <C-R>=expand("<cword>")<CR><CR>			"functions calling this
		nmap <C-\>d :cs find d <C-R>=expand("<cword>")<CR><CR>			"functions called by this
		nmap <C-\>t :cs find t <C-R>=expand("<cword>")<CR><CR>			"string
		nmap <C-\>e :cs find e <C-R>=expand("<cword>")<CR><CR>			"egrep mod to find string
		nmap <C-\>f :cs find f <C-R>=expand("<cfile>"><CR><CR>			"file
		nmap <C-\>i :cs find i ^<C-R>=expand("<cfile>"><CR>$<CR>		"files #include this
	endif
endif



"schemes config
colorscheme molokai	"this colorscheme need to been download from
					"https://raw.githubusercontent.com/tomasr/molokai/master/colors/molokai.vim
					"and then mv to ~/.vim/colors/
hi normal ctermbg=none	"set vim background none show the terminal background
"
"
"
"macros
run macros/heading_file.vim
