"
"----------------------------------------------------------
"	Name: quick_fix.vim
"	Author: xyy15926
"	Created at:
"	Updated at:
"	Description: quick_fix window related
"----------------------------------------------------------

nnoremap <leader>qf :call QuickfixToggle()<cr>

let g:quickfix_is_open=0
	"define a global variable to store quick fix window
	"status

function! QuickfixToggle()

	if g:quickfix_is_open
		cclose
		let g:quickfix_is_open=0
		execute g:quickfix_return_to_window . "wincmd w"
			"return to previous window
	else
		let g:quickfix_return_to_window = winnr()
			"store window index in a global variable for 
			"returning to previous window when escaping
			"from quickfix
		copen
		let g:quickfix_is_open=1
	endif

endfunction

