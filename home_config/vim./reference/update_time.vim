st modified time of a file  
function SetLastModifiedTime(lineno)  
    let modif_time = strftime("%Y-%m-%d %H:%M:%S")
    if a:lineno == "-1"  
            let line = getline(7)  
    else  
            let line = getline(a:lineno)  
    endif
    if line =~ '\sLast Modified:'
            let line = strpart(line, 0, stridx(line, ":")) . ": " . modif_time
    endif  
    if a:lineno == "-1"  
            call setline(7, line)  
    else  
            call append(a:lineno, line)  
    endif  
endfunc

" map the SetLastModifiedTime command automatically  
autocmd BufWrite *.cc,*.sh,*.java,*.cpp,*.h,*.hpp,*.py,*.lua call SetLastModifiedTime(-1)
