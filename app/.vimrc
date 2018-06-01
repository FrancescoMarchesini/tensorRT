set noswapfile
set syntax=on
set tabstop=4
set softtabstop=4
set shiftwidth=4
set noexpandtab

augroup project
	    autocmd!
	    autocmd BufRead,BufNewFile *.h,*.c,*.cu  set filetype=c.doxygen
augroup END

set path+=/usr/local/cuda/include
set path+=/usr/local/cuda/lib64
set path+=/usr/local/cuda/target/aarch64-linux/include
set path+=/usr/local/cuda/target/aarch64-linux/include/thrust
set path+=/usr/local/cuda/target/aarch64-linux/include/crt
set path+=/usr/local/cuda/sample/7_CUDALibraries
set path+=/usr/include
set path+=/usr/include/EGL
set path+=/usr/include/GL
set path+=/usr/include/GLES2
set path+=/usr/include/GLES3
set path+=/usr/include/c++
set path+=/usr/include/NVX
set path+=/usr/include/VX
set path+=/usr/include/aarch64-linux-gnu
set path+=/usr/include/X11
set path+=/usr/include/glib-2.0
set path+=/usr/include/gstreamer-1.0
set path+=/usr/include/opencv
set path+=/usr/include/opencv2

" vision work directorires ????
set path+=/usr/share/visionworks/sources/3rdparty
set path+=/usr/share/visionworks/sources/nvxio/include/NVX
set path+=/usr/share/visionworks/sources/nvxio/include/OVX
set path+=/usr/share/visionworks/sources/nvxio/src/NVX
set path+=/usr/share/visionworks/sources/nvxio/src/OVX
set path+=/usr/share/visionworks/sources/nvxio/src

" tesortRT

" ---------------------------------------------------------------
let g:netrw_banner = 0
let g:netrw_browse_split = 1
let g:netrw_winsize = 25
let g:netrw_banner = 0
let g:netrw_liststyle = 3
let g:netrw_browse_split = 4
let g:netrw_altv = 1
let g:netrw_winsize = 25
augroup ProjectDrawer
  autocmd!
  autocmd VimEnter * :Vexplore
augroup END

