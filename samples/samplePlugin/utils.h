#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>

#define LOG_CUDA "[CUDA ]"
inline void CHECK_CUDA(int status)
{
	if(status !=0)
	{
		std::cout<<LOG_CUDA<<"Fallito CUDA"<<std::endl;		
	}
}
#endif;
