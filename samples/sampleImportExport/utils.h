#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <cmath>
#include <sys/stat.h>
#include <time.h>
#include <string.h>

#define LOG_CUDA "[CUDA] "
inline void CHECK_CUDA(int status)
{
	if(status !=0)
	{
		std::cout<<LOG_CUDA<<"Fallito CUDA"<<std::endl;		
	}
}

inline std::string locateFile(const std::string& input)
{
	std::string file = "data/samples/mnist/" + input;
	struct stat info;
	int i, MAX_DEPTH = 10;
	for (i = 0; i < MAX_DEPTH && stat(file.c_str(), &info); i++)
		file = "../" + file;

    if (i == MAX_DEPTH)
    {
		file = std::string("data/mnist/") + input;
		for (i = 0; i < MAX_DEPTH && stat(file.c_str(), &info); i++)
			file = "../" + file;		
    }

	assert(i != MAX_DEPTH);

	return file;
}

// simple PGM (portable greyscale map) reader
inline void readPGMFile(const std::string& fileName,  uint8_t buffer[], int INPUT_H, int INPUT_W)
{
	std::ifstream infile(locateFile(fileName), std::ifstream::binary);
	std::string magic, h, w, max;
	infile >> magic >> h >> w >> max;
	infile.seekg(1, infile.cur);
	infile.read(reinterpret_cast<char*>(buffer), INPUT_H*INPUT_W);
}

#endif
