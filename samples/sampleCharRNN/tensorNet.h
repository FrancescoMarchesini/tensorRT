#ifndef _NET_H__
#define _NET_H__


#include "NvInfer.h"
#include "NvUtils.h"
#include "cuda_runtime_api.h"
#include <cassert>
#include <cmath>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <cstdio>

using namespace nvinfer1;
/////////////////////////////
//TensorNet Main
/////////////////////////////
class tensorNet
{
	public:

	protected:

		//tag per il logger
		#define LOG_GIE "[GIE] " 
		// Logger for info/warning/errors
		class Logger : public nvinfer1::ILogger			
		{
			void log(Severity severity, const char* msg) override
			{
				//suppress info-level messages
				if (severity != Severity::kINFO)
					std::cout << LOG_GIE << msg << std::endl;
			}
		} gLogger;

		IRuntime* mRuntime;			 //esecuzione dell'inferenza
		ICudaEngine* mEngine;		 //engine
		IExecutionContext* mContext; //eseguzione kernel

		//oggetto per serializare o deserializzare il gie da e su file
		IHostMemory *gieModelStream{nullptr};
		PluginFactory 	pluginFactory ; //interfaccia per creazine modello	
};
#endif;
