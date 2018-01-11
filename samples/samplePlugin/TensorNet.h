#ifndef __TENSORNET_H__
#define __TENSORNET_H__

#include "addPlugin.h"
#include <iostream>

using namespace nvinfer1
using namespace nvcaffeparser1

class tensorNet
{
	public:
		tensorNet();
		static void importTrainedCaffeModel(const std::string& deployFile, const std::string& modelFile, const std::vector<std::string>& outputs, unsigned int maxBatchSize, nvcaffeparser1::IPluginFactory* pluginFactory, IHostMemory *&gieModelStream);					    
		
		static void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
	
	private:
		class Logger : public ILogger			
		{
			void log(Severity severity, const char* msg) override
			{
			// suppress info-level messages
			if (severity != Severity::kINFO)
				std::cout << msg << std::endl;
			}
		} gLogger;

		
};
#endif
