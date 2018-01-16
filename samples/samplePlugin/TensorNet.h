#ifndef __TENSORNET_H__
#define __TENSORNET_H__

#include "addPlugin.h"
#include <iostream>

using namespace nvinfer1;
using namespace nvcaffeparser1;

class tensorNet
{
	public:
		tensorNet(int input_h, int input_w, int output_size, std::string input_blob_name, std::string output_blob_name);
		void importTrainedCaffeModel(const std::string& deployFile, const std::string& modelFile, const std::vector<std::string>& outputs, unsigned int maxBatchSize, nvcaffeparser1::IPluginFactory* pluginFactory, IHostMemory *&gieModelStream);					    
		void doInference(IExecutionContext& context, float* input, float* output, int batchSize);
	
		bool initBuilderSize(IBuilder* builder, unsigned int maxBatchSize, bool fp16);
		
		class Logger : public ILogger			
		{
			void log(Severity severity, const char* msg) override
			{
			// suppress info-level messages
			if (severity != Severity::kINFO)
				std::cout << msg << std::endl;
			}
		} gLogger;


	private:	
		/*
		 *	struttura che rappresenta gli input e output del sistema
		 * */
		struct IO
		{
			int input_h;
			int input_w;
			int output_size;
			const char* input_blob_name;
			const char* output_blob_name;
		}system;
};
#endif
