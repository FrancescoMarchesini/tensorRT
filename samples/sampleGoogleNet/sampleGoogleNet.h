#ifndef __SAMPLE_GOOGLE_NET_H_
#define __SAMPLE_GOOGLE_NET_H_


#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"


using namespace nvinfer1;
using namespace nvcaffeparser1;

#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}

//varibili che sappiamo a priori dal modello stesso
//BATCH_SIZE : numeri di esempio, traing sampels, che entrano per volta nel modello, 
//			   maggiore è il numero maggiore è il quantitativo di memoria richiesta
//
//se il batch size + >1 il modo piu veloce di eseguire l'enginge è half2mode
#define BATCH_SIZE 4
#define TIMING_ITERATIONS 1000
#define INPUT_BLOB_NAME "data"
#define OUTPUT_BLOB_NAME "prob"

class GIEgoogleNET
{
	public:
		GIEgoogleNET();
		bool lunch();
		std::string locateFile(const std::string& input);
		//differeisce dai sample precendeti poichè la preciosne + 16 bit
		bool caffeToGIEModel(const std::string& deployFile,
					 const std::string& modelFile,		 
					 const std::vector<std::string>& outputs,
					 unsigned int maxBatchSize,				
					 IHostMemory *&gieModelStream);
		//differisce dal sample precedente poichè l'inferenza è di un tempo contrallato TIME_ITERATIONS
		bool timeInference(ICudaEngine* engine, int batchsize); 
	protected:
		// Logger for GIE info/warning/errors, necessario per la costruzione dell'engigne
		class Logger : public ILogger			
		{
			void log(Severity severity, const char* msg) override
			{
				if (severity!=Severity::kINFO)
				std::cout << msg << std::endl;
			}
		} gLogger;
		
		//profilazione di un network per ognuno dei suoi layer
		struct Profiler : public IProfiler
		{
			typedef std::pair<std::string, float> Record;
			std::vector<Record> mProfile;

			virtual void reportLayerTime(const char* layerName, float ms)
			{
				auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
				if (record == mProfile.end())
					mProfile.push_back(std::make_pair(layerName, ms));
				else
					record->second += ms;
			}

			void printLayerTimes()
			{
				float totalTime = 0;
				for (size_t i = 0; i < mProfile.size(); i++)
				{
					printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / TIMING_ITERATIONS);
					totalTime += mProfile[i].second;
				}
				printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
			}
		} gProfiler;
		
	IRuntime* infer;
	ICudaEngine* engine;	
};
#endif
