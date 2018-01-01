#ifndef __SAMPLE_MNIST_H__
#define __SAMPLE_MNIST_H__

#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include <time.h>

//utilizzo della libreria NvInfer per costruire il motere di inferenze
#include "NvInfer.h"
// il motore è basata su un modello di caffe il seguende headers è un parser
//dei file di caffe
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

static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE= 10;
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME= "prob";



class GIE
{
public:
	/**
	 * Distruggi
	 */
	virtual ~GIE();

	std::string locateFile(const std::string& input);
	void readPGMFile(const std::string& fileName,  uint8_t buffer[INPUT_H*INPUT_W]);
	void caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					     const std::string& modelFile,				// name for model 
					     const std::vector<std::string>& outputs,   // network outputs
					     unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
						 IHostMemory *&gieModelStream);		// output buffer for the GIE model
	void doInference(IExecutionContext& context, float* input, float* output, int batchSize);
	bool plot();
	int getNum(){return num;}

protected:
	/**
	 *	Costruttore
	 */
	GIE();
// Logger for GIE info/warning/errors
	// il logger in tensorRT è necessario
	class Logger : public ILogger			
	{
		void log(Severity severity, const char* msg) override
		{
			// suppress info-level messages
			if (severity != Severity::kINFO)
				std::cout << msg << std::endl;
		}
	} gLogger;
	
protected:
	/*variabili membro*/
	nvinfer1::IRuntime *runtime;  //oggetto per l'inferenenza runtime in base al modello
	nvinfer1::ICudaEngine *engine; //oggetto che rappresenta l'enginge
	nvinfer1::IExecutionContext *context; //constesto

	float prob[OUTPUT_SIZE];
	int num;
};

#endif
