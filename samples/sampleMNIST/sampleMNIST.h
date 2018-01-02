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
//
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

#define INPUT_W 28
#define INPUT_H 28
#define OUTPUT_SIZE 10
#define INPUT_BLOB_NAME "data"
#define OUTPUT_BLOB_NAME "prob"



class GIE
{
public:
	/**
	 * Distruggi
	 */
	//virtual ~GIE();

	//Costruttore
	GIE();
	
	

	bool init();
	std::string locateFile(const std::string& input);
	
	//Funzione per leggere i file in formato PGM
	void readPGMFile(const std::string& fileName,  uint8_t buffer[]);
	
	/**
	 * fase di costruzione 
	 *	*file di archiettura del network(deploy.prototxt)
	 *	*pesi (net.caffemodel)
	 *	*a label che determina un nome per ogni classe di outup possobile
	 */
	bool caffeToGIEModel(const std::string& deployFile,				
					     const std::string& modelFile,				 
					     const std::vector<std::string>& outputs,   
					     unsigned int maxBatchSize,					 
						 IHostMemory *&gieModelStream);		
	
	/**
	 * Fase di Deploy
	 */
	bool doInference(IExecutionContext& context, float* input, float* output, int batchSize);
	
	bool plot();
	int getNum(){return num;}

	
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
	
	/*variabili membro*/
	nvinfer1::IRuntime *runtime;  //oggetto per l'inferenenza runtime in base al modello
	nvinfer1::ICudaEngine *engine; //oggetto che rappresenta l'enginge
	nvinfer1::IExecutionContext *context; //constesto

	float prob[OUTPUT_SIZE];
	int num;
};

#endif
