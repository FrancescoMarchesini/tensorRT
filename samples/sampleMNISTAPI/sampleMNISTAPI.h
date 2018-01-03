#ifndef __SAMPLE_MNISTAPI_H__
#define __SAMPLE_MNISTAPI_H__
/**
 *	il seguente programmma produce ed esegue un modello di MNIST
 *	con la differenza che il modello e caricato manualmente e il
 *	network stesso costruito manualmente
 * */

#include "NvInfer.h"
#include "NvCaffeParser.h"
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

#define CHECK(status)					\
{							\
    if (status != 0)				\
    {						\
        std::cout << "Cuda failure: " << status;\
		abort();				\
	}						\
}

// stuff we know about the network and the input/output blobs
#define INPUT_H  28
#define INPUT_W  28
#define OUTPUT_SIZE  10

#define INPUT_BLOB_NAME  "data"
#define OUTPUT_BLOB_NAME  "prob"

using namespace nvinfer1;
using namespace nvcaffeparser1;


class GIEMnistAPI
{
	public:
		//Costruttore
		GIEMnistAPI();
		//distruttore
	    ~GIEMnistAPI();

		//macro funzione che fa tutto tramite le funzione che seguono
		void initAndLunch();

		//trovaFile
		std::string locateFile(const std::string& input);
		//leggi il file
		void readPGMFile(const std::string& fileName, uint8_t buffer[INPUT_H*INPUT_H]);
		//loadImage
		uint8_t* loadPGMFile();
		//plot DigitFile
		void plotPGMasAsci(uint8_t fileData[]);

		//sottrazione dell'iimagine corrente con il l'immagine media del modello e salvattagio nuova immagine
		float* subtractImage(uint8_t* currentImg);

		//funzione per caricare i pesi da un file costum, aka un parser
		std::map<std::string, Weights> loadWeights(const std::string file);
		
		//creazione dell'egine cuda usando direttamente le cuda API
		//configurazione della fase di building
		ICudaEngine * createMNISTEngine(unsigned int maxBatchSize, IBuilder *builder, DataType dt);
		//Creazione dell'engine e serializzane su file(PLAN)
		void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream);

		//fase di deploy 
		//ovvero l'inferenza in base al modello
		void doInference(IExecutionContext& context, float* input, float* output, int batchSize);
	
		//Plot del risultato sul terminale
		int getNum(){return num;};
		bool plotAsciResult();
		
	protected:
		// Logger for info/warning/errors
		class Logger : public ILogger			
		{
			void log(Severity severity, const char* msg) override
			{
				// suppress info-level messages
				if (severity != Severity::kINFO)
				std::cout << msg << std::endl;
			}
		} gLogger;

		//oggetto per inferenza runtime
		IRuntime* mRuntime;
		//oggeto per costruire l'engine
		ICudaEngine* mEngine;
		//constesto per far partire i kernel
		IExecutionContext* mContext;
		
		//vettore di output
		float prob[OUTPUT_SIZE];
		int num;
};
#endif
