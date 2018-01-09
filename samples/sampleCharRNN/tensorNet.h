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

#include "plugin.h"

// These mappings came from training with tensorflow 0.12.1
// and emitting the word to id and id to word mappings from
// the checkpoint data after loading it.
// The only difference is that in the data set that was used,
static std::map<char, int> char_to_id{{'#', 40},
    { '$', 31}, { '\'', 28}, { '&', 35}, { '*', 49},
    { '-', 32}, { '/', 48}, { '.', 27}, { '1', 37},
    { '0', 36}, { '3', 39}, { '2', 41}, { '5', 43},
    { '4', 47}, { '7', 45}, { '6', 46}, { '9', 38},
    { '8', 42}, { '<', 22}, { '>', 23}, { '\0', 24},
    { 'N', 26}, { '\\', 44}, { ' ', 0}, { 'a', 3},
    { 'c', 13}, { 'b', 20}, { 'e', 1}, { 'd', 12},
    { 'g', 18}, { 'f', 15}, { 'i', 6}, { 'h', 9},
    { 'k', 17}, { 'j', 30}, { 'm', 14}, { 'l', 10},
    { 'o', 5}, { 'n', 4}, { 'q', 33}, { 'p', 16},
    { 's', 7}, { 'r', 8}, { 'u', 11}, { 't', 2},
    { 'w', 21}, { 'v', 25}, { 'y', 19}, { 'x', 29},
    { 'z', 34}
};

// A mapping from index to character.
static std::vector<char> id_to_char{{' ', 'e', 't', 'a',
    'n', 'o', 'i', 's', 'r', 'h', 'l', 'u', 'd', 'c',
    'm', 'f', 'p', 'k', 'g', 'y', 'b', 'w', '<', '>',
    '\0', 'v', 'N', '.', '\'', 'x', 'j', '$', '-', 'q',
    'z', '&', '0', '1', '9', '3', '#', '2', '8', '5',
    '\\', '7', '6', '4', '/', '*'}};

// Information describing the network
#define LAYER_COUNT  2
#define BATCH_SIZE  1
#define HIDDEN_SIZE  512
#define  SEQ_SIZE  1
#define DATA_SIZE 512
#define OUTPUT_SIZE 50

#define INPUT_BLOB_NAME "data"
#define HIDDEN_IN_BLOB_NAME "hiddenIn"
#define CELL_IN_BLOB_NAME "cellIn"
#define OUTPUT_BLOB_NAME "prob"
#define HIDDEN_OUT_BLOB_NAME "hiddenOut"
#define CELL_OUT_BLOB_NAME "cellOut"


using namespace nvinfer1;
/////////////////////////////
//TensorNet Main
/////////////////////////////
class tensorNet
{
	public:
		//////////////////////////////////////////////////////////////////////////////////////
		//Caricamento del Network e costruzione del Mdello : Building Phase
		/////////////////////////////////////////////////////////////////////////////////////
		//funzione per convertire i pesi di tensorFlow in TensoRT
		Weights convertRNNWeights(Weights input);
		//funzione per convettire i bias tensflow in tensoRT
		Weights convertRNNBias(Weights input);
		//Funzione per far qualcosa da tesnflow a tensorRT
		Weights transposeFCWeights(Weights input);
		//Creazione del modello tensorRT da tensorFlow
		void APIToModel(std::map<std::string, Weights> &weightMap, IHostMemory **modelStream);
		/////////////////////////////////////////////////////////////////////////////////////
		
		
		//////////////////////////////////////////////////////////////////////////////////////
		//Creazione dell'engine e inferenza : Deploy Phase
		/////////////////////////////////////////////////////////////////////////////////////
		void stepOnce(float **data, void **buffers, int *sizes, int *indices, int numBindings, cudaStream_t &stream, IExecutionContext &context);
		bool doInference(IExecutionContext& context, std::string input, std::string expected, std::map<std::string, Weights> &weightMap);
		/////////////////////////////////////////////////////////////////////////////////////
		IRuntime* mRuntime;			 //esecuzione dell'inferenza
		ICudaEngine* mEngine;		 //engine
		IExecutionContext* mContext; //eseguzione kernel

		//oggetto per serializare o deserializzare il gie da e su file
		IHostMemory *gieModelStream{nullptr};
		//istanzione l'oggetto per layer creato
		PluginFactory pluginfactory;
	protected:
		tensorNet()
		{
			 mRuntime=NULL;			 //esecuzione dell'inferenza
			 mEngine=NULL;		 //engine
			 mContext=NULL; //eseguzione kernel
		}
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


};
#endif
