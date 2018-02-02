#include <fstream>

#include "NvCaffeParser.h"
#include "InferenceEngine.h"

#include <iostream>
#include <sstream>
using namespace nvinfer1;
using namespace nvcaffeparser1;
using std::string;
#define LOG_GIE "[CULO] "
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
}gLogger;


InferenceEngine::InferenceEngine()
{

}

InferenceEngine(const std::string& model_file,
                const std::string& trained_file,
			    const std::vector<std::string>& output,
				unsigned int bathSize);

{
	std::cout<<LOG_GIE<<"creo il builder"<<std::endl;
	IBuilder* builder = createInferBuilder(gLogger);

	std::cout<<LOG_GIE<<"creo il network dal builder"<std::endl;
    INetworkDefinition* network = builder->createNetwork();

	std::cout<<LOG_GIE<<"creo il parser"<std::endl;
    ICaffeParser* parser = createCaffeParser();


	//////////////////////////////////////////////////////////////
	//qui devo mettere il plugin per il parse del modello costum//
	//														    //
	//////////////////////////////////////////////////////////////
	

    auto blob_name_to_tensor = parser->parse(model_file.c_str(),
                                            trained_file.c_str(),
                                            *network,
                                            DataType::kFLOAT);
    blob_name_to_tensor;

    // specify which tensors are outputs
    network->markOutput(*blob_name_to_tensor->find("prob"));

    // Build the engine
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(1 << 30);

    engine_ = builder->buildCudaEngine(*network);
    engine_;

    network->destroy();
    builder->destroy();
}

InferenceEngine::~InferenceEngine()
{
    engine_->destroy();
}

void InferenceEngine::Import(const string& plan_file) 
{
	std::stringstream gieModelStream;
	gieModelStream.seekg(0, gieModelStream.beg);

	std::cout<<LOG_GIE<<"cerco file plane del modello"<<std::endl;
	char cache_path[512];
	//sprintf(cache_path, "%s.tensorcache", "plane");
	sprintf(cache_path, plan_file.c_str());
	std::cout<<LOG_GIE<<"apro il file "<<cache_path<<std::endl;

	std::ifstream cache( cache_path );
	
	if(cache)
	{
		std::cout<<LOG_GIE<<"file plane trovato carico modello.."<<std::endl;
		gieModelStream << cache.rdbuf();
		cache.close();

		std::cout<<LOG_GIE<<"costruisco l'engine a partire dal plane"<<std::endl;
		
		std::cout<<LOG_GIE<<"Creo il Builder"<<std::endl;
		nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
		if(builder != NULL){
			bool fp16 = builder->platformHasFastFp16();
			if(fp16) std::cout<<LOG_GIE<<"bella storia c'è il suporto per fp16 ovvero dataType KHALF"<<std::endl;
			builder->destroy();
			std::cout<<LOG_GIE<<"distruggo il builder"<<std::endl;
		}
	}
	
	std::cout<<LOG_GIE<<"creo il contesto per l'esecuzione run time"<<std::endl;
	nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(gLogger);

	//determina la fine del file
	gieModelStream.seekg(0, std::ios::end);
	//determino la lunghezza del file
	const int modelSize = gieModelStream.tellg();
	std::cout<<LOG_GIE<<"grandezza file plane byte = "<<modelSize<<std::endl;
	//determino la fine del file
	gieModelStream.seekg(0, std::ios::beg);

	std::cout<<LOG_GIE<<"alloco la memoria per deserializzara il modello"<<std::endl;
	void* modelMem = malloc(modelSize);

	if(!modelMem){
		std::cout<<LOG_GIE<<"azz fallito ad allocare la memoria"<<std::endl;
	}

	std::cout<<LOG_GIE<<"leggo il file"<<std::endl;
	gieModelStream.read((char*)modelMem, modelSize);
	engine_ = infer->deserializeCudaEngine(modelMem, modelSize, NULL); 
	free(modelMem);

	if(!engine_){
		std::cout<<LOG_GIE<<"Fallito a creare l'engine dal file"<<std::endl;
		exit(true);
	}else{
		std::cout<<LOG_GIE<<"Bella storia possiamo fare inferenze :) "<<std::endl;
	}
}

void InferenceEngine::Export(const string& plan_file) const 
{
	std::cout<<LOG_GIE<<"serializzo il modello su file"<<std::endl;
	std::ofstream gieModelStream(plan_file.c_str(), std::ofstream::binary); 	
	nvinfer1::IHostMemory* serMem = engine_->serialize();
	gieModelStream.write((const char*)serMem->data(), serMem->size());
}
