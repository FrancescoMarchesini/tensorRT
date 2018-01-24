#include "InferenceEngine.h"

#include <fstream>
#include <iostream>
#include <sstream>

motoreDiInferenza::motoreDiInferenza()
{

}

motoreDiInferenza::motoreDiInferenza(
		const std::string& model_file, 
		const std::string& weights_file)
{
	std::cout<<LOG_GIE<<"Creo il Builder"<<std::endl;
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);

	std::cout<<LOG_GIE<<"Creo il Modello tramite il parser"<<std::endl;
	nvinfer1::INetworkDefinition* network = builder->createNetwork();

	std::cout<<LOG_GIE<<"parso il modello"<<std::endl;
	nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();

	bool fp16 = builder->platformHasFastFp16();
	if(fp16)
		std::cout<<LOG_GIE<<"bella storia c'è il suporto per fp16 ovvero dataType KHALF"<<std::endl;

	std::cout<<LOG_GIE<<"Mappo i blob di Caffe sui tensori parsando il file caffe"<<std::endl;
	const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(model_file.c_str(),
															  weights_file.c_str(),
															  *network,
															  fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT);
	
	if(blobNameToTensor)
		std::cout<<LOG_GIE<<"Modello parsato correttamente bella storia abbiamo i tensori"<<std::endl;
	
	network->markOutput(*blobNameToTensor->find("prob"));

	std::cout<<LOG_GIE<<"costruisco l'engine"<<std::endl;
	builder->setMaxBatchSize(1);
	builder->setMaxWorkspaceSize(1 << 30);
	builder->setHalf2Mode(fp16);
	engine_ = builder->buildCudaEngine(*network);

	if(engine_)
	{	
		std::cout<<LOG_GIE<<"Motore Costruito Corretamente"<<std::endl;
	}

	std::cout<<LOG_GIE<<"distruggo builder e network"<<std::endl;
	network->destroy();
	builder->destroy();
}


motoreDiInferenza::~motoreDiInferenza()
{
	std::cout<<LOG_GIE<<"distruggo l'oggetto motoreInferenza"<<std::endl;
	engine_->destroy();
}

void motoreDiInferenza::Esporta(const std::string& plan_file) 
{
	std::cout<<LOG_GIE<<"serializzo il modello su file"<<std::endl;
	std::ofstream gieModelStream(plan_file.c_str(), std::ofstream::binary); 	
	nvinfer1::IHostMemory* serMem = engine_->serialize();
	gieModelStream.write((const char*)serMem->data(), serMem->size());
}

void motoreDiInferenza::Importa(const std::string& plan_file)
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
/////////////////////////////////////////////////////////////////////////////////////
//main//////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	motoreDiInferenza* motore = new motoreDiInferenza(); //argv[1], argv[2]);
//	motore->Esporta(argv[3]);
	motore->Importa(argv[3]);
	return 0;
}
