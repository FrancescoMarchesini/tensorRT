#include "InferenceEngine.h"
#include <fstream>
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
	

	/*std::cout<<LOG_GIE<<"definisco che il mio layer di output: softmax"<<std::endl;
	for(int i=0; i<engine_->getNbBindings(); i++)
	{
		if(engine_->bindingIsInput(i) == false)
		{
				std::cout<<engine_->getBindingName(i)<<std::endl;		
		}
	}*/

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
	std::cout<<LOG_GIE<<"serializzo il modello"<<std::endl;
	std::ofstream gieModelStream(plan_file.c_str(), std::ofstream::binary); 	
	nvinfer1::IHostMemory* serMem = engine_->serialize();
	gieModelStream.write((const char*)serMem->data(), serMem->size());
}
/////////////////////////////////////////////////////////////////////////////////////
//main//////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	motoreDiInferenza* motore = new motoreDiInferenza(argv[1], argv[2]);
	motore->Esporta(argv[3]);
	return 0;
}
