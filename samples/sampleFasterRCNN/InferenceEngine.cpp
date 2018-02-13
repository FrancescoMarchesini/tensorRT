#include "InferenceEngine.h"

using namespace::nvinfer1;
using namespace::nvcaffeparser1;

#define LOG_GIE "[GIE] "
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    };
}gLogger;

InferenceEngine::InferenceEngine(){
    
	std::cout<<LOG_GIE<<"Dai Dai Dai...\n"<<std::endl;
}

InferenceEngine::InferenceEngine(const std::string& model_file,
                const std::string& trained_file,
			    const std::vector<std::string>& output,
				unsigned int bathSize)

{
    _deploy = model_file.c_str();
    _model = trained_file.c_str();
    _output = output;
    _batchSize = bathSize;
    engine_ = NULL;

    printf("%sParametri del network\n", LOG_GIE);
    printf("%smodel_file: %s\n", LOG_GIE, _deploy);
    printf("%strained_file: %s\n", LOG_GIE, _model);
    for(int i=0; i<_output.size(); i++)
        printf("%soutput[%d]: %s\n",LOG_GIE, i, _output[i].c_str());
    printf("%sbatchSize: %d\n",LOG_GIE, _batchSize);

}

InferenceEngine::~InferenceEngine()
{
    engine_->destroy();
}

bool InferenceEngine::loadFastRCNN()
{
	std::cout<<LOG_GIE<<"creo il builder"<<std::endl;
	IBuilder* builder = createInferBuilder(gLogger);

	std::cout<<LOG_GIE<<"creo il network dal builder"<<std::endl;
    INetworkDefinition* network = builder->createNetwork();

	std::cout<<LOG_GIE<<"creo il parser"<<std::endl;
    ICaffeParser* parser = createCaffeParser();

	std::cout<<LOG_GIE<<"creo il plugin"<<std::endl;
	parser->setPluginFactory(&pluginFactory);

    printf("%sDetrmino se ce la fp16\n", LOG_GIE);
   bool fp16 = builder->platformHasFastFp16();
    if(fp16) printf("%sYes Baby", LOG_GIE);
    DataType modelDataType = fp16 ? DataType::kHALF : DataType::kFLOAT;

	std::cout<<LOG_GIE<<"Parsing del modello"<<std::endl;
    auto blob_name_to_tensor = parser->parse(_deploy, _model, *network, modelDataType);
    
    printf("%sHo finito di parsare il modello\n", LOG_GIE);
    
    for(auto&s : _output){
		printf("%stensore di output: %s\n", LOG_GIE, s.c_str());
		network->markOutput(*blob_name_to_tensor->find(s.c_str()));
    }	

    // Build the engine
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(1 << 30);

    if(fp16) builder->setHalf2Mode(true);

	printf("%scostruisco l'engine\n", LOG_GIE);
    engine_ = builder->buildCudaEngine(*network);

	network->destroy();
    parser->destroy();

	if(saveModelToPlane("tensorPlan"))
		printf("%sYEs baby\n", LOG_GIE);
  
    builder->destroy();
    pluginFactory.destroyPlugin();

    return true;
}

bool InferenceEngine::loadPlane(const std::string& plan_file)
{
	std::stringstream gieModelStream;
	gieModelStream.seekg(0, gieModelStream.beg);

	std::cout<<LOG_GIE<<"cerco file plane del modello"<<std::endl;
	char cache_path[512];
	//sprintf(cache_path, "%s.tensorcache", "plane");
	sprintf(cache_path, plan_file.c_str());
	std::cout<<LOG_GIE<<"il file cache esiste?"<<cache_path<<std::endl;
	
    std::ifstream cache(cache_path);

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
//	void* modelMem = malloc(modelSize);
    char* modelMem = new char[modelSize];

	if(!modelMem)
    {
		std::cout<<LOG_GIE<<"azz fallito ad allocare la memoria"<<std::endl;
	}

	std::cout<<LOG_GIE<<"leggo il file"<<std::endl;
	gieModelStream.read(modelMem, modelSize);
    engine_ = infer->deserializeCudaEngine(modelMem, modelSize, &pluginFactory);

    free(modelMem);

    if(!engine_){
		std::cout<<LOG_GIE<<"Fallito a creare l'engine dal file"<<std::endl;
		exit(true);
	}else{
		std::cout<<LOG_GIE<<"Bella storia possiamo fare inferenze :) "<<std::endl;
	}

	return true;
}

bool InferenceEngine::saveModelToPlane(const std::string& plan_file)  
{
	std::cout<<LOG_GIE<<"serializzo il modello su inella moria dell'host"<<std::endl;
	std::ofstream gieModelStream(plan_file.c_str(), std::ofstream::binary); 	
	serMem = engine_->serialize();
	gieModelStream.write((const char*)serMem->data(), serMem->size());
	return(true);
}

bool InferenceEngine::doInference()
{
    printf("%sadesso icominciomao a divertirci", LOG_GIE);
    for(int i=0; i<engine_->getNbBindings(); i++)
    {
        printf("%s%s = %d\n", LOG_GIE, engine_->getBindingName(i),  engine_->getBindingDimensions(i));        
    }

    return true;
    
}
