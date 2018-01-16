#include "TensorNet.h"

#define LOG_GIE "[LOG_GIE] "

tensorNet::tensorNet(int input_h, int input_w, int output_size, std::string input_blob_name, std::string output_blob_name)
{
	system.input_h = input_h;
	system.input_w = input_w;
	system.output_size = output_size;
	system.input_blob_name = input_blob_name.c_str();
	system.output_blob_name = output_blob_name.c_str();

	runtime = NULL;
	engine = NULL;
}

void tensorNet::importTrainedCaffeModel(const std::string& deployFile,	// name for caffe prototxt
					 const std::string& modelFile,					    // name for model 
					 const std::vector<std::string>& outputs,		    // network outputs
					 unsigned int maxBatchSize)						    // batch size - NB must be at least as large as the batch we want to run with)
{
	std::cout<<LOG_GIE<<"Creo il Builder"<<std::endl;
	IBuilder* builder = createInferBuilder(gLogger);

	std::cout<<LOG_GIE<<"Creo il Modello tramite il parser"<<std::endl;
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();

	std::cout<<LOG_GIE<<"Aggiungo al Parser il Plugin"<<std::endl;
	parser->setPluginFactory(&pluginFactory);

	bool fp16 = builder->platformHasFastFp16();
	if(fp16)
		std::cout<<LOG_GIE<<"bella storia c'è il suporto per fp16 ovvero dataType KHALF"<<std::endl;
	
	std::cout<<LOG_GIE<<"Mappo i blob di Caffe sui tensori parsando il file caffe"<<std::endl;
	const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile).c_str(),
															  locateFile(modelFile).c_str(),
															  *network,
															  fp16 ? DataType::kHALF : DataType::kFLOAT);

	if(blobNameToTensor)
		std::cout<<LOG_GIE<<"bella storia abbiamo i tensori"<<std::endl;

	for (auto& s : outputs){
		std::cout<<LOG_GIE<<"mappo gli output con i tensori di output: "<<s<<std::endl;
		network->markOutput(*blobNameToTensor->find(s.c_str()));
	}

	// Build the engine
	if(initBuilderSize(builder, maxBatchSize, fp16)){
		std::cout<<LOG_GIE<<"parametri ok"<<std::endl;
	}

	std::cout<<LOG_GIE<<"costrusco l'engine sul network"<<std::endl;
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	std::cout<<LOG_GIE<<"salvo su file l'engine: serializzazione"<<std::endl;
	gieModelStream = engine->serialize();

	std::cout<<LOG_GIE<<"distruggo engine e builder"<<std::endl;
	engine->destroy();
	builder->destroy();
	pluginFactory.destroyPlugin();
	shutdownProtobufLibrary();
}

bool tensorNet::initBuilderSize(IBuilder* builder, unsigned int maxBatchSize, bool fp16)
{
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);
	builder->setHalf2Mode(fp16);
	return true;
}

void tensorNet::doInference(float* input, float* output, int batchSize)
{
	std::cout<<LOG_GIE<<"creo il meccanismo di inferenza runtime"<<std::endl;
	runtime = createInferRuntime(gLogger);

	std::cout<<LOG_GIE<<"Creo l'engine a partire dal runtime serializzato"<<std::endl;
	engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);


	std::cout<<LOG_GIE<<"Mappo gli input e output del sistema"<<std::endl;
	int inputIndex;
	int outputIndex;	
	for(int i=0; i<engine->getNbBindings(); i++){
		if(engine->bindingIsInput(i) == true){
			std::cout<<LOG_GIE<<"Bindind  input : "<< i <<" "<<engine->getBindingName(i)<<std::endl;
			inputIndex = i;
		}else{
			std::cout<<LOG_GIE<<"Bindind output :"<< i <<" "<<engine->getBindingName(i)<<std::endl;
			outputIndex = i;
		}
	}
	
	std::cout<<LOG_GIE<<"Creo il contesto per l'esecuzione dei kernel"<<std::endl;
	IExecutionContext *context = engine->createExecutionContext();
	
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine->getNbBindings() == 2);
	void* buffers[engine->getNbBindings()];


	// create GPU buffers and a stream
	CHECK_CUDA(cudaMalloc(&buffers[inputIndex], batchSize * system.input_h * system.input_w * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&buffers[outputIndex], batchSize * system.output_size * sizeof(float)));

	cudaStream_t stream;
	CHECK_CUDA(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK_CUDA(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * system.input_h * system.input_w  * sizeof(float), cudaMemcpyHostToDevice, stream));

	/*
	 *la funzione enqueue serve per esegure i layer
	 * */
	context->enqueue(batchSize, buffers, stream, nullptr);
	
	CHECK_CUDA(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * system.output_size *sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK_CUDA(cudaFree(buffers[inputIndex]));
	CHECK_CUDA(cudaFree(buffers[outputIndex]));
	
	std::cout<<LOG_GIE<<"distruggo tutti gli oggetti"<<std::endl;
	context->destroy();
	engine->destroy();
	runtime->destroy();
	pluginFactory.destroyPlugin();
}

void tensorNet::plotClassification(float* output)
{
	for (unsigned int i = 0; i < 10; i++){
		std::cout <<"[CLASSIFICAZIONE] "<< i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*') << std::endl;;
	}

}
