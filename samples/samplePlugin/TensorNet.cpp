#include "TensorNet.h"

#define LOG_GIE "[LOG_GIE_ ]"

void tensorNet::importTrainedCaffeModel(const std::string& deployFile,	// name for caffe prototxt
					 const std::string& modelFile,					    // name for model 
					 const std::vector<std::string>& outputs,		    // network outputs
					 unsigned int maxBatchSize,						    // batch size - NB must be at least as large as the batch we want to run with)
					 nvcaffeparser1::IPluginFactory* pluginFactory,	    // factory for plugin layers
					 IHostMemory *&gieModelStream)					    // output stream for the GIE model
{
	std::cout<<LOG_GIE<<"Creo il Builder"<<std::endl;
	IBuilder* builder = createInferBuilder(gLogger);

	std::cout<<LOG_GIE<<"Creo il Modello ed il Parse"<<std::endl;
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();

	std::cout<<LOG_GIE<<"Aggiungo al Parser il Plugin"<<std::endl;
	parser->setPluginFactory(pluginFactory);

	bool fp16 = builder->platformHasFastFp16();
	if(fp16)
		std::cout<<LOG_GIE<<"bella storia c'è il suporto per fp16 ovvero dataType KHALF"std::endl;
	
	std::cout<<LOG_GIE<<"Mappo i blob di Caffe sui tensori parsando il modello"<<std::endl;
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
	if(initBuilderSize(builder, maxBatchSize, fp16){
		std::cout<<LOG_GIE<<"parametri ok"<<std::endl;
	}

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	gieModelStream = engine->serialize();

	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

bool initBuilderSize(IBuilder& builder, unsigned int maxBatchSize, bool fp16)
{
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);
	builder->setHalf2Mode(fp16);
	return true;
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME), 
		outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}

