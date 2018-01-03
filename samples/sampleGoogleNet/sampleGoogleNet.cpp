#include "sampleGoogleNet.h"

std::string GIEgoogleNET::locateFile(const std::string& input)
{
	std::string file = "data/samples/googlenet/" + input;
	struct stat info;
	int i, MAX_DEPTH = 10;
	for (i = 0; i < MAX_DEPTH && stat(file.c_str(), &info); i++)
		file = "../" + file;

    if (i == MAX_DEPTH)
    {
		file = std::string("data/googlenet/") + input;
		for (i = 0; i < MAX_DEPTH && stat(file.c_str(), &info); i++)
			file = "../" + file;		
    }

	assert(i != MAX_DEPTH);

	return file;
}

bool GIEgoogleNET::caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					 const std::string& modelFile,				// name for model 
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
					 IHostMemory *&gieModelStream)
{
	// create API root class - must span the lifetime of the engine usage
	IBuilder* builder = createInferBuilder(gLogger);
	INetworkDefinition* network = builder->createNetwork();

	// parse the caffe model to populate the network, then set the outputs
	ICaffeParser* parser = createCaffeParser();

	//determino se la piattaforma permette le operazioni aritmetiche e tensori con 16 al posto di 32 o 64
	bool useFp16 = builder->platformHasFastFp16();

	if(useFp16){
		std::cout << "ok la piattoaforma supporta i calcoli a 16 bit" << std::endl;
	}

	//determino con un condizionale quale tipo di precisione usare KHALT(16 bit) o KFLOAt(32)
	//queste configurazioni sono deteminte sia dalla piattaforma che dal modello (pesi a 32 bit float o 16 bit float)
	DataType modelDataType = useFp16 ? DataType::kHALF : DataType::kFLOAT; // create a 16-bit model if it's natively supported
	const IBlobNameToTensor *blobNameToTensor =
		parser->parse(locateFile(deployFile).c_str(),				// caffe deploy file
								 locateFile(modelFile).c_str(),		// caffe model file
								 *network,							// network definition that the parser will populate
								 modelDataType);

	if(!blobNameToTensor){
		std::cout<<"errato a parsare il modello"<<std::endl;
	}else{
		std::cout<<"modello parsato correttamente"<<std::endl;
	}
	//assert(blobNameToTensor != nullptr);
	
	// the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate	
	for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(16 << 20);

	// set up the network for paired-fp16 format if available
	// se posso usare i pesi con precisione 16 bit tale opzione ottimizza i calcoli ovvero
	// intervalla i tensori a 16 bit di distanza ???
	if(useFp16)
		builder->setHalf2Mode(true);

	//Costruisco l'engine
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	if(!engine){
		std::cout<<"fallito a costruire l engine"<<std::endl;
	}else{
		std::cout<<"engine costruito"<<std::endl;
	}
	//assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	// ovvero metto in forma binaria salvato su file(plan) l'intero engine con i relativi pesi 
	gieModelStream = engine->serialize();
	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();

	return true;
}

bool GIEgoogleNET::timeInference(ICudaEngine* engine, int batchSize)
{
	// input and output buffer pointers that we pass to the engine - the engine requires exactly ICudaEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	if(engine->getNbBindings() == 2){
		std::cout<<"Numero di tensori input e output corretti prosegue" << std::endl;
	}else{
		std::cout<<"errore nel settagio input output"<<std::endl;
		return false;
	}
	//alloco i buffer
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than ICudaEngine::getNbBindings()
	int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME), 
		outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

	// allocate GPU buffers
	// l'oggetto DimsCHW va a calcolare la dimensione del tensor di inpput e output in questo caso
	// il dato a un solo canale e due varibili per la dimensione spaziale
	DimsCHW inputDims = static_cast<DimsCHW&&>(engine->getBindingDimensions(inputIndex)), 
			outputDims = static_cast<DimsCHW&&>(engine->getBindingDimensions(outputIndex));
	
	size_t inputSize = batchSize * inputDims.c() * inputDims.h() * inputDims.w() * sizeof(float);
	size_t outputSize = batchSize * outputDims.c() * outputDims.h() * outputDims.w() * sizeof(float);

	//alloco le memorie per la computazione
	CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
	CHECK(cudaMalloc(&buffers[outputIndex], outputSize));

	//creo il contesto di esecuzione
	IExecutionContext* context = engine->createExecutionContext();
	context->setProfiler(&gProfiler);

	// zero the input buffer
	CHECK(cudaMemset(buffers[inputIndex], 0, inputSize));

	//faccio l'inferenza per un tempo controllato
	for (int i = 0; i < TIMING_ITERATIONS;i++)
		context->execute(batchSize, buffers);

	// release the context and buffers
	context->destroy();
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));

	return true;
}


bool GIEgoogleNET::lunch()
{
	std::cout << "Building and running a GPU inference engine for GoogleNet, N=4..." << std::endl;

	// parse the caffe model and the mean file
    IHostMemory *gieModelStream{nullptr};
	
	if(caffeToGIEModel("googlenet.prototxt", "googlenet.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, BATCH_SIZE, gieModelStream)){
		std::cout<<"modello parsata e engine costruito corretamente" <<std::endl;
	}

	// create an engine
	infer = createInferRuntime(gLogger);
	engine = infer->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), nullptr);

        printf("Bindings after deserializing:\n"); 
        for (int bi = 0; bi < engine->getNbBindings(); bi++) { 
               if (engine->bindingIsInput(bi) == true) { 
        printf("Binding %d (%s): Input.\n",  bi, engine->getBindingName(bi)); 
               } else { 
        printf("Binding %d (%s): Output.\n", bi, engine->getBindingName(bi)); 
               } 
           } 

	// run inference with null data to time network performance
	if(timeInference(engine, BATCH_SIZE)){
		std::cout<<"inferenza eseguita corretamente"<<std::endl;
	}

	engine->destroy();
	infer->destroy();

	gProfiler.printLayerTimes();

	std::cout << "Done." << std::endl;

	return true;
}

GIEgoogleNET::GIEgoogleNET(void)
{
	infer = NULL;
	engine = NULL;
}
