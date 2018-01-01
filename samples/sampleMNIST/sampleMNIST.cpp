#include "sampleMNIST.h"
#include <cuda_runtime_api.h>

GIE:~GIE()
{
	if(engigne != NULL){
		engigne->destroy();
		engigne = NULL;
	}
	if(runtime != NULL){
		runtime->destroy();
		runtime = NULL;
	}
}

GIE::GIE(void)
{
	// create a GIE model from the caffe model and serialize it to a stream
    IHostMemory *gieModelStream{nullptr};

	//creazione dell'engigne a partire dal modello di caffe
   	caffeToGIEModel("mnist.prototxt", "mnist.caffemodel", std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, gieModelStream);

	// read a random digit file
	srand(unsigned(time(nullptr)));
	uint8_t fileData[INPUT_H*INPUT_W];
    num = rand() % 10;
	//copio il file caricato nella struttura dati fileData
	readPGMFile(std::to_string(num) + ".pgm", fileData);

	// print an ascii representation
	std::cout << "\n\n\n---------------------------" << "\n\n\n" << std::endl;
	for (int i = 0; i < INPUT_H*INPUT_W; i++)
		std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");

	// parse the mean file and 	subtract it from the image
	// carico il file della media e distruggo il parser
	ICaffeParser* parser = createCaffeParser();
	IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(locateFile("mnist_mean.binaryproto").c_str());
	parser->destroy();

	const float *meanData = reinterpret_cast<const float*>(meanBlob->getData());

	//faccio la sottrazioni dell'immagine correttene con quella della media
	float data[INPUT_H*INPUT_W];
	for (int i = 0; i < INPUT_H*INPUT_W; i++)
		data[i] = float(fileData[i])-meanData[i];

	meanBlob->destroy();

	// deserialize the engine 
	// deserizilizzazione dell'engigne non so cos'è
	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), nullptr);
    if (gieModelStream) gieModelStream->destroy();

	//creo l'esecuzion dell'enginge
	IExecutionContext *context = engine->createExecutionContext();

	// run inference
	//float prob[OUTPUT_SIZE];
	
	// e copio in prob il risultato
	doInference(*context, data, prob, 1);

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	// print a histogram of the output distribution
	/*std::cout << "\n\n";
    float val{0.0f};
    int idx{0};
	for (unsigned int i = 0; i < 10; i++)
    {
        val = std::max(val, prob[i]);
        if (val == prob[i]) idx = i;
		std::cout << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
    }
	std::cout << std::endl;

//	return (idx == num && val > 0.9f) ? EXIT_SUCCESS : EXIT_FAILURE;
*/
}

bool GIE::plot()
{
	// print a histogram of the output distribution
	std::cout << "\n\n";
    float val{0.0f};
    int idx{0};
	for (unsigned int i = 0; i < 10; i++)
    {
        val = std::max(val, prob[i]);
        if (val == prob[i]) idx = i;
		std::cout << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
    }
	std::cout << std::endl;

	return (idx == getNum() && val > 0.9f) ? EXIT_SUCCESS : EXIT_FAILURE;

}
//funzione per caricare il modello e i relativi file
std::string GIE::locateFile(const std::string& input)
{
	std::string file = "data/samples/mnist/" + input;
	struct stat info;
	int i, MAX_DEPTH = 10;
	for (i = 0; i < MAX_DEPTH && stat(file.c_str(), &info); i++)
		file = "../" + file;

    if (i == MAX_DEPTH)
    {
		file = std::string("data/mnist/") + input;
		//stat: funzione per rifotrnare informazioni riguardo al file di inpu
		for (i = 0; i < MAX_DEPTH && stat(file.c_str(), &info); i++)
			file = "../" + file;		
    }

	assert(i != MAX_DEPTH);
	std::cout<<"file: " << file << std::endl;
	return file;
}

// simple PGM (portable greyscale map) reader
// lettura del file grafici in formato PGM
// uint8_t : unsigned int di lunghezza 8 bit . l'array è 28 * 28 = 784 
void GIE::readPGMFile(const std::string& fileName,  uint8_t buffer[INPUT_H*INPUT_W])
{
	std::ifstream infile(locateFile(fileName), std::ifstream::binary);
	std::string magic, h, w, max;
	infile >> magic >> h >> w >> max;
	infile.seekg(1, infile.cur);
	infile.read(reinterpret_cast<char*>(buffer), INPUT_H*INPUT_W);
}

/**
 *Fase di costruzione "Build Fase"
 */
/**
 *nella fase di costruzione tensoRT prende la definizione del network, fa ottimizzazioni e generea il meccanismo di inferenza(inferece engine "GIE")
 *le ottimizzzioni sui leyer del  modello sono le segunti operazioni:
 *	eliminazione layer non usati
 *  funzione di operazioni su layer simili
 *  aggregazione di operazioni con parametri simili
 *  elisione delle concatezione tra i layer per percorsi pi brevi
 */
bool GIE::caffeToGIEModel(const std::string& deployFile,				// name for caffe prototxt
					 const std::string& modelFile,				// name for model 
					 const std::vector<std::string>& outputs,   // network outputs
					 unsigned int maxBatchSize,					// batch size - NB must be at least as large as the batch we want to run with)
					 IHostMemory *&gieModelStream)    // output buffer for the GIE model
{
	// create the builder
	// costruzione dell'oggetto tramite la classe logger
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	// creazione del network
	INetworkDefinition* network = builder->createNetwork();

	//funzioni di logger
	builder-setDebugSync(true);

	//funzioni di ottimizzazione
	builder->setMinFindIterations(3);
	builder->setAverageFindIterations(2);
	
	// popolazione del network tramite i parametri del modello caffe estratti tramite la lib parser
	ICaffeParser* parser = createCaffeParser();
	//qui avviene la mappatura tra i layer del modello caffe e i tensori di tensorRT
	const IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(deployFile).c_str(),
															  locateFile(modelFile).c_str(),
															  *network,
															  DataType::kFLOAT);

	//funzione per fare il check
	if(!IBlobNameToTensor){
		std::cout << "fallito a caricare il modello e mapparlo" << std::endl;
		return false;
	}

	// specify which tensors are outputs
	// il modello di caffe non dice quali sono i layer di output. Per questo dobbiamo mapparli manualmente: legge i network output e tramite la funzione find che restituiesce il nome del layer lo mappa con i tensori
	/*for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));
	*/
	const size_t num_ouputs = output.size();
	for(size_t n=0; n < num_ouputs; n++){
		//in uqesto modo posso avere gli tensor di output come argomento di ritono e fare un check
		nvinfer1::ITensor* tensor = blobNameToTensor->find(output[n].c_string());		

		if(!tensors){
			std::cout<< "fallito a mappare gli output: " << output[n].c_string() <<std::endl;
		}else{
			std::cout<< "trovato aoutput comaptibili: " << output[n].getName() <<std::endl;
		}

		network->markOutput(*tensor);
	}

	// Build the engine
	// Costuzione dell'engigne a partire dalla definizione del network
	// BatchSize : gradezza con la quale engihne è tuned
	std::cout << "configuro l'engigne cuda" << endl;
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(1 << 20);

	//creazione dell'engigne a partire dal network
	std::cout <<"costruisco l'engigne cuda"<< std::endl;
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	//funzione assert: comparazione con lo zero se è diverso da 0 da errore, ovvero non
	//si è costruito l'engigne per un qualche problema
	if(!engine){
		std::cout << "fallito a costruire lengigne" << endl;
		return false;
	}

	std::cout << "engigne costrutito corretamente" << std::endl;

	// we don't need the network any more, and we can destroy the parser
	// distruggo gli oggetti non piu necessari
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	gieModelStream = engine->serialize();
	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();

	return true;
}

/**
 *Fase di esecuzione "execution phase"
 *
 *
 *	esecuzione a run time dell'engigne
 *	le inferenze vengo fatte tramite gli input e ouput buffer sulla gpu
 */

void GIE::doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
	/**
	 * gli input dell'engigne sono array di puntatori agli input e output buffer sulla GPU
	 */

	const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	// nella fase di custruzione il network è stato settata con due "blobs" inpu e ouput, qindi check
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	// mappo gli input e output del modello sui tensori tramite il match dei nomi dei layer
	int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME), 
		outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

	// create GPU buffers and a stream
	// creazione dei rispettivi buffer di input e output, memoria sulla GPU, dandogli all'indirizio di memoria di ciscuno la rispettiva 
	// grandezza
	CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

	//creazione di un oggetto cuda stream per il passaggio dei dati: copia, inferenze, copia:
	//stream : A sequence of operations that execute in issue-order on the GPU
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	// 1: le informazioni sono copiate sulla  dalla CPU alla GPU
	CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	// 2: avvienei l'inferenza statica all'interno del contesto, ovvero la classe esecuzione figlia del cuda engigne
	context.enqueue(batchSize, buffers, stream, nullptr);
	context.setDebugSync(true);
	// 3: le informazioni vengo copiate dalla GPU alla CPU
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
	// 4: lo stream viene sincronizzato
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	// distruggi tutto
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}
