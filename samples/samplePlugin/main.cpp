#include "TensorNet.h"
#include "addPlugin.h"
#include "utils.h"
#include <cmath>
#include <time.h>

#define INPUT_H  28
#define INPUT_W  28
#define OUTPUT_SIZE 10

#define INPUT_BLOB_NAME "data"
#define OUTPUT_BLOB_NAME "prob"

#define LOG_MAIN "[LOG_MAIN] "
int main(int argc, char** argv)
{
	std::cout<<LOG_MAIN<<"----------Start-----------"<<std::endl;

	PluginFactory pluginFactory;
	IHostMemory *gieModelStream{nullptr};
	tensorNet * net = new tensorNet(INPUT_W, INPUT_H, OUTPUT_SIZE, INPUT_BLOB_NAME, OUTPUT_BLOB_NAME);
	net->importTrainedCaffeModel("mnist.prototxt", "mnist.caffemodel", std::vector<std::string>{OUTPUT_BLOB_NAME}, 1, &pluginFactory, gieModelStream);
	pluginFactory.destroyPlugin();
	std::cout<<LOG_MAIN<<"modello caricato correttamente e distruzione dell'interfaccia PLugin"<<std::endl;

	std::cout<<LOG_MAIN<<"leggo file"<<std::endl;
	srand(unsigned(time(nullptr)));
	uint8_t fileData[INPUT_H*INPUT_W];
	readPGMFile(std::to_string(rand() % 10) + ".pgm", fileData, INPUT_W, INPUT_H);
	std::cout<<LOG_MAIN<<"file caricato correttamente"<<std::endl;

	std::cout<<LOG_MAIN<<"print rappresentazione asci"<<std::endl;
	for (int i = 0; i < INPUT_H*INPUT_W; i++)
		std::cout << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");
	
	std::cout<<LOG_MAIN<<"carico il parser per l'immagine modello"<<std::endl;
	nvcaffeparser1::ICaffeParser *parser = nvcaffeparser1::createCaffeParser();
	nvcaffeparser1::IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(locateFile("mnist_mean.binaryproto").c_str());
	std::cout<<LOG_MAIN<<"distroggo il parser"<<std::endl;
	parser->destroy();

	//reinterpret_cast : consente di convertire un tipo puntatore a qualsiasi altro tipo puntatore.
	//allo stesso modo convertire un intero in qualsiasi tipo di puntatore.
	const float *meanData = reinterpret_cast<const float*>(meanBlob->getData());
	float data[INPUT_H*INPUT_W];
	for(int i=0; i<INPUT_H*INPUT_W; i++){
		data[i] = float(fileData[i]) - meanData[i];
	}
	std::cout<<LOG_MAIN<<"distrutto il punatore mean dopo averlo usato"<<std::endl;
	meanBlob->destroy();

	std::cout<<LOG_MAIN<<"faccio l'inferza"<<std::endl;
	std::cout<<LOG_MAIN<<"creo il meccanismo di inferenza runtime"<<std::endl;
	IRuntime* runtime = createInferRuntime(net->gLogger);

	std::cout<<LOG_MAIN<<"Creo l'engine a partire dal runtime serializzato"<<std::endl;
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);

	std::cout<<LOG_MAIN<<"Creo il contesto per l'esecuzione dei kernel"<<std::endl;
	IExecutionContext *context = engine->createExecutionContext();

	float prob[OUTPUT_SIZE];
	net->doInference(*context, data, prob, 1);	
	
	std::cout<<LOG_MAIN<<"distruggo tutti gli oggetti"<<std::endl;
	context->destroy();
	engine->destroy();
	runtime->destroy();
	pluginFactory.destroyPlugin();
	
	// print a histogram of the output distribution
	std::cout << "\n\n";
	for (unsigned int i = 0; i < 10; i++){
		std::cout << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
		std::cout << std::endl;
	}
		
	return 0;	
};
