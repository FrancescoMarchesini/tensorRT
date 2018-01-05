#ifndef __PLUGIN_H__
#define __PLUGIN_H__

/**
 * Questo programma integra in tensorRT un Reccurent Neural Network
 * non supportatato nativamente da tensoRT, tramite la definizione dello
 * stesso tramite l'interfaccia Plugin
 */
#include <cassert>
#include <iostream>
#include "cuda_runtime_api.h"

#include "Nvinfer1.h"
#include "NvUtils.h"

using namespace nvinfer1;

// Reshape plugin to feed RNN into FC layer correctly.
// Aggiungo il Plugin per la creazione del layer desiderato
class Reshape : public IPlugin
{
public:
	Reshape(size_t size) : mSize(size) {}

	//costruttore	
	Reshape(const void*buf, size_t size)
    {
        assert(size == sizeof(mSize));
		std::cout << "grandezza del tensore: "<< size << std::endl;
        mSize = *static_cast<const size_t*>(buf);
    }

	//////////////////////////////////////////////////////////////////////////////////
	//creazione del network
	/////////////////////////////////////////////////////////////////////////////////
	
	//Numero di tensori di output
	int getNbOutputs() const override{	return 1;	}
	
	// The RNN outputs in {L, N, C}, but FC layer needs {C, 1, 1}, so we can convert RNN
    // output to {L*N, C, 1, 1} and TensorRT will handle the rest.
	//Funzione per ritornare la grandezza del tensore
	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		std::cout << "determino la grandezza delle parti che compongono il tensore" << std::endl;
        if(nbInputDims == 1){std::cout<<"dimensione input 1 OK"<<std::endl;}
        if(index == 0){std::cout<<"indice è a 0  OK"<<std::endl;}
        if(inputs[index].nbDims == 3){std::cout<<"3 canali 1 diemsione e sapziali OK"};
		return DimsNCHW(inputs[index].d[1] * inputs[index].d[0], inputs[index].d[2], 1, 1);
	}
	
	//////////////////////////////////////////////////////////////////////////////////
	//Utilizzati dal Builder
	/////////////////////////////////////////////////////////////////////////////////
	
	//la configurazione dell engine con la possibilità di scegliere l'algoritmo in base alla dimensione del tensore di input.
	// Il metodo è chiamato solo una volta nel momento della costruzione dell engine. Per questo per usare le configurazione 
	// a runtime bisogna salvarla in una varibile da usare succesivamente
	void configure(const Dims*, int, const Dims*, int, int)	override{ }
	
	//spazio di lavoro utilizzato da tensoRT nel momento della costruzione dell'engine
	//una speci di memoria condivisa tra i tensori	
	size_t getWorkspaceSize(int) const override{	return 0;	}
	
	//////////////////////////////////////////////////////////////////////////////////
	//Utilizzati a runtime
	/////////////////////////////////////////////////////////////////////////////////
	
	//funzioni per allocare e disallocare le risorse tali funzioni sono chimate
	//da IExecutionContext a runtime
	int initialize() override{	return 0;	}

	//esecuzione a runtime dei singoli layer	
	int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        CHECK(cudaMemcpyAsync(static_cast<float*>(outputs[0]),
                   static_cast<const float*>(inputs[0]),
                   sizeof(float) * mSize * batchSize, cudaMemcpyDefault, stream));
        return 0;
    }
	void terminate() override{}

	//////////////////////////////////////////////////////////////////////////////////
	//Serializzazione dell'engine
	/////////////////////////////////////////////////////////////////////////////////
	
	//per seriailizzare su file l'engine devo sapere in primis la dimensione
	size_t getSerializationSize() override
    {
        return sizeof(mSize);
    }

	//serializzo su file lengine
	void serialize(void* buffer) override
    {
        (*static_cast<size_t*>(buffer)) = mSize;

    }

    
    private:
    size_t mSize{0};
};


//////////////////////////////////////////////////////////////////////////////////
//Aggiunta del Layer al Network
/////////////////////////////////////////////////////////////////////////////////
class PluginFactory : public nvinfer1::IPluginFactory
{
public:
	// deserialization plugin implementation
	IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{
		std::cout<<"aggiungo il layer tramite la deserializazione dell'engine"<<std::endl;
        assert(!strncmp(layerName, "reshape", 7));
        if (!mPlugin) mPlugin = new Reshape(serialData, serialLength);
        return mPlugin;
    }
    void destroyPlugin()
    {
        if (mPlugin) delete mPlugin;
        mPlugin = nullptr;
    }
private:
    Reshape *mPlugin{nullptr};
}; // PluginFactory
	
#endif
