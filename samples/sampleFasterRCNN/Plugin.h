/*
 * Reshape Plugin
 */
#ifndef __PLUGIN_H__
#define __PLUGIN_H__

#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "utils.h"
#include <assert.h>

using namespace::nvinfer1;

#define LOG_PLUG "[PLG] "
template<int OutC>
class Reshape: public IPlugin
{
public:
	Reshape()
	{
		printf("%sCostruttore del Plugin Reshape\
				fa una resize dell'input senza cambiare\
				i dati all'interno :)\n", LOG_PLUG);
	}
	
	Reshape(const void* buffer, size_t size)
	{
		printf("%sCostruttore del Plugin Reshape\
				fa una resize dell'input senza cambiare\
				i dati all'interno :)\n", LOG_PLUG);
		assert(size == sizeof(mCopySize));
		mCopySize = *reinterpret_cast<const size_t*>(buffer);
	}

	~Reshape(){printf("%sAsta la Vista Baby\n", LOG_PLUG);}
	
	
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	//COnfigurazione del Layer:                                                                           // 
	//ritorno il numero di layer di ouput, in questo caso 1											//				
	//calcolo la dimensione di output in base alla dimensione degli input d
	//infine configuro l'ggetto ??
	////////////////////////////////////////////////////////////////////////////////////////////////////////
	int getNbOutputs() const override
	{
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		printf("%s definisco la dimensione del tensore per il layer in questione\n", LOG_PLUG);
		printf("%s questo tensore a 3 dimensione : 1 canale 1 widht 1 height\n", LOG_PLUG);
		assert(index == 0 && nbInputDims == 1 && inputs[index].nbDims == 3);
		assert((inputs[0].d[0]) * (inputs[0].d[1]) %OutC == 0);
//		Dims* a = new DimsCHW(OutC, inputs[0].d[0] * inputs[0].d[1] /OutC, inputs[0].d[2]);
		//printf("%s canale: %d, widht: %d, height: %d", LOG_PLUG, a->c(), a->w(), a->h());
		printf("%s capire perche???\n", LOG_PLUG);
	return(DimsCHW(OutC, inputs[0].d[0] * inputs[0].d[1] /OutC, inputs[0].d[2]));
	}

	void configure(const Dims* inputs, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override
	{
		printf("%sconfigurazione della nuova dimensione\n", LOG_PLUG);
		mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Resource Management:                                                                                           // 
	//i metodi di inizializzazione e termine, sono chimata dal builder per la creazione, e a run time dall'oggetto   //
	//IExecutionContext. Queste due funzioni servono per allocare e rialasciare risorse necessarie alla computazione //
	//dei singoli layer.						                                                                     //
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	int initialize() override
	{
		printf("%sinizializzazione del layer rashape senza conf costum\n", LOG_PLUG);
		return 0;
	}

	virtual void terminate() override
	{
		printf("%stermine del layer rashape\n", LOG_PLUG);
	}
	

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		printf("%sspazione di lavoro run time del tensore\n", LOG_PLUG);
		return 0;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Runtime execution:
	//Esecuzione dei singoli layer a runtime ovvero coda di kernel funcion
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
	{
		printf("%scopio i dati da una parte del device all'altra input out poichè il plugin non può eseguire codice in loco:(\n", LOG_PLUG);
		CHECK_CUDA(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice));
		return 0;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Serializzazione:
	//serializzazione dei parametri del layer accounto a quelli dell'intero network
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual size_t getSerializationSize() override
	{
		printf("%sritorno di byte = %d per allocara la memoria per la serializzazione di McopySize\n", LOG_PLUG, sizeof(mCopySize));
		return sizeof(mCopySize);
	}

	virtual void serialize(void* buffer) override
	{
		printf("%sserializzo il layer in questione\n", LOG_PLUG);
		*reinterpret_cast<size_t*>(buffer) = mCopySize;
	}

private:
	size_t mCopySize;

};
#endif
