/**
 *		!!!il seguente pluging server per implemetare la rete FsterR-CNN-RPN,ovvero una rete di convoluzione
 *		che non è applica su tutta l'immagine ma su singole parti di interesse(Region Proposa Network )
 *		il seguente plugin quindi fonde, come nel papaer di riferimento, due network RPN ROIpolling in un unica
 *		istanza:)	
 *
 **/

#ifndef __ADD_PLUGIN_H__
#define __ADD_PLUGIN_H__

#include <iostream>
#include <memory> //classe per unique_ptr o smart_pointer
#include "Plugin.h"
#include "NvInferPlugin.h"

class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:

	//parsing del modello dal file .protxt e creazione dei layer
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
	{
		printf("%sCreazione del layer dei Plugin", LOG_PLUG);
		assert(isPlugin(layerName));
		if(!strcmp(layerName, "ReshapeCTo2"))
		{
			assert(mPluginRshp2 == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			printf("%sInstanzione l'unique ptr per l'oggetto ReshapeCTo2", LOG_PLUG);
			mPluginRshp2 = std::unique_ptr<Reshape<2>>(new Reshape<2>());
			return mPluginRshp2.get();
		}
		else if(!strcmp(layerName, "ReshapeCTo18"))
		{
			assert(mPluginRshp2 == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			printf("%sInstanzione l'unique ptr per l'oggetto ReshapeCTo18", LOG_PLUG);
			mPluginRshp18 = std::unique_ptr<Reshape<18>>(new Reshape<18>());
			return mPluginRshp18.get();
		}
		else if(!strcmp(layerName, "RPROIFused"))
		{
			assert(mPluginRPROI == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			printf("%sInstanzione l'unique ptr per l'oggetto RPROIFused", LOG_PLUG);
			printf("%sPasso i parametri al layer", LOG_PLUG);
			mPluginRPROI = std::unique_ptr<nvinfer1::INvPlugin, decltype(nvinfer1::nvPluginDeleter)>
				(nvinfer1::createFasterRCNNPlugin(featureStride, preNmsTop, nmsMaxOut, iouThreshold, minBoxSize, spatialScale,
					DimsHW(poolingH, poolingW), 
					Weights{ nvinfer1::DataType::kFLOAT, anchorsRatios, anchorsRatioCount },
					Weights{ nvinfer1::DataType::kFLOAT, anchorsScales, anchorsScaleCount }), nvPluginDeleter);
			return mPluginRPROI.get();
		}
		else
		{
			assert(0);
			return nullptr;
		}
	}

	//creazione dei layer dal file plancreazione dei layer dal file planee 
	nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{		
		printf("%sCreazione del Plugin tramite Serial Data", LOG_PLUG);
		assert(isPlugin(layerName));
		if(!strcmp(layerName, "ReshapeCTo2"))
		{
			assert(mPluginRshp2 == nullptr);
			printf("%sInstanzione l'unique ptr per l'oggetto resiale ReshapeCTo2", LOG_PLUG);
			mPluginRshp2 = std::unique_ptr<Reshape<2>>(new Reshape<2>(serialData, serialLength));
			return mPluginRshp2.get();
		}
		else if(!strcmp(layerName, "ReshapeCTo18"))
		{
			assert(mPluginRshp18 == nullptr);
			printf("%sInstanzione l'unique ptr per l'oggetto resiale ReshapeCTo18", LOG_PLUG);
			mPluginRshp18 = std::unique_ptr<Reshape<18>>(new Reshape<18>(serialData, serialLength));
			return mPluginRshp8.get();
		}
		else(!strcmp(layerName, "RPROIFused")
		{
			assert(mPluginRPROI =nullptr);
			printf("%sInstanzione l'unique ptr per l'oggetto resiale RPROIFused", LOG_PLUG);
			mPluginRPROI = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
				(nvinfer1::createFasterRCNNPlugin(serialData, serialLength), PluginDeleter);
			return mPluginRPROI.get();
		}
		else
		{
			assert(0);
			return nullptr;
		}
	}

	void destroyPlugin()
	{	
		printf("%sdistruggo tuttoooo ",LOG_PLUG);
		mPluginRshp2.release(); mPluginRshp2 = nullptr;
		mPluginRshp18.release(); mPluginRshp18 = nullptr;
		mPluginRPROI.release(); mPluginRPROI = nullptr;
	}
	
	//funzione bool per vedere se il layer è uno di quelli che devono essere manualmente implementati
	bool isPlugin(const char* name) override
	{
		printf("%sil nome del layer è:%s ", LOG_PLUG, name);
		return !strcmp(name, "ReshapeCTo2")
			|| !strcmp(name, "ReshapeCTo18")
			|| !strcmp(name, "RPROIFused")
	}


private:
	//parametri della creazione del layer presi direttamente dal paper sopra citato
	int poolingH = 7;
	int poolingW = 7;
	int featureStride = 16;
	int preNmsTop = 6000;
	int nmsMaxOut = 300;
	int anchorsRatioCount = 3;
	int anchorsScaleCount = 3;
	float iouThreshold = 0.7f;
	float minBoxSize = 16;
	float spatialScale = 0.0625f;
	float anchorsRatios[3] = { 0.5f, 1.0f, 2.0f };
	float anchorsScales[3] = { 8.0f, 16.0f, 32.0f };
	
	//Creo l'oggeto Reshape come puntatore unico, il puntatore unico e derivato dagli Smart Pointer
	//un oggetto per un e unica allocazione di risorse.
	std::unique_ptr<Reshape<2>> mPluginRshp2{ nullptr };
	std::unique_ptr<Reshape<18>> mPluginRshp18{ nullptr };
	void(*nvinfer1::nvPluginDeleter)(nvinfer1::INvPlugin*){[](nvinfer1::INvPlugin* ptr){ptr->destroy();}};
	std::unique_ptr<nvinfer1::INvPlugin, decltype(nvPluginDeleter)> mPluginRPROI{nullptr, nvinfer1::nvPluginDeleter};
};

#endif
