/**
 *		!!!il seguente pluging server per implemetare la rete FsterR-CNN-RPN,ovvero una rete di convoluzione
 *		che non è applica su tutta l'immagine ma su singole parti di interesse(Region Proposa Network )
 *		istanza:)	
 *
 **/

#ifndef __ADD_PLUGIN_H__
#define __ADD_PLUGIN_H__

#include <iostream>
#include <memory> //classe per unique_ptr o smart_pointer
#include "Plugin.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "NvInfer.h"


using namespace nvcaffeparser1;
using namespace nvinfer1;
using namespace plugin;

class info
{
    public:
    info()
    {
        poolingH = 7;
        poolingW = 7;
        featureStride = 16;
        preNmsTop = 6000;
        nmsMaxOut = 300;
        anchorsRatioCount = 3;
        anchorsScaleCount = 3;
        iouThreshold = 0.7f;
        minBoxSize = 16;
        spatialScale = 0.0625f;
    }

    int poolingH;
    int poolingW;
    int featureStride;
    int preNmsTop;
    int nmsMaxOut;
    int anchorsRatioCount;
    int anchorsScaleCount;
    float iouThreshold;
    float minBoxSize;
    float spatialScale;
    float anchorsRatios[3]={ 0.5f, 1.0f, 2.0f };
    float anchorsScales[3]={ 8.0f, 16.0f, 32.0f };
};

#define LOG_ADD "[ADD_PLG] "
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:

	//parsing del modello dal file .protxt e creazione dei layer
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
	{
		printf("%sCreazione del layer coustum parsando il il file\n", LOG_ADD);
		assert(isPlugin(layerName));
		if(!strcmp(layerName, "ReshapeCTo2"))
		{
			assert(mPluginRshp2 == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			printf("%sInstanzio l'unique ptr l'oggetto ReshapeCTo2\n", LOG_ADD);
			mPluginRshp2 = std::unique_ptr<Reshape<2>>(new Reshape<2>());
			return mPluginRshp2.get();
		}
		else if(!strcmp(layerName, "ReshapeCTo18"))
		{
			assert(mPluginRshp18 == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			printf("%sInstanzio l'unique ptr per l'oggetto ReshapeCTo18\n", LOG_ADD);
			mPluginRshp18 = std::unique_ptr<Reshape<18>>(new Reshape<18>());
			return mPluginRshp18.get();
		}
		else if(!strcmp(layerName, "RPROIFused"))
		{
			assert(mPluginRPROI == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			printf("%sInstanzio l'unique ptr per l'oggetto RPROIFused\n", LOG_ADD);
			printf("%sCreo il plugin FasterRCNNPlugin \n", LOG_ADD);
			mPluginRPROI = std::unique_ptr<nvinfer1::plugin::INvPlugin, decltype(nvPluginDeleter)>
				(createFasterRCNNPlugin(in.featureStride, in.preNmsTop, in.nmsMaxOut, in.iouThreshold, in.minBoxSize,in.spatialScale,
					DimsHW(in.poolingH, in.poolingW), 
					Weights{ nvinfer1::DataType::kHALF, in.anchorsRatios, in.anchorsRatioCount },
					Weights{ nvinfer1::DataType::kHALF, in.anchorsScales, in.anchorsScaleCount }), nvPluginDeleter);
			return mPluginRPROI.get();
		}
		else
		{
			assert(0);
			return nullptr;
		}
	}

	//creazione dei layer dal file plancreazione dei layer dal file planee 
	IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{		
		printf("%sCreazione del Plugin tramite Serial Data\n", LOG_ADD);
		assert(isPlugin(layerName));
		if(!strcmp(layerName, "ReshapeCTo2"))
		{
			assert(mPluginRshp2 == nullptr);
			printf("%sInstanzione l'unique ptr per l'oggetto resiale ReshapeCTo2\n", LOG_ADD);
			mPluginRshp2 = std::unique_ptr<Reshape<2>>(new Reshape<2>(serialData, serialLength));
			return mPluginRshp2.get();
		}
		else if(!strcmp(layerName, "ReshapeCTo18"))
		{
			assert(mPluginRshp18 == nullptr);
			printf("%sInstanzione l'unique ptr per l'oggetto resiale ReshapeCTo18\n", LOG_ADD);
			mPluginRshp18 = std::unique_ptr<Reshape<18>>(new Reshape<18>(serialData, serialLength));
			return mPluginRshp18.get();
		}
		else if(!strcmp(layerName, "RPROIFused"))
		{
			assert(mPluginRPROI == nullptr);
			printf("%sInstanzione l'unique ptr per l'oggetto resiale RPROIFused\n", LOG_ADD);
			mPluginRPROI = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
				(createFasterRCNNPlugin(serialData, serialLength), nvPluginDeleter);
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
		printf("%sdistruggo tuttoooo\n",LOG_ADD);
		mPluginRshp2.release(); mPluginRshp2 = nullptr;
		mPluginRshp18.release(); mPluginRshp18 = nullptr;
		mPluginRPROI.release(); mPluginRPROI = nullptr;
	}
	
	//funzione bool per vedere se il layer è uno di quelli che devono essere manualmente implementati
	bool isPlugin(const char* name) override
	{
		printf("%slayer %s\n", LOG_ADD, name);
		return !strcmp(name, "ReshapeCTo2")
			|| !strcmp(name, "ReshapeCTo18")
			|| !strcmp(name, "RPROIFused");
	}
    
	//Creo l'oggeto Reshape come puntatore unico, il puntatore unico e derivato dagli Smart Pointer
	//un oggetto per un e unica allocazione di risorse.
	std::unique_ptr<Reshape<2>> mPluginRshp2{ nullptr };
	std::unique_ptr<Reshape<18>> mPluginRshp18{ nullptr };
	void(*nvPluginDeleter)(INvPlugin*){[](INvPlugin* ptr){ptr->destroy();}};
	std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mPluginRPROI{nullptr, nvPluginDeleter};
    info in;
};

#endif
