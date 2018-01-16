/**
 *	Classe per aggiungere il layer "Fully Conneted" al modello, fatto tramite l'interfaccia
 *
 **/

#ifndef __ADD_PLUGIN_H__
#define __ADD_PLUGIN_H__

#include <iostream>
#include <memory> //classe per unique_ptr o smart_pointer
#include "Plugin.h"

#define LOG_PLUG "[LOG_PLUG] "

// integration for serialization
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
	// caffe parser plugin implementation
	bool isPlugin(const char* name) override
	{
		std::cout<<LOG_PLUG<<"il nome del layer è: "<<name<<std::endl;
		return !strcmp(name, "ip2");
	}

	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
	{
		static const int NB_OUTPUT_CHANNELS = 10;	
		
		std::cout<<LOG_PLUG<<"creo il plugin con i seguenti parametri: "<<std::endl;
		std::cout<<LOG_PLUG<<"numero di ouput: "<<NB_OUTPUT_CHANNELS<<std::endl;
		std::cout<<LOG_PLUG<<"tipo di peso: "<<"KFLOAT"<<std::endl;
		std::cout<<LOG_PLUG<<"numero pesi: "<<nbWeights<<std::endl;

		assert(isPlugin(layerName) && nbWeights == 2 && weights[0].type == DataType::kFLOAT && weights[1].type == DataType::kFLOAT);
		assert(mPlugin.get() == nullptr);


		std::cout<<LOG_PLUG<<"Creo L'instanza del PLuging come unique pointer"<<std::endl;
		mPlugin = std::unique_ptr<FCPlugin>(new FCPlugin(weights, nbWeights, NB_OUTPUT_CHANNELS));
		return mPlugin.get();
	}

	// deserialization plugin implementation
	IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{		
		std::cout<<LOG_PLUG<<"Creazione del Plugin tramite Serial Data"<<std::endl;
		assert(isPlugin(layerName));
		assert(mPlugin.get() == nullptr);
		mPlugin = std::unique_ptr<FCPlugin>(new FCPlugin(serialData, serialLength));
		return mPlugin.get();
	}

	// the application has to destroy the plugin when it knows it's safe to do so
	void destroyPlugin()
	{	
		std::cout<<LOG_PLUG<<"distruggo il Plugin"<<std::endl;
		mPlugin.release();
	}

	//Creo l'oggeto FCPlugin come puntatore unico, il puntatore unico e derivato dagli Smart Pointer
	//un oggetto per un e unica allocazione di risorse.
	std::unique_ptr<FCPlugin> mPlugin{ nullptr };
};

#endif
