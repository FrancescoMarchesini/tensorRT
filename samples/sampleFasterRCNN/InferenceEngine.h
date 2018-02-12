#ifndef __INFERENCE_ENGINE_H__
#define __INFERENCE_ENGINE_H__

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "addPlugin.h"
#include <iostream>
#include <fstream>
#include <sstream>

class InferenceEngine
{
public:
	InferenceEngine();

	/**
	 *	model_file : file.prototext
	 *	trained_file: file.caffemodel
	 *  outputs: vettori degli output classi
	 *  bathsize: la batch size del modello
	 *  plugin: il plugin per il network custom
	 *  IhostMemory: oggetto per serializzare il network
	 **/
    InferenceEngine(const std::string& model_file,
                    const std::string& trained_file,
					const std::vector<std::string>& output,
					unsigned int bathSize);

    ~InferenceEngine();

   
    bool doInference(const std::string& plan_file);
    bool modelToPlane(const std::string& plan_file);

    nvinfer1::ICudaEngine* Get() const
    {
        return engine_;
    }

private:
    //cuda engine 
    nvinfer1::ICudaEngine* engine_;
    //interfaccia per il plugin
    PluginFactory pluginFactory;
    
    //oggetto per Serializzare il modello sulla memoria dell'host
    nvinfer1::IHostMemory* serMem{nullptr};
};

#endif
