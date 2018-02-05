#ifndef __INFERENCE_ENGINE_H__
#define __INFERENCE_ENGINE_H__

#include "NvInfer.h"
#include "NvCaffeParser.h"
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
					unsigned int bathSize,
					nvcaffeparser1::IPluginFactory* pluginFactory,
					nvinfer1::IHostMemory **gieModelStream);

    ~InferenceEngine();

    bool planeToModel(const std::string& plan_file);
    bool modelToPlane(const std::string& plan_file);

    nvinfer1::ICudaEngine* Get() const
    {
        return engine_;
    }

private:
    nvinfer1::ICudaEngine* engine_;
};

#endif
