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
    ~InferenceEngine();

    InferenceEngine();
  
	/**
	 *	model_file : file.prototext
	 *	trained_file: file.caffemodel
	 *  outputs: vettori degli output classi
	 *  bathsize: la batch size del modello
	 *  plugin: il plugin per il network custom
	 **/
    InferenceEngine(const std::string& model_file,
                    const std::string& trained_file,
					const std::vector<std::string>& output,
					unsigned int bathSize);
    
    bool loadFastRCNNFromModel();
    bool loadFastRCNNFromPlane(const std::string& plan_file);
    bool saveModelToPlane(const std::string& plan_file);
   
    bool doInference();
    
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
    //file di deploy 
    const char* _deploy;
    //file di model
    const char* _model;
    //output del network
    std::vector<std::string> _output;
    //batchSize
    unsigned int _batchSize;
};

#endif
