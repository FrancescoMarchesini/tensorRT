#ifndef __INFERENCE_ENGINE_H__
#define __INFERENCE_ENGINE_H__

#include "NvInfer.h"
#include <iostream>
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
	 **/
    InferenceEngine(const std::string& model_file,
                    const std::string& trained_file,
					const std::vector<std::string>& output,
					unsigned int bathSize);

    ~InferenceEngine();

    void Import(const std::string& plan_file);
    void Export(const std::string& plan_file) const;

    nvinfer1::ICudaEngine* Get() const
    {
        return engine_;
    }

private:
    nvinfer1::ICudaEngine* engine_;
};

#endif
