#ifndef __INFERENCE_ENGINE_H__
#define __INFERENCE_ENGINE_H__

#include "NvInfer.h"

class InferenceEngine
{
public:
    InferenceEngine();

    InferenceEngine(const std::string& model_file,
                    const std::string& trained_file);

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
