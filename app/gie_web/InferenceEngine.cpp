#include <fstream>
#include <glog/logging.h>

#include "NvCaffeParser.h"
#include "InferenceEngine.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using std::string;

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
};
static Logger gLogger;


InferenceEngine::InferenceEngine()
{

}

InferenceEngine::InferenceEngine(
  const string& model_file,
  const string& trained_file
)
{
    IBuilder* builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();

    ICaffeParser* parser = createCaffeParser();
    auto blob_name_to_tensor = parser->parse(model_file.c_str(),
                                            trained_file.c_str(),
                                            *network,
                                            DataType::kFLOAT);
    CHECK(blob_name_to_tensor) << "Could not parse the model";

    // specify which tensors are outputs
    network->markOutput(*blob_name_to_tensor->find("softmax"));

    // Build the engine
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(1 << 30);

    engine_ = builder->buildCudaEngine(*network);
    CHECK(engine_) << "Failed to create inference engine.";

    network->destroy();
    builder->destroy();
}

InferenceEngine::~InferenceEngine()
{
    engine_->destroy();
}

void InferenceEngine::Import(const string& plan_file) 
{
    std::ifstream infile(plan_file.c_str(), std::ifstream::binary);
    IRuntime* runtime = createInferRuntime(gLogger);
    engine_ = runtime->deserializeCudaEngine(infile);
    infile.close();
    runtime->destroy();
}

void InferenceEngine::Export(const string& plan_file) const 
{
    std::ofstream outfile(plan_file.c_str(), std::ofstream::binary);
    engine_->serialize(outfile);
    outfile.close();
}
