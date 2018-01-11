#include "TensorNet.h"
#include "addPlugin.h"
#include "Plugin.h"
#include "NvInfer.h"
#include "NvCaffeParser.h"
static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int OUTPUT_SIZE = 10;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

int main(int argc, char** argv)
{
	PluginFactory pluginFactory;
	IHostMemory *gieModelStream{nullptr};
	tensorNet* net = new tensorNet::importTrainedCaffeModel("mnist.prototxt", "mnist.caffemodel", std::vector<string>{OUTPUT_BLOB_NAME}, 1, &pluginFactory, gieModelStream);

	pluginFactory.destroyPlugin();
	return 0;	
}
