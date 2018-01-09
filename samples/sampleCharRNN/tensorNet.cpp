#include "tensorNet.h"

Weights tensorNet::convertRNNWeights(Weights input)
{
    float* ptr = static_cast<float*>(malloc(sizeof(float)*input.count));
    int indir[4]{ 1, 2, 0, 3 };
    int order[5]{ 0, 1, 4, 2, 3};
    int dims[5]{LAYER_COUNT, 2, 4, HIDDEN_SIZE, HIDDEN_SIZE};
    utils::reshapeWeights(input, dims, order, ptr, 5);
    utils::transposeSubBuffers(ptr, DataType::kFLOAT, LAYER_COUNT * 2, HIDDEN_SIZE * HIDDEN_SIZE, 4);
    int subMatrix = HIDDEN_SIZE * HIDDEN_SIZE;
    int layerOffset = 8 * subMatrix;
    for (int z = 0; z < LAYER_COUNT; ++z)
    {
        utils::reorderSubBuffers(ptr + z * layerOffset, indir, 4, subMatrix * sizeof(float));
        utils::reorderSubBuffers(ptr + z * layerOffset + 4 * subMatrix, indir, 4, subMatrix * sizeof(float));
    }
    return Weights{input.type, ptr, input.count};
}


Weights tensorNet::convertRNNBias(Weights input)
{
    float* ptr = static_cast<float*>(malloc(sizeof(float)*input.count*2));
    std::fill(ptr, ptr + input.count*2, 0);
    const float* iptr = static_cast<const float*>(input.values);
    int indir[4]{ 1, 2, 0, 3 };
    for (int z = 0, y = 0; z < LAYER_COUNT; ++z)
        for (int x = 0; x < 4; ++x, ++y)
            std::copy(iptr + y * HIDDEN_SIZE , iptr + (y + 1) * HIDDEN_SIZE, ptr + (z * 8 + indir[x]) * HIDDEN_SIZE);
    return Weights{input.type, ptr, input.count*2};
}


Weights tensorNet::transposeFCWeights(Weights input)
{
    float* ptr = static_cast<float*>(malloc(sizeof(float)*input.count));
    const float* iptr = static_cast<const float*>(input.values);
    assert(input.count == HIDDEN_SIZE * OUTPUT_SIZE);
    for (int z = 0; z < HIDDEN_SIZE; ++z)
        for (int x = 0; x < OUTPUT_SIZE; ++x)
            ptr[x * HIDDEN_SIZE + z] = iptr[z * OUTPUT_SIZE + x];
    return Weights{input.type, ptr, input.count};
}

void tensorNet::APIToModel(std::map<std::string, Weights> &weightMap, IHostMemory **modelStream)
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // create the model to populate the network, then set the outputs and create an engine
    INetworkDefinition* network = builder->createNetwork();

    auto data = network->addInput(INPUT_BLOB_NAME, DataType::kFLOAT, DimsCHW{ SEQ_SIZE, BATCH_SIZE, DATA_SIZE});
    assert(data != nullptr);

    auto hiddenIn = network->addInput(HIDDEN_IN_BLOB_NAME, DataType::kFLOAT, DimsCHW{ LAYER_COUNT, BATCH_SIZE, HIDDEN_SIZE});
    assert(hiddenIn != nullptr);

    auto cellIn = network->addInput(CELL_IN_BLOB_NAME, DataType::kFLOAT, DimsCHW{ LAYER_COUNT, BATCH_SIZE, HIDDEN_SIZE});
    assert(cellIn != nullptr);

    // Create an RNN layer w/ 2 layers and 512 hidden states
    auto tfwts = weightMap["rnnweight"];
    Weights rnnwts = convertRNNWeights(tfwts);
    auto tfbias = weightMap["rnnbias"];
    Weights rnnbias = convertRNNBias(tfbias);

    auto rnn = network->addRNN(*data, LAYER_COUNT, HIDDEN_SIZE, SEQ_SIZE,
            RNNOperation::kLSTM, RNNInputMode::kLINEAR, RNNDirection::kUNIDIRECTION,
            rnnwts, rnnbias);
    assert(rnn != nullptr);
    rnn->getOutput(0)->setName("RNN output");
    rnn->setHiddenState(*hiddenIn);
    if (rnn->getOperation() == RNNOperation::kLSTM)
        rnn->setCellState(*cellIn);
    
    Reshape reshape(SEQ_SIZE * BATCH_SIZE * HIDDEN_SIZE);
    ITensor *ptr = rnn->getOutput(0);
    auto plugin = network->addPlugin(&ptr, 1, reshape);
    plugin->setName("reshape");

    // Add a second fully connected layer with 50 outputs.
    auto tffcwts = weightMap["rnnfcw"];
    auto wts = transposeFCWeights(tffcwts);
    auto bias = weightMap["rnnfcb"];
    auto fc = network->addFullyConnected(*plugin->getOutput(0), OUTPUT_SIZE, wts, bias);
    assert(fc != nullptr);
    fc->getOutput(0)->setName("FC output");

    // Add a softmax layer to determine the probability.
    auto prob = network->addSoftMax(*fc->getOutput(0));
    assert(prob != nullptr);
    prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*prob->getOutput(0));
    rnn->getOutput(1)->setName(HIDDEN_OUT_BLOB_NAME);
    network->markOutput(*rnn->getOutput(1));
    if (rnn->getOperation() == RNNOperation::kLSTM)
    {
        rnn->getOutput(2)->setName(CELL_OUT_BLOB_NAME);
        network->markOutput(*rnn->getOutput(2));
    }

    // Build the engine
    builder->setMaxBatchSize(1);
    builder->setMaxWorkspaceSize(1 << 25);

    // Store the transformed weights in the weight map so the memory can be properly released later.
    weightMap["rnnweight2"] = rnnwts;
    weightMap["rnnbias2"] = rnnbias;
    weightMap["rnnfcw2"] = wts;

    auto engine = builder->buildCudaEngine(*network);
    assert(engine != nullptr);
    // we don't need the network any more
    network->destroy();

    // serialize the engine, then close everything down
    (*modelStream) = engine->serialize();
    engine->destroy();
    builder->destroy();
}

void tensorNet::stepOnce(float **data, void **buffers, int *sizes, int *indices,
        int numBindings, cudaStream_t &stream, IExecutionContext &context)
{
    for (int z = 0, w = numBindings/2; z < w; ++z)
        CHECK(cudaMemcpyAsync(buffers[indices[z]], data[z], sizes[z] * sizeof(float), cudaMemcpyHostToDevice, stream));

    // Execute asynchronously
    context.enqueue(1, buffers, stream, nullptr);

    // DMA the input from the GPU
    for (int z = numBindings/2, w = numBindings; z < w; ++z)
        CHECK(cudaMemcpyAsync(data[z], buffers[indices[z]], sizes[z] * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // Copy Ct/Ht to the Ct-1/Ht-1 slots.
    CHECK(cudaMemcpyAsync(data[1], buffers[indices[4]], sizes[1] * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(data[2], buffers[indices[5]], sizes[2] * sizeof(float), cudaMemcpyDeviceToHost, stream));
}

bool tensorNet::doInference(IExecutionContext& context, std::string input, std::string expected, std::map<std::string, Weights> &weightMap)
{
    const ICudaEngine& engine = context.getEngine();
    // We have 6 outputs for LSTM, this needs to be changed to 4 for any other RNN type
    static const int numBindings = 6;
    assert(engine.getNbBindings() == numBindings);
    void* buffers[numBindings];
    float* data[numBindings];
    std::fill(buffers, buffers + numBindings, nullptr);
    std::fill(data, data + numBindings, nullptr);
    const char *names[numBindings] = {INPUT_BLOB_NAME,
        HIDDEN_IN_BLOB_NAME,
        CELL_IN_BLOB_NAME,
        OUTPUT_BLOB_NAME,
        HIDDEN_OUT_BLOB_NAME,
        CELL_OUT_BLOB_NAME
    };
    int indices[numBindings];
    std::fill(indices, indices + numBindings, -1);
    int sizes[numBindings] = { SEQ_SIZE * BATCH_SIZE * DATA_SIZE,
        LAYER_COUNT * BATCH_SIZE * HIDDEN_SIZE,
        LAYER_COUNT * BATCH_SIZE * HIDDEN_SIZE,
        OUTPUT_SIZE,
        LAYER_COUNT * BATCH_SIZE * HIDDEN_SIZE,
        LAYER_COUNT * BATCH_SIZE * HIDDEN_SIZE
    };

    for (int x = 0; x < numBindings; ++x)
    {
        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // note that indices are guaranteed to be less than IEngine::getNbBindings()
        indices[x] = engine.getBindingIndex(names[x]);
        if (indices[x] == -1) continue;
        // create GPU buffers and a stream
        assert(indices[x] < numBindings);
        CHECK(cudaMalloc(&buffers[indices[x]], sizes[x] * sizeof(float)));
        data[x] = new float[sizes[x]];
    }
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    // Initialize input/hidden/cell state to zero
    for (int x = 0; x < numBindings; ++x) std::fill(data[x], data[x] + sizes[x], 0.0f);

    auto embed = weightMap["embed"];
    std::string genstr;
    assert(BATCH_SIZE == 1 && "This code assumes batch size is equal to 1.");
    // Seed the RNN with the input.
    for (auto &a : input)
    {
        std::copy(reinterpret_cast<const float*>(embed.values) + char_to_id[a]*DATA_SIZE,
                reinterpret_cast<const float*>(embed.values) + char_to_id[a]*DATA_SIZE + DATA_SIZE,
                data[0]);
        stepOnce(data, buffers, sizes, indices, 6, stream, context);
        cudaStreamSynchronize(stream);
        genstr.push_back(a);
    }
    // Now that we have gone through the initial sequence, lets make sure that we get the sequence out that
    // we are expecting.
    for (size_t x = 0, y = expected.size(); x < y; ++x)
    {
        std::copy(reinterpret_cast<const float*>(embed.values) + char_to_id[*genstr.rbegin()]*DATA_SIZE,
                reinterpret_cast<const float*>(embed.values) + char_to_id[*genstr.rbegin()]*DATA_SIZE + DATA_SIZE,
                data[0]);

        stepOnce(data, buffers, sizes, indices, 6, stream, context);
        cudaStreamSynchronize(stream);

		float* probabilities = reinterpret_cast<float*>(data[indices[3]]);
		ptrdiff_t idx = std::max_element(probabilities, probabilities + sizes[3]) - probabilities;
        genstr.push_back(id_to_char[idx]);
    }
    printf("Received: %s\n", genstr.c_str() + input.size());

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    for (int x = 0; x < numBindings; ++x)
    {
        CHECK(cudaFree(buffers[indices[x]]));
        if (data[x]) delete [] data[x];
    }
    return genstr == (input + expected);
}


