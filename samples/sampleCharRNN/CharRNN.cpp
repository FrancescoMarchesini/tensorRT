#include "CharRNN.h"

CharRNN::CharRNN(void):tensorNet()
{
	std::cout<<"creo il CharRNN"<<std::endl;
}

// Our weight files are in a very simple space delimited format.
// type is the integer value of the DataType enum in NvInfer.h.
// <number of buffers>
// for each buffer: [name] [type] [size] <data x size in hex> 
std::map<std::string, Weights> CharRNN::loadWeights(const std::string file)
{
    std::map<std::string, Weights> weightMap;
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while(count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t type, size;
        std::string name;
        input >> name >> std::dec >> type >> size;
        wt.type = static_cast<DataType>(type);
        if (wt.type == DataType::kFLOAT)
        {
            uint32_t *val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];

            }
            wt.values = val;
        } else if (wt.type == DataType::kHALF)
        {
            uint16_t *val = reinterpret_cast<uint16_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}


// We have the data files located in a specific directory. This 
// searches for that directory format from the current directory.
std::string CharRNN::locateFile(const std::string& input)
{
    std::string file = "data/samples/char-rnn/" + input;
    struct stat info;
    int i, MAX_DEPTH = 10;
    for (i = 0; i < MAX_DEPTH && stat(file.c_str(), &info); i++)
        file = "../" + file;

    if (i == MAX_DEPTH)
    {
		file = std::string("data/char-rnn/") + input;
		for (i = 0; i < MAX_DEPTH && stat(file.c_str(), &info); i++)
			file = "../" + file;		
    }

    assert(i != MAX_DEPTH);

    return file;
}

void CharRNN::init()
{
    srand(unsigned(time(nullptr)));
    
	std::map<std::string, Weights> weightMap = loadWeights(locateFile("char-rnn.wts"));
    
	CharRNN* RNN = new CharRNN();
	//costruisco l'engine
	RNN->APIToModel(weightMap, &gieModelStream);

	//seleziono una frase random
    int num = rand() % 10;

	mRuntime = createInferRuntime(gLogger);
	mEngine = mRuntime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginfactory);
    
	if (gieModelStream) gieModelStream->destroy();

	//qui si potrebbe ritornare il puntatore al constesto
	//e passarlo alla funzione successiva per maggiore utilizzabilità
	IExecutionContext *context = mEngine->createExecutionContext();
	if(!context){
		std::cout<<"porco cazzao" <<std::endl <<std::endl;
	} 
const char* strings[10]{ 
		"customer serv",
        "business plans",
        "help",
        "slightly under",
        "market",
        "holiday cards",
        "bring it",
        "what time",
        "the owner thinks",
        "money can be use"
};
const char* outs[10]{ 
		"es and the",
        " to be a",
        "en and",
        "iting the company",
        "ing and",
        " the company",
        " company said it will",
        "d and the company",
        "ist with the",
        "d to be a"
};

	bool pass {false};
    std::cout << "\n---------------------------" << "\n";
    std::cout << "RNN Warmup: " << strings[num] << std::endl;
    std::cout << "Expect: " << outs[num] << std::endl;
	pass =RNN->doInference(*context, strings[num], outs[num], weightMap);
    if (!pass) std::cout << "Failure!" << std::endl;
    std::cout << "---------------------------" << "\n";

	for (auto &mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
    
	// destroy the engine
    mContext->destroy();
	context->destroy();
    mEngine->destroy();
    mRuntime->destroy();
    pluginfactory.destroyPlugin();
	delete RNN;
}

