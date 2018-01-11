/*
 * this samplePlugin example demonstrates how to add a custom layer to TensorRT. It
 *replaces the final fully connected layer of the MNIST sample with a direct call to CUDA ®
 *Basic Linear Algebra Subroutines library TM (cuBLAS).
 */
#ifndef __PLUGIN_H__
#define __PLUGIN_H__

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <string.h>
#include "utils.h"
using namespace::nvinfer1;
using namespace::nvcaffeparser1;

class FCPlugin: public IPlugin
{
public:
	FCPlugin(const Weights *weights, int nbWeights, int nbOutputChannels): mNbOutputChannels(nbOutputChannels)
	{
		// since we want to deal with the case where there is no bias, we can't infer
		// the number of channels from the bias weights.

		assert(nbWeights == 2);
		std::cout<<LOG_CUDA<<"Costruttore"<<std::endl;

		std::cout<<LOG_CUDA<<"Inizializzo i due Weights : mKernelW, mBiasWeights"<<std::endl;
		mKernelWeights = copyToDevice(weights[0].values, weights[0].count);
		mBiasWeights = copyToDevice(weights[1].values, weights[1].count);
		assert(mBiasWeights.count == 0 || mBiasWeights.count == nbOutputChannels);

		mNbInputChannels = int(weights[0].count / nbOutputChannels);
	}

	// create the plugin at runtime from a byte stream
	FCPlugin(const void* data, size_t length)
	{
		std::cout<<LOG_CUDA<<"Costruttore per utilizzo a Run time dal file seriale"<<std::endl;
		const char* d = reinterpret_cast<const char*>(data), *a = d;
		mNbInputChannels = read<int>(d);
		mNbOutputChannels = read<int>(d);
		int biasCount = read<int>(d);

		mKernelWeights = deserializeToDevice(d, mNbInputChannels * mNbOutputChannels);
		mBiasWeights = deserializeToDevice(d, biasCount);
		assert(d == a + length);
	}

	~FCPlugin()
	{
		cudaFree(const_cast<void*>(mKernelWeights.values));
		cudaFree(const_cast<void*>(mBiasWeights.values));
	}
	
	
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//COnfigurazione del Layer:                                                                                      // 
	//ritorno il numero di layer di ouput, in questo caso 1															//				
	//calcolo la dimensione di output in base alla dimensione degli input d, sull'indice 0, 
	//ovvero il canale
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	int getNbOutputs() const override
	{
		return 1;
	}

	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
		assert(mNbInputChannels == inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2]);
		std::cout<<"input[0]: "<<input[0]<<std::endl;
		return DimsCHW(mNbOutputChannels, 1, 1);
	}

	void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override
	{
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Resource Management:                                                                                           // 
	//i metodi di inizializzazione e termine, sono chimata dal builder per la creazione, e a run time dall'oggetto   //
	//IExecutionContext. Queste due funzioni servono per allocare e rialasciare risirse necessarie alla computazione //
	//dei singoli layer. In questo cuBLAS e cuDNN                                                                    //
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	int initialize() override
	{
		CHECK_CUDA(cudnnCreate(&mCudnn));							// initialize cudnn and cublas
		CHECK_CUDA(cublasCreate(&mCublas));
		CHECK_CUDA(cudnnCreateTensorDescriptor(&mSrcDescriptor));	// create cudnn tensor descriptors we need for bias addition
		CHECK_CUDA(cudnnCreateTensorDescriptor(&mDstDescriptor));

		return 0;
	}

	virtual void terminate() override
	{
		CHECK_CUDA(cublasDestroy(mCublas));
		CHECK_CUDA(cudnnDestroy(mCudnn));
	}
	

	virtual size_t getWorkspaceSize(int maxBatchSize) const override
	{
		return 0;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Runtime execution:
	//Esecuzione dei singoli layer a runtime
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
	{
		float kONE = 1.0f, kZERO = 0.0f;
		cublasSetStream(mCublas, stream);
		cudnnSetStream(mCudnn, stream);
		CHECK_CUDA(cublasSgemm(mCublas, CUBLAS_OP_T, CUBLAS_OP_N, mNbOutputChannels, batchSize, mNbInputChannels, &kONE, 
				reinterpret_cast<const float*>(mKernelWeights.values), mNbInputChannels, 
				reinterpret_cast<const float*>(inputs[0]), mNbInputChannels, &kZERO, 
				reinterpret_cast<float*>(outputs[0]), mNbOutputChannels));
		if (mBiasWeights.count)
		{
			CHECK_CUDA(cudnnSetTensor4dDescriptor(mSrcDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, mNbOutputChannels, 1, 1));
			CHECK_CUDA(cudnnSetTensor4dDescriptor(mDstDescriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, mNbOutputChannels, 1, 1));
			CHECK_CUDA(cudnnAddTensor(mCudnn, &kONE, mSrcDescriptor, mBiasWeights.values, &kONE, mDstDescriptor, outputs[0]));
		}
		return 0;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//Serializzazione:
	//serializzazione dei parametri del layer accounto a quelli dell'interno network
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	virtual size_t getSerializationSize() override
	{
		// 3 integers (number of input channels, number of output channels, bias size), and then the weights:
		return sizeof(int)*3 + mKernelWeights.count*sizeof(float) + mBiasWeights.count*sizeof(float);
	}

	virtual void serialize(void* buffer) override
	{
		char* d = reinterpret_cast<char*>(buffer), *a = d;

		write(d, mNbInputChannels);
		write(d, mNbOutputChannels);
		write(d, (int)mBiasWeights.count);
		serializeFromDevice(d, mKernelWeights);
		serializeFromDevice(d, mBiasWeights);
		std::cout<<LOG_CUDA<<"serializzazione d:"<<d<<std::endl;
		assert(d == a + getSerializationSize());
	}
private:
	template<typename T> void write(char*& buffer, const T& val)
	{
		std::cout<<LOG_CUDA<<"Definizione del Tipo T write"<<std::endl;
		*reinterpret_cast<T*>(buffer) = val;
		buffer += sizeof(T);
	}

	template<typename T> T read(const char*& buffer)
	{
		std::cout<<LOG_CUDA<<"Definizione del Tipo T read"<<std::endl;
		T val = *reinterpret_cast<const T*>(buffer);
		buffer += sizeof(T);
		return val;
	}

	Weights copyToDevice(const void* hostData, size_t count)
	{
		std::cout<<LOG_CUDA<<"Calcolo dei pesi in GPU"<<std::endl;
		void* deviceData;
		
		std::cout<<LOG_CUDA<<"alloco memoria in CUDA = "<< count * sizeof(float)<<std::endl;
		CHECK_CUDA(cudaMalloc(&deviceData, count * sizeof(float)));
		
		std::cout<<LOG_CUDA<<"Copio i dati dall'host al device"<<std::endl;
		CHECK_CUDA(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));

		std::cout<<LOG_CUDA<<"ritorno i pesi calcolati: tipo=KFLOAT, valore="<<*deviceData<<" numero di pesi= "<< count <<std::endl;
		return Weights{ DataType::kFLOAT, deviceData, int64_t(count) };
	}

	void serializeFromDevice(char*& hostBuffer, Weights deviceWeights)
	{	
		std::cout<<LOG_CUDA<<"Serializzo i Pesi dal Device All'Host"<<std::endl;	
		std::cout<<LOG_CUDA<<"valore= "<< deviceWeights.values<<" numero="<<deviceWeights.cout<<std::endl;	
		cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost);
		hostBuffer += deviceWeights.count * sizeof(float);
		std::cout<LOG_CUDA<<"grandezza finale della Host memory = "<<*hostBuffer<<std::endl;
	}

	Weights deserializeToDevice(const char*& hostBuffer, size_t count)
	{
		std::cout<<LOG_CUDA<<"Deserializzo i Pesi nel Device"<<std::endl;	
		Weights w = copyToDevice(hostBuffer, count);
		hostBuffer += count * sizeof(float);
		return w;	
	}

	//valori di input e output del layer
	int mNbOutputChannels, mNbInputChannels;
	//istanza di cuDNN
	cudnnHandle_t mCudnn;
	//istanza di cuBLAST
	cublasHandle_t mCublas;
	//oggeto Pesi e bias per il calcolo dei pensi in entrata e uscita dal layer
	Weights mKernelWeights, mBiasWeights;
	//Istanza di tensori per cuDNN
	cudnnTensorDescriptor_t mSrcDescriptor, mDstDescriptor;
};
#endif
