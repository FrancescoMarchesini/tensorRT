/*
 *	Il seguente programma carica un modello caffe .model .weight
 *	e lo serializza producendo il file plan. tramite funzione
 *	export e import
 */

#ifndef __INFERENCE_ENGINE_H__
#define __INFERENCE_ENGINE_H__

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <iostream>
#define LOG_GIE "[GIE] "

class motoreDiInferenza
{
	public:
		//distruttore
		~motoreDiInferenza();			
		
		//costruttore
		motoreDiInferenza();
		motoreDiInferenza(const std::string& model_file,
						  const std::string& wieght_file);

		//funzioni per caricare il modello ed esportarlo come file
		//per plan file si intende il modello caricato in tensorrt con
		//le relative parametrizzazione e salvato su disco come
		//file binario
		void Importa(const std::string& plan_file);
		void Esporta(const std::string& plan_file);

		//funzione per ritornare l'engine
		nvinfer1::ICudaEngine* Get() const
		{
			return engine_;
		}

		//classe logger necessaria per la costruzione dell'engine
		class Logger : public nvinfer1::ILogger
		{
			void log(Severity severity, const char* msg) override
			{
				 // suppress info-level messages
				 if (severity != Severity::kINFO)
				     std::cout <<LOG_GIE<< msg << std::endl;
			}
		}gLogger;
	
	private:
		nvinfer1::ICudaEngine* engine_;

};
#endif

