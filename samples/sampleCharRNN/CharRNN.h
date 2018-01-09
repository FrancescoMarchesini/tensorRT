#ifndef _CHARRNN_H__
#define _CHARRNN_H__

#include  <iostream>
#include "tensorNet.h"

#define LOG_CHR "[RNN] "

#define CHECK(status)					\
{							\
    if (status != 0)				\
    {						\
        std::cout << "Cuda failure: " << status;\
        abort();				\
    }						\
}
class CharRNN: public tensorNet
{
	public:
		void init();

		//carico il file dei pesi e lo parso
		std::map<std::string, Weights> loadWeights(const std::string file);
		//carico il modello
		std::string locateFile(const std::string& input);
	//protected:
		CharRNN(); 
};
#endif
