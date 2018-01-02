#include "sampleMNIST.h"

int main()
{
	GIE gie = GIE();
	if(!gie.init()){
		std::cout<< "errore"<<std::endl;
	}
	gie.plot();
};
