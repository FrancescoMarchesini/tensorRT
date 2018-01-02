#include "sampleGoogleNet.h"

int main(){
	GIEgoogleNET gie = GIEgoogleNET();
	if(gie.lunch()){
		std::cout<<"bella inferenza"<<std::endl;
	}

	return 0;
};
