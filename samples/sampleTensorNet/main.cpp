#include "tensorNet.h"
#include <iostream>

class segNet: public tensorNet
{
	public:
		static segNet* create() 
		{
			segNet* s = new segNet();
			return s;
		};
	protected:
		segNet() : tensorNet()
		{
			std::cout<<"eridato tensorNet in myMind"<<std::endl;
		};
};


int main(){
	segNet* t = segNet::create();
	t->EnableProfiler();
	delete t;
	return 0;
};
