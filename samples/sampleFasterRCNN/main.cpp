#include <iostream>
//#include "InferenceEngine.h"
//#include "NvInfer.h"
//#include "NvCaffeParser.h"
//#include "addPlugin.h"
#include "addPlugin.h"

#define LOG_MAIN "[MAIN] "
//parametri del network
struct parametri{
	int input_c = 3;	//depth del tensore
	int input_h = 375;	//height dell'immagine
	int iput_w = 500;	//width dell'immagine
	int m_info_size = 3; // non lo so
	int output_class = 21; //classi di output
	int output_box_size = output_class * 4; //vettore dei box * 4 ??

	char* input_blob_name0 = "data";		//input 1 img
	char* input_blob_name1 = "im_info";		//input 2 labels 
	char* output_blob_name0 = "bbox_pred";	//output box
	char* output_blob_name1 = "cls_prob";	//output probabilità
	char* output_blob_name2 = "rois";		//output region of interested
	char* output_blob_name3 = "count";		//output count
	
}rete;

//using namespace::nvinfer1;
//using namespace::nvcaffeparser1;


int main(int argc, char** argv){
		
	printf("%sCreo un GIE dal modello Caffe e lo serializzo",LOG_MAIN);
/*	pluginFactory pluginFactory;
	IHostMemory *gieModelStream{ nullptr};

	const int N = 2;
	printf("%sBatchSize= %d",LOG_MAIN, N);
	
	std::string proto_path = "../../data/faster-rcnn/faster_rcnn_test_plugin.prototxt" ; 
	std::string model_path = "../../data/faster-rcnn/VGG16_faster_rcnn_final.caffemodel";  

	printf("%sInstanzione l'infere engine",LOG_MAIN);
	InferenceEngine gie = InferenceEngine(proto_path, model_path, 
			std::vector<std::string>{rete.output_blob_name0,rete.output_blob_name1,rete.output_blob_name2,rete.output_blob_name3}, 
			N,  &pluginFactory, &gieModelStream);
*/
	return 0;
};
