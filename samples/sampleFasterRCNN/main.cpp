#include <iostream>
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

//queste sono le classi che il network può trovare
const std::string CLASSES[21]{ "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

int main(int argc, char** argv){
	for(int i=0; i<rete.output_class; i++)
		printf("%sclasse = %s\n",LOG_MAIN, CLASSES[i].c_str());
	
	return 0;
}
