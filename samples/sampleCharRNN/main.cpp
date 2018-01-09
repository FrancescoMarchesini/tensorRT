#include "CharRNN.h"
// To train the model that this sample uses the dataset can be found here:
// http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
//
// The ptb_w model was created retrieved from:
// https://github.com/okuchaiev/models.git
//
// The tensorflow command used to train:
// python models/tutorials/rnn/ptb/ptb_word_lm.py --data_path=data --file_prefix=ptb.char --model=charlarge --save_path=charlarge/ --seed_for_sample='consumer rep'
//
// Epochs trained: 30 
// Test perplexity: 2.697
//
// Training outputs a params.p file, which contains all of the weights in pickle format.
// This data was converted via a python script that did the following.
// Cell0 and Cell1 Linear weights matrices were concatenated as rnnweight
// Cell0 and Cell1 Linear bias vectors were concatenated as rnnbias
// Embedded is added as embed.
// softmax_w is added as rnnfcw
// softmax_b is added as rnnfcb
//
// The floating point values are converted to 32bit integer hexadecimal and written out to char-rnn.wts.


int main(){

	std::cout<<"--------------------MAIN----------------------"<<std::endl;
	CharRNN t;  
	t.init();
	std::cout<<"--------------------END----------------------"<<std::endl;
	return 0;	
};
