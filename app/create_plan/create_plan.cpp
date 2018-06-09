#include "InferenceEngine.h"

void create_plan(char *model_file, char *trained_file, char *plan_file) 
{
    InferenceEngine *engine = new InferenceEngine(model_file, trained_file);
    engine->Export(plan_file);
}

int main(int argc, char *argv[]) 
{
    create_plan(argv[1], argv[2], argv[3]);
    return 0;
}
 
