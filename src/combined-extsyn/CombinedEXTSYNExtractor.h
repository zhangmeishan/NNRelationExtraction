#ifndef SRC_Extractor_H_
#define SRC_Extractor_H_

#include "N3LDG.h"

#include "Driver.h"
#include "Options.h"
#include "Pipe.h"
#include "Utf.h"
#include "Action.h"
#include "State.h"

class Extractor {
  public:
    Extractor(size_t memsize);
    virtual ~Extractor();

  public:
    Driver m_driver;
    Options m_options;
    Pipe m_pipe;

    vector<BucketNode> extern_nodes;
    int ext_count;

  public:
    int createAlphabet(vector<Instance>& vecInsts);

  public:
    void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile);
    void predict(vector<Instance>& inputs, vector<CResult>& outputs);
    void test(const string& testFile, const string& outputFile, const string& modelFile);

    // static training
    void getGoldActions(vector<Instance>& vecInsts, vector<vector<CAction> >& vecActions);

    void getExternalFeats(vector<Instance>& vecInsts, const string& folder);

  public:
    void writeModelFile(const string& outputModelFile);
    void loadModelFile(const string& inputModelFile);

};

#endif /* SRC_Extractor_H_ */
