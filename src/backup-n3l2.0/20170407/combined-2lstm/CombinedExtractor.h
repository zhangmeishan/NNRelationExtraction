#ifndef SRC_Extractor_H_
#define SRC_Extractor_H_

#include "N3L.h"

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

  public:
    int createAlphabet( vector<Instance>& vecInsts);

  public:
    void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile);
    void predict(Instance& input, CResult& output);
    void test(const string& testFile, const string& outputFile, const string& modelFile);

    // static training
    void getGoldActions( vector<Instance>& vecInsts, vector<vector<CAction> >& vecActions);


  public:
    void writeModelFile(const string& outputModelFile);
    void loadModelFile(const string& inputModelFile);

};

#endif /* SRC_Extractor_H_ */
