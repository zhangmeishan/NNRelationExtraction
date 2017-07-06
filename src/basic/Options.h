#ifndef _PARSER_OPTIONS_
#define _PARSER_OPTIONS_

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "N3LDG.h"

const static int max_token_size = 80;
const static int max_step_size = max_token_size * (max_token_size + 1) / 2;
class Options {
public:
  int wordCutOff;
  int tagCutOff;
  dtype initRange;
  int maxIter;
  int maxNERIter;
  int batchSize;
  dtype adaEps;
  dtype adaAlpha;
  dtype regParameter;
  dtype dropProb;
  dtype delta;
  dtype clip;
  dtype decay;
  dtype scale;
  int beam;

  int charEmbSize;
  int charContext;
  int charHiddenSize;

  int wordEmbSize;
  int wordExtEmbSize;
  int tagEmbSize;
  int wordHiddenSize;
  int wordRNNHiddenSize;

  int startBeam;
  int biWordRNNHiddenSize;

  bool wordEmbFineTune;
  bool wordEmbNormalize;
  string wordEmbFile;
  int wordContext;

  int nerEmbSize;
  int labelEmbSize;

  int actionEmbSize;
  int actionHiddenSize;
  int actionRNNHiddenSize;

  int treeRNNSize;
  int state_hidden_dim;

  int verboseIter;
  bool saveIntermediate;
  int maxInstance;
  vector<string> testFiles;
  string outBest;

  int unk_strategy = 1;

  Options() {
    wordCutOff = 0;
    tagCutOff = 0;
    initRange = 0.01;
    maxIter = 1000;
    maxNERIter = 20;
    batchSize = 1;
    adaEps = 1e-6;
    adaAlpha = 0.001;
    regParameter = 0;
    dropProb = 0.0;
    delta = 0.01;
    clip = 10;
    decay = 0.5;
    scale = 4.0;
    beam = 16;

    charEmbSize = 50;
    charContext = 2;
    charHiddenSize = 100;

    wordEmbSize = 100;
    wordExtEmbSize = -1;
    tagEmbSize = 100;
    wordHiddenSize = 300;
    wordRNNHiddenSize = 300;

    startBeam = 40;
    biWordRNNHiddenSize = 100;

    wordEmbFineTune = true;
    wordEmbNormalize = false;
    wordEmbFile = "";
    wordContext = 2;

    labelEmbSize = 150;
    nerEmbSize = 50;


    actionEmbSize = 50;
    actionHiddenSize = 100;
    actionRNNHiddenSize = 100;

    treeRNNSize = 100;
    state_hidden_dim = 300;

    verboseIter = 1000;
    saveIntermediate = true;

    maxInstance = -1;
    testFiles.clear();
    outBest = "";
    unk_strategy = 1;
  }

  virtual ~Options() {

  }

  void setOptions(const vector<string> &vecOption) {
    int i = 0;
    for (; i < vecOption.size(); ++i) {
      pair<string, string> pr;
      string2pair(vecOption[i], pr, '=');
      if (pr.first == "wordCutOff")
        wordCutOff = atoi(pr.second.c_str());
      if (pr.first == "tagCutOff")
        tagCutOff = atoi(pr.second.c_str());
      if (pr.first == "initRange")
        initRange = atof(pr.second.c_str());
      if (pr.first == "maxIter")
        maxIter = atoi(pr.second.c_str());
      if (pr.first == "maxNERIter")
        maxNERIter = atoi(pr.second.c_str());
      if (pr.first == "batchSize")
        batchSize = atoi(pr.second.c_str());
      if (pr.first == "adaEps")
        adaEps = atof(pr.second.c_str());
      if (pr.first == "adaAlpha")
        adaAlpha = atof(pr.second.c_str());
      if (pr.first == "regParameter")
        regParameter = atof(pr.second.c_str());
      if (pr.first == "dropProb")
        dropProb = atof(pr.second.c_str());
      if (pr.first == "delta")
        delta = atof(pr.second.c_str());
      if (pr.first == "clip")
        clip = atof(pr.second.c_str());
      if (pr.first == "decay")
        decay = atof(pr.second.c_str());
      if (pr.first == "scale")
        scale = atof(pr.second.c_str());
      if (pr.first == "beam")
        beam = atoi(pr.second.c_str());

      if (pr.first == "charEmbSize")
        charEmbSize = atoi(pr.second.c_str());
      if (pr.first == "charHiddenSize")
        charHiddenSize = atoi(pr.second.c_str());
      if (pr.first == "charContext")
        charContext = atoi(pr.second.c_str());

      if (pr.first == "wordExtEmbSize")
        wordExtEmbSize = atoi(pr.second.c_str());
      if (pr.first == "wordEmbSize")
        wordEmbSize = atoi(pr.second.c_str());
      if (pr.first == "tagEmbSize")
        tagEmbSize = atoi(pr.second.c_str());
      if (pr.first == "wordHiddenSize")
        wordHiddenSize = atoi(pr.second.c_str());
      if (pr.first == "wordRNNHiddenSize")
        wordRNNHiddenSize = atoi(pr.second.c_str());
      if (pr.first == "startBeam")
        startBeam = atoi(pr.second.c_str());
      if (pr.first == "biWordRNNHiddenSize")
        biWordRNNHiddenSize = atoi(pr.second.c_str());
      if (pr.first == "wordEmbFineTune")
        wordEmbFineTune = (pr.second == "true") ? true : false;
      if (pr.first == "wordEmbNormalize")
        wordEmbNormalize = (pr.second == "true") ? true : false;
      if (pr.first == "wordEmbFile")
        wordEmbFile = pr.second;
      if (pr.first == "wordContext")
        wordContext = atoi(pr.second.c_str());


      if (pr.first == "labelEmbSize")
        labelEmbSize = atoi(pr.second.c_str());
      if (pr.first == "nerEmbSize")
        nerEmbSize = atoi(pr.second.c_str());

      if (pr.first == "actionEmbSize")
        actionEmbSize = atoi(pr.second.c_str());
      if (pr.first == "actionHiddenSize")
        actionHiddenSize = atoi(pr.second.c_str());
      if (pr.first == "actionRNNHiddenSize")
        actionRNNHiddenSize = atoi(pr.second.c_str());
      if (pr.first == "treeRNNSize")
        treeRNNSize = atoi(pr.second.c_str());
      if (pr.first == "state_hidden_dim")
        state_hidden_dim = atoi(pr.second.c_str());

      if (pr.first == "verboseIter")
        verboseIter = atoi(pr.second.c_str());
      if (pr.first == "saveIntermediate")
        saveIntermediate = (pr.second == "true") ? true : false;
      if (pr.first == "maxInstance")
        maxInstance = atoi(pr.second.c_str());
      if (pr.first == "testFile")
        testFiles.push_back(pr.second);
      if (pr.first == "outBest")
        outBest = pr.second;

    }
  }

  void showOptions() {
    std::cout << "wordCutOff = " << wordCutOff << std::endl;
    std::cout << "tagCutOff = " << tagCutOff << std::endl;
    std::cout << "initRange = " << initRange << std::endl;
    std::cout << "maxIter = " << maxIter << std::endl;
    std::cout << "maxNERIter = " << maxNERIter << std::endl;
    std::cout << "batchSize = " << batchSize << std::endl;
    std::cout << "adaEps = " << adaEps << std::endl;
    std::cout << "adaAlpha = " << adaAlpha << std::endl;
    std::cout << "regParameter = " << regParameter << std::endl;
    std::cout << "dropProb = " << dropProb << std::endl;
    std::cout << "delta = " << delta << std::endl;
    std::cout << "clip = " << clip << std::endl;
    std::cout << "decay = " << decay << std::endl;
    std::cout << "scale = " << scale << std::endl;
    std::cout << "beam = " << beam << std::endl;

    std::cout << "charEmbSize = " << charEmbSize << std::endl;
    std::cout << "charHiddenSize = " << charHiddenSize << std::endl;
    std::cout << "charContext = " << charContext << std::endl;

    std::cout << "wordEmbSize = " << wordEmbSize << std::endl;
    std::cout << "wordExtEmbSize = " << wordExtEmbSize << std::endl;
    std::cout << "wordHiddenSize = " << wordHiddenSize << std::endl;
    std::cout << "wordRNNHiddenSize = " << wordRNNHiddenSize << std::endl;
    std::cout << "startBeam = " << startBeam << std::endl;
    std::cout << "biWwordRNNHiddenSize = " << biWordRNNHiddenSize << std::endl;
    std::cout << "wordEmbFineTune = " << wordEmbFineTune << std::endl;
    std::cout << "wordEmbNormalize = " << wordEmbNormalize << std::endl;
    std::cout << "wordEmbFile = " << wordEmbFile << std::endl;
    std::cout << "wordContext = " << wordContext << std::endl;

    std::cout << "labelEmbSize = " << labelEmbSize << std::endl;
    std::cout << "nerEmbSize = " << nerEmbSize << std::endl;

    std::cout << "actionEmbSize = " << actionEmbSize << std::endl;
    std::cout << "actionHiddenSize = " << actionHiddenSize << std::endl;
    std::cout << "actionRNNHiddenSize = " << actionRNNHiddenSize << std::endl;

    std::cout << "treeRNNSize = " << treeRNNSize << std::endl;
    std::cout << "state_hidden_dim = " << state_hidden_dim << std::endl;

    std::cout << "verboseIter = " << verboseIter << std::endl;
    std::cout << "saveItermediate = " << saveIntermediate << std::endl;
    std::cout << "maxInstance = " << maxInstance << std::endl;
    for (int idx = 0; idx < testFiles.size(); idx++) {
      std::cout << "testFile = " << testFiles[idx] << std::endl;
    }
    std::cout << "outBest = " << outBest << std::endl;
  }

  void load(const std::string &infile) {
    ifstream inf;
    inf.open(infile.c_str());
    vector<string> vecLine;
    while (1) {
      string strLine;
      if (!my_getline(inf, strLine)) {
        break;
      }
      if (strLine.empty())
        continue;
      vecLine.push_back(strLine);
    }
    inf.close();
    setOptions(vecLine);
  }
};
#endif

