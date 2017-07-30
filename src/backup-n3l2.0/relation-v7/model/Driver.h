#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include "N3L.h"
#include "State.h"
#include "ActionedNodes.h"
#include "Action.h"
#include "ComputionGraph.h"

class Driver {
  public:
    Driver(size_t memsize) : aligned_mem(memsize) {
        _pcg = NULL;
        _batch = 0;
    }

    ~Driver() {
        if (_pcg != NULL)
            delete _pcg;
        _pcg = NULL;
        _batch = 0;
    }

  public:
    ComputionGraph *_pcg;
    ModelParams _modelparams;  // model parameters
    HyperParams _hyperparams;

    Metric _eval;
    CheckGrad _checkgrad;
    ModelUpdate _ada;  // model update

    AlignedMemoryPool aligned_mem;
    int _batch;

  public:

    inline void initial() {
        if (!_hyperparams.bValid()) {
            std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
            return;
        }
        if (!_modelparams.initial(_hyperparams, &aligned_mem)) {
            std::cout << "model parameter initialization Error, Please check!" << std::endl;
            return;
        }
        _hyperparams.print();

        _pcg = new ComputionGraph();
        _pcg->initial(_modelparams, _hyperparams, &aligned_mem);

        std::cout << "allocated memory: " << aligned_mem.capacity << ", total required memory: " << aligned_mem.required
                  << ", perc = " << aligned_mem.capacity * 1.0 / aligned_mem.required << std::endl;

        setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
        _batch = 0;
    }


  public:
    dtype train(std::vector<Instance > &sentences, const vector<vector<CAction> > &goldACs) {
        _eval.reset();
        dtype cost = 0.0;
        int num = sentences.size();
        for (int idx = 0; idx < num; idx++) {
            _pcg->forward((sentences[idx]), &(goldACs[idx]));

            //_batch += goldACs[idx].size();
            cost += loss_google();

            if (_pcg->outputs.size() != goldACs[idx].size()) {
                std::cout << "strange error: step not equal action_size" << std::endl;
            }

            _pcg->backward();

        }

        return cost;
    }

    void decode(Instance &sentence, CResult &result) {
        _pcg->forward(sentence);
        predict(result);
    }

    void updateModel() {
        if (_ada._params.empty()) {
            _modelparams.exportModelParams(_ada);
        }
        //_ada.update(10);
        _ada.updateAdam(10);
        _batch = 0;
    }

    void writeModel();

    void loadModel();

  private:


    dtype loss_google() {
        int maxstep = _pcg->outputs.size();
        if (maxstep == 0) return 1.0;
        //_eval.correct_label_count += maxstep;
        static PNode pBestNode = NULL;
        static PNode pGoldNode = NULL;
        static PNode pCurNode;
        static dtype sum, max;
        static int curcount, goldIndex;
        static vector<dtype> scores;
        dtype cost = 0.0;

        for (int step = 0; step < maxstep; step++) {
            curcount = _pcg->outputs[step].size();
            if (curcount == 1)continue;
            max = 0.0;
            goldIndex = -1;
            pBestNode = pGoldNode = NULL;
            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = _pcg->outputs[step][idx].in;
                if (pBestNode == NULL || pCurNode->val[0] > pBestNode->val[0]) {
                    pBestNode = pCurNode;
                }
                if (_pcg->outputs[step][idx].bGold) {
                    pGoldNode = pCurNode;
                    goldIndex = idx;
                }
            }

            if (goldIndex == -1) {
                std::cout << "impossible" << std::endl;
            }
            pGoldNode->loss[0] = -1.0;
            pGoldNode->lossed = true;

            max = pBestNode->val[0];
            sum = 0.0;
            scores.resize(curcount);
            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = _pcg->outputs[step][idx].in;
                scores[idx] = exp(pCurNode->val[0] - max);
                sum += scores[idx];
            }

            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = _pcg->outputs[step][idx].in;
                pCurNode->loss[0] += scores[idx] / sum;
                pCurNode->lossed = true;
            }

            if (pBestNode == pGoldNode)_eval.correct_label_count++;
            _eval.overall_label_count++;
            _batch++;

            cost += -log(scores[goldIndex] / sum);

            if (std::isnan(cost)) {
                std::cout << "debug" << std::endl;
            }

        }

        return cost;
    }


    void predict(CResult &result) {
        int step = _pcg->outputs.size();
        _pcg->states[step - 1].getResults(result, _hyperparams); //TODO:
    }


    inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
        _ada._alpha = adaAlpha;
        _ada._eps = adaEps;
        _ada._reg = nnRegular;
    }

};

#endif /* SRC_Driver_H_ */
