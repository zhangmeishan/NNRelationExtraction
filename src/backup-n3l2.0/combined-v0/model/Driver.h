#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include "N3L.h"
#include "State.h"
#include "ActionedNodes.h"
#include "Action.h"
#include "BeamGraph.h"
#include "GreedyGraph.h"

class Driver {
  public:
    Driver(size_t memsize) : aligned_mem(memsize) {
        _bcg = NULL;
        _gcg = NULL;
        _batch = 0;
    }

    ~Driver() {
        if (_bcg != NULL)
            delete _bcg;
        _bcg = NULL;
        if (_gcg != NULL)
            delete _gcg;
        _gcg = NULL;
        _batch = 0;
    }

  public:
    BeamGraph *_bcg;
    GreedyGraph *_gcg;
    ModelParams _modelparams;  // model parameters
    HyperParams _hyperparams;

    Metric _eval;
    CheckGrad _checkgrad;
    ModelUpdate _ada;  // model update

    AlignedMemoryPool aligned_mem;
    int _batch;
    bool _useBeam;

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

        _bcg = new BeamGraph();
        _bcg->initial(_modelparams, _hyperparams, &aligned_mem);
        _gcg = new GreedyGraph();
        _gcg->initial(_modelparams, _hyperparams, &aligned_mem);

        std::cout << "allocated memory: " << aligned_mem.capacity << ", total required memory: " << aligned_mem.required
                  << ", perc = " << aligned_mem.capacity * 1.0 / aligned_mem.required << std::endl;

        setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
        _batch = 0;
        _useBeam = false;
    }


  public:
    dtype train(std::vector<Instance > &sentences, const vector<vector<CAction> > &goldACs, bool nerOnly) {
        _eval.reset();
        dtype cost = 0.0;
        int num = sentences.size();
        for (int idx = 0; idx < num; idx++) {
            if (_useBeam) {
                _bcg->forward((sentences[idx]), nerOnly, &(goldACs[idx]));
                if (nerOnly) {
                    _eval.overall_label_count += sentences[idx].words.size();
                    cost += loss_google_beam(sentences[idx].words.size());
                } else {
                    _eval.overall_label_count += goldACs[idx].size();
                    cost += loss_google_beam(-1);
                }
                _bcg->backward();
            } else {
                _gcg->forward((sentences[idx]), nerOnly, &(goldACs[idx]));
                cost += loss_google_greedy();
                if (_gcg->outputs.size() != goldACs[idx].size()) {
                    std::cout << "strange error: step not equal action_size" << std::endl;
                }
                _gcg->backward();
            }

        }

        return cost;
    }

    void decode(Instance &sentence, CResult &result) {
        if (_useBeam) {
            _bcg->forward(sentence, false);
        } else {
            _gcg->forward(sentence, false);
        }
        predict(result);
    }

    void updateModel() {
        if (_ada._params.empty()) {
            _modelparams.exportModelParams(_ada);
        }
        //_ada.rescaleGrad(1.0 / _batch);
        //_ada.update(10);
        _ada.updateAdam(10);
        _batch = 0;
    }


    void writeModel();

    void loadModel();

  private:
    dtype loss_google_beam(int upper_step) {
        int maxstep = _bcg->outputs.size();
        if (maxstep == 0) return 1.0;
        if (upper_step > 0 && maxstep > upper_step) maxstep = upper_step;
        //_eval.correct_label_count += maxstep;
        static PNode pBestNode = NULL;
        static PNode pGoldNode = NULL;
        static PNode pCurNode;
        static dtype sum, max;
        static int curcount, goldIndex;
        static vector<dtype> scores;
        dtype cost = 0.0;

        for (int step = 0; step < maxstep; step++) {
            curcount = _bcg->outputs[step].size();
            if (curcount == 1) {
                _eval.correct_label_count++;
                continue;
            }
            max = 0.0;
            goldIndex = -1;
            pBestNode = pGoldNode = NULL;
            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = _bcg->outputs[step][idx].in;
                if (pBestNode == NULL || pCurNode->val[0] > pBestNode->val[0]) {
                    pBestNode = pCurNode;
                }
                if (_bcg->outputs[step][idx].bGold) {
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
                pCurNode = _bcg->outputs[step][idx].in;
                scores[idx] = exp(pCurNode->val[0] - max);
                sum += scores[idx];
            }

            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = _bcg->outputs[step][idx].in;
                pCurNode->loss[0] += scores[idx] / sum;
                pCurNode->lossed = true;
            }

            if (pBestNode == pGoldNode)_eval.correct_label_count++;
            //_eval.overall_label_count++;
            _batch++;

            cost += -log(scores[goldIndex] / sum);

            if (std::isnan(cost)) {
                std::cout << "debug" << std::endl;
            }

        }

        return cost;
    }

    dtype loss_google_greedy() {
        int maxstep = _gcg->outputs.size();
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
            curcount = _gcg->outputs[step].size();
            if (curcount == 1)continue;
            max = 0.0;
            goldIndex = -1;
            pBestNode = pGoldNode = NULL;
            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = _gcg->outputs[step][idx].in;
                if (pBestNode == NULL || pCurNode->val[0] > pBestNode->val[0]) {
                    pBestNode = pCurNode;
                }
                if (_gcg->outputs[step][idx].bGold) {
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
                pCurNode = _gcg->outputs[step][idx].in;
                scores[idx] = exp(pCurNode->val[0] - max);
                sum += scores[idx];
            }

            for (int idx = 0; idx < curcount; idx++) {
                pCurNode = _gcg->outputs[step][idx].in;
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
        if (_useBeam) {
            int step = _bcg->outputs.size();
            _bcg->states[step - 1][0].getResults(result, _hyperparams); //TODO:
        } else {
            int step = _gcg->outputs.size();
            _gcg->states[step - 1].getResults(result, _hyperparams); //TODO:
        }
    }

  public:
    inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
        _ada._alpha = adaAlpha;
        _ada._eps = adaEps;
        _ada._reg = nnRegular;
    }

    //useBeam = true, beam searcher
    inline void setGraph(bool useBeam) {
        _useBeam = useBeam;
    }

};

#endif /* SRC_Driver_H_ */
