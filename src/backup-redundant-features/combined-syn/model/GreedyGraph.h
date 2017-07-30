#ifndef SRC_GreedyGraph_H_
#define SRC_GreedyGraph_H_

#include "ModelParams.h"
#include "State.h"

// Each model consists of two parts, building neural graph and defining output losses.
// This framework wastes memory
struct GreedyGraph : Graph {

  public:
    GlobalNodes globalNodes;
    // node instances
    CStateItem start;
    vector<CStateItem> states;
    vector<vector<COutput> > outputs;

  private:
    ModelParams *pModel;
    HyperParams *pOpts;

    // node pointers
  public:
    GreedyGraph() : Graph() {
        clear();
    }

    ~GreedyGraph() {
        clear();
    }

  public:
    //allocate enough nodes
    inline void initial(ModelParams &model, HyperParams &opts, AlignedMemoryPool *mem) {
        std::cout << "state size: " << sizeof(CStateItem) << std::endl;
        std::cout << "action node size: " << sizeof(ActionedNodes) << std::endl;
        globalNodes.resize(max_token_size, max_word_length);
        states.resize(opts.maxlength + 1);

        globalNodes.initial(model, opts, mem);
        for (int idx = 0; idx < states.size(); idx++) {
            states[idx].initial(model, opts, mem);
        }
        start.clear();
        start.initial(model, opts, mem);

        pModel = &model;
        pOpts = &opts;
    }

    inline void clear() {
        Graph::clear();
        states.clear();
        pModel = NULL;
        pOpts = NULL;
    }


  public:
    // some nodes may behave different during training and decode, for example, dropout
    inline int forward(Instance &inst, bool nerOnly, const vector<CAction> *goldAC = NULL) {
        //first step, clear node values
        if (goldAC != NULL) {
            clearValue(true);  //train
        } else {
            clearValue(false); // decode
        }

        globalNodes.forward(this, inst, pOpts);
        //second step, build graph
        static CStateItem *pGenerator;
        static int step, offset;
        static std::vector<CAction> actions; // actions to apply for a candidate
        static CScoredState scored_action; // used rank actions
        static COutput output;
        static bool correct_action_scored;
        static bool correct_in_beam;
        static CAction answer, action;
        static vector<COutput> per_step_output;
        static NRHeap<CScoredState, CScoredState_Compare> beam;
        beam.resize(pOpts->action_num + 1);

        start.setInput(inst);
        pGenerator = &start;

        step = 0;
        while (true) {
            //prepare for the next
            pGenerator->prepare(pOpts, pModel, &globalNodes);

            answer.clear();
            per_step_output.clear();
            correct_action_scored = false;
            if (train) answer = (*goldAC)[step];
            beam.clear();
            pGenerator->getCandidateActions(actions, pOpts, pModel);
            if (train && nerOnly && pGenerator->_next_dist > 0) {
                actions.clear();
                actions.push_back(answer);
            }
            pGenerator->computeNextScore(this, actions, false);
            scored_action.item = pGenerator;
            for (int idy = 0; idy < actions.size(); ++idy) {
                scored_action.ac.set(actions[idy]); //TODO:
                if (actions[idy]._label > 0) {
                    output.nPLabel = pGenerator->_nPLabel + 1;
                } else {
                    output.nPLabel = pGenerator->_nPLabel;
                }

                if (answer._label > 0) {
                    output.nGLabel = pGenerator->_nGLabel + 1;
                } else {
                    output.nGLabel = pGenerator->_nGLabel;
                }

                if (answer._label > 0 && actions[idy] == answer) {
                    output.nCLabel = pGenerator->_nCLabel + 1;
                } else {
                    output.nCLabel = pGenerator->_nCLabel;
                }
                dtype factor = 0.0;
                if (output.nGLabel + output.nPLabel > 0) {
                    factor = 1.0 - 2.0 * output.nCLabel / (output.nGLabel + output.nPLabel);
                }
                if (pGenerator->_bGold && actions[idy] == answer) {
                    scored_action.bGold = true;
                    correct_action_scored = true;
                    output.bGold = true;
                    if (factor > 0.0001 || factor < -0.0001) {
                        std::cout << "error, factor of gold state should be zero" << std::endl;
                    }
                } else {
                    scored_action.bGold = false;
                    output.bGold = false;
                    if (train)pGenerator->_nextscores.outputs[idy].val[0] += factor * pOpts->delta;
                    if (train && factor < -0.0001) {
                        std::cout << "error, factor of predicted state should be larger than zero" << std::endl;
                    }
                }
                scored_action.score = pGenerator->_nextscores.outputs[idy].val[0];
                scored_action.position = idy;
                output.in = &(pGenerator->_nextscores.outputs[idy]);
                beam.add_elem(scored_action);
                per_step_output.push_back(output);
            }

            outputs.push_back(per_step_output);

            // FIXME:
            if (train && !correct_action_scored) { //training
                std::cout << "error during training, gold-standard action is filtered: " << step << std::endl;
                return -1;
            }

            offset = beam.elemsize();
            if (offset == 0) { // judge correctiveness
                std::cout << "error, reach no output here, please find why" << std::endl;

                std::cout << "" << std::endl;
                return -1;
            }

            beam.sort_elem();
            if (train) {
                bool find_next = false;
                for (int idx = 0; idx < offset; idx++) {
                    if (beam[idx].bGold) {
                        pGenerator = beam[idx].item;
                        action.set(beam[idx].ac);
                        pGenerator->move(&(states[step]), action);
                        states[step]._bGold = beam[idx].bGold;
                        states[step]._score = &(pGenerator->_nextscores.outputs[beam[idx].position]);
                        if (action._label > 0) {
                            states[step]._nPLabel = pGenerator->_nPLabel + 1;
                        } else {
                            states[step]._nPLabel = pGenerator->_nPLabel;
                        }

                        if (answer._label > 0) {
                            states[step]._nGLabel = pGenerator->_nGLabel + 1;
                        } else {
                            states[step]._nGLabel = pGenerator->_nGLabel;
                        }

                        if (answer._label > 0 && action == answer) {
                            states[step]._nCLabel = pGenerator->_nCLabel + 1;
                        } else {
                            states[step]._nCLabel = pGenerator->_nCLabel;
                        }
                        find_next = true;
                    }
                }

                if (!find_next) {
                    std::cout << "serious bug here" << std::endl;
                    exit(0);
                }
            } else {
                pGenerator = beam[0].item;
                action.set(beam[0].ac);
                pGenerator->move(&(states[step]), action);
                states[step]._bGold = beam[0].bGold;
                states[step]._score = &(pGenerator->_nextscores.outputs[beam[0].position]);
            }
            pGenerator = &(states[step]);
            if (states[step].IsTerminated()) {
                break;
            }

            step++;
        }

        return 1;
    }


  public:
    inline void clearValue(const bool &bTrain) {
        Graph::clearValue(bTrain);
        clearVec(outputs);
    }

};

#endif /* SRC_GreedyGraph_H_ */