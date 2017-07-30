#ifndef SRC_BeamGraph_H_
#define SRC_BeamGraph_H_

#include "ModelParams.h"
#include "State.h"

// Each model consists of two parts, building neural graph and defining output losses.
// This framework wastes memory
struct BeamGraph : Graph {

  public:
    GlobalNodes globalNodes;
    // node instances
    CStateItem start;
    vector<vector<CStateItem> > states;
    vector<vector<COutput> > outputs;

  private:
    ModelParams *pModel;
    HyperParams *pOpts;

    // node pointers
  public:
    BeamGraph() : Graph() {
        clear();
    }

    ~BeamGraph() {
        clear();
    }

  public:
    //allocate enough nodes
    inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem) {
        std::cout << "state size: " << sizeof(CStateItem) << std::endl;
        std::cout << "action node size: " << sizeof(ActionedNodes) << std::endl;
        globalNodes.resize(max_token_size, max_word_length);
        states.resize(opts.maxlength + 1);

        globalNodes.initial(model, opts, mem);
        for (int idx = 0; idx < states.size(); idx++) {
            states[idx].resize(opts.beam);
            for (int idy = 0; idy < states[idx].size(); idy++) {
                states[idx][idy].initial(model, opts, mem);
            }
        }
        start.clear();
        start.initial(model, opts, mem);


        pModel = &model;
        pOpts = &opts;
    }

    inline void clear() {
        Graph::clear();
        //beams.clear();

        clearVec(states);
        pModel = NULL;
        pOpts = NULL;
    }


  public:
    // some nodes may behave different during training and decode, for example, dropout
    inline int forward( Instance& inst, bool nerOnly, const vector<CAction>* goldAC = NULL) {
        //first step, clear node values
        if (goldAC != NULL) {
            clearValue(true);  //train
        } else {
            clearValue(false); // decode
        }

        globalNodes.forward(this, inst, pOpts);
        //second step, build graph
        static vector<CStateItem*> lastStates;
        static CStateItem* pGenerator;
        static int step, offset;
        static std::vector<CAction> actions; // actions to apply for a candidate
        static CScoredState scored_action; // used rank actions
        static COutput output;
        static bool correct_action_scored;
        static bool correct_in_beam;
        static CAction answer, action;
        static vector<COutput> per_step_output;
        static NRHeap<CScoredState, CScoredState_Compare> beam;
        beam.resize(pOpts->beam);

        lastStates.clear();
        start.setInput(inst);
        lastStates.push_back(&start);

        step = 0;
        while (true) {
            //prepare for the next
            for (int idx = 0; idx < lastStates.size(); idx++) {
                pGenerator = lastStates[idx];
                pGenerator->prepare(pOpts, pModel, &globalNodes);
            }

            answer.clear();
            per_step_output.clear();
            correct_action_scored = false;
            if (train) answer = (*goldAC)[step];
            beam.clear();
            for (int idx = 0; idx < lastStates.size(); idx++) {
                pGenerator = lastStates[idx];
                pGenerator->getCandidateActions(actions, pOpts, pModel);
                if (train && nerOnly && pGenerator->_next_dist > 0) {
                    actions.clear();
                    if(pGenerator->_bGold)actions.push_back(answer);
                    else {
                        action.set(CAction::REL, 0);
                        actions.push_back(action);
                    }
                }
                pGenerator->computeNextScore(this, actions, true);
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
            }

            outputs.push_back(per_step_output);

            if (train && !correct_action_scored) { //training
                std::cout << "error during training, gold-standard action is filtered: " << step << std::endl;
//				for (int idx = 0; idx < inst.size(); idx++) {
//					std::cout << inst.words[idx] << "\t" << inst.tags[idx] << "\t" << inst.result.heads[idx] << "\t" << inst.result.labels[idx] << endl;
//				}
//				std::cout << std::endl;
                std::cout << answer.str(pOpts) << std::endl;
                for (int idx = 0; idx < lastStates.size(); idx++) {
                    pGenerator = lastStates[idx];
//					std::cout << pGenerator->str(pOpts) << std::endl;
                    if (pGenerator->_bGold) {
                        pGenerator->getCandidateActions(actions, pOpts, pModel);
                        for (int idy = 0; idy < actions.size(); ++idy) {
                            std::cout << actions[idy].str(pOpts) << " ";
                        }
                        std::cout << std::endl;
                    }
                }
                return -1;
            }

            offset = beam.elemsize();
            if (offset == 0) { // judge correctiveness
                std::cout << "error, reach no output here, please find why" << std::endl;
//				for (int idx = 0; idx < pCharacters->size(); idx++) {
//					std::cout << (*pCharacters)[idx] << std::endl;
//				}
                std::cout << "" << std::endl;
                return -1;
            }

            beam.sort_elem();
            for (int idx = 0; idx < offset; idx++) {
                pGenerator = beam[idx].item;
                action.set(beam[idx].ac);
                pGenerator->move(&(states[step][idx]), action);
                states[step][idx]._bGold = beam[idx].bGold;
                states[step][idx]._score = &(pGenerator->_nextscores.outputs[beam[idx].position]);
                if (action._label > 0) {
                    states[step][idx]._nPLabel = pGenerator->_nPLabel + 1;
                } else {
                    states[step][idx]._nPLabel = pGenerator->_nPLabel;
                }

                if (answer._label > 0) {
                    states[step][idx]._nGLabel = pGenerator->_nGLabel + 1;
                } else {
                    states[step][idx]._nGLabel = pGenerator->_nGLabel;
                }

                if (answer._label > 0 && action == answer) {
                    states[step][idx]._nCLabel = pGenerator->_nCLabel + 1;
                } else {
                    states[step][idx]._nCLabel = pGenerator->_nCLabel;
                }
            }

            if (states[step][0].IsTerminated()) {
                break;
            }

            //for next step
            lastStates.clear();
            correct_in_beam = false;
            for (int idx = 0; idx < offset; idx++) {
                lastStates.push_back(&(states[step][idx]));
                if (lastStates[idx]->_bGold) {
                    correct_in_beam = true;
                }
            }

            if (train && !correct_in_beam) {
                break;
            }

            step++;
        }

        return 1;
    }


  public:
    inline void clearValue(const bool& bTrain) {
        Graph::clearValue(bTrain);
        clearVec(outputs);
    }

};

#endif /* SRC_BeamGraph_H_ */