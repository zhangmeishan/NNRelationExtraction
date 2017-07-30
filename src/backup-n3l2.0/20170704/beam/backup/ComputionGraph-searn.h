#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"
#include "State.h"

struct COutput {
    PNode in;
    bool bGold;

    COutput() : in(NULL), bGold(0) {
    }

    COutput(const COutput& other) : in(other.in), bGold(other.bGold) {
    }
};

// Each model consists of two parts, building neural graph and defining output losses.
// This framework wastes memory
struct ComputionGraph : Graph {

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
    ComputionGraph() : Graph() {
        clear();
    }

    ~ComputionGraph() {
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

        //beams.resize(opts.maxlength + 1);
        //for (int idx = 0; idx < states.size(); idx++){
        //	beams[idx].resize(opts.beam);
        //}


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
    //searn
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
        static CStateItem* pGoldGenerator;
        static PNode pGoldNode;
        beam.resize(pOpts->beam);

        lastStates.clear();
        start.setInput(inst);
        lastStates.push_back(&start);

        step = 0;
        while (true) {
            pGoldGenerator = NULL;
            pGoldNode = NULL;
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
                pGenerator->computeNextScore(this, actions);
                scored_action.item = pGenerator;
                for (int idy = 0; idy < actions.size(); ++idy) {
                    scored_action.ac.set(actions[idy]); //TODO:
                    if (pGenerator->_bGold && actions[idy] == answer) {
                        scored_action.bGold = true;
                        correct_action_scored = true;
                        output.bGold = true;
                    } else {
                        scored_action.bGold = false;
                        output.bGold = false;
                        if (train)pGenerator->_nextscores.outputs[idy].val[0] += pOpts->delta;
                    }
                    scored_action.score = pGenerator->_nextscores.outputs[idy].val[0];
                    scored_action.position = idy;
                    output.in = &(pGenerator->_nextscores.outputs[idy]);
                    if (output.bGold) {
                        pGoldGenerator = pGenerator;
                        pGoldNode = output.in;
                    }
                    beam.add_elem(scored_action);
                    per_step_output.push_back(output);
                }
            }

            outputs.push_back(per_step_output);

            if (train && !correct_action_scored) { //training
                std::cout << "error during training, gold-standard action is filtered: " << step << std::endl;
                std::cout << answer.str(pOpts) << std::endl;
                for (int idx = 0; idx < lastStates.size(); idx++) {
                    pGenerator = lastStates[idx];
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
                return -1;
            }

            beam.sort_elem();

            lastStates.clear();
            if (train) {
                correct_in_beam = false;
                for (int idx = 0; idx < offset; idx++) {
                    if (beam[idx].bGold) {
                        correct_in_beam = true;
                        break;
                    }
                }
                if (correct_in_beam) {
                    for (int idx = 0; idx < offset; idx++) {
                        pGenerator = beam[idx].item;
                        action.set(beam[idx].ac);
                        pGenerator->move(&(states[step][idx]), action);
                        states[step][idx]._bGold = beam[idx].bGold;
                        states[step][idx]._score = &(pGenerator->_nextscores.outputs[beam[idx].position]);
                        lastStates.push_back(&(states[step][idx]));
                    }
                } else {
                    pGoldGenerator->move(&(states[step][0]), answer);
                    states[step][0]._bGold = true;
                    states[step][0]._score = pGoldNode;
                    lastStates.push_back(&(states[step][0]));
                }
            } else {
                for (int idx = 0; idx < offset; idx++) {
                    pGenerator = beam[idx].item;
                    action.set(beam[idx].ac);
                    pGenerator->move(&(states[step][idx]), action);
                    states[step][idx]._bGold = beam[idx].bGold;
                    states[step][idx]._score = &(pGenerator->_nextscores.outputs[beam[idx].position]);
                    lastStates.push_back(&(states[step][idx]));
                }
            }

            if (lastStates[0]->IsTerminated()) {
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

#endif /* SRC_ComputionGraph_H_ */