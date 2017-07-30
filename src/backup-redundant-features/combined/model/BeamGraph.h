#ifndef SRC_BeamGraph_H_
#define SRC_BeamGraph_H_

#include "ModelParams.h"
#include "State.h"

// Each model consists of two parts, building neural graph and defining output losses.
// This framework wastes memory
class BeamGraphBuilder {

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
    BeamGraphBuilder() {
        clear();
    }

    ~BeamGraphBuilder() {
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
        //beams.clear();
        clearVec(outputs);
        clearVec(states);
        pModel = NULL;
        pOpts = NULL;
    }

  public:
    inline void encode(Graph* pcg, Instance& inst) {
        globalNodes.forward(pcg, inst, pOpts);
    }

  public:
    // some nodes may behave different during training and decode, for example, dropout
    inline void decode(Graph* pcg, Instance& inst, bool nerOnly, const vector<CAction>* goldAC = NULL) {
        //first step, clear node values
        clearVec(outputs);

        //second step, build graph
        vector<CStateItem*> lastStates;
        CStateItem* pGenerator;
        int step, offset;
        vector<vector<CAction> > actions; // actions to apply for a candidate
        CScoredState scored_action; // used rank actions
        COutput output;
        bool correct_action_scored;
        bool correct_in_beam;
        CAction answer, action;
        vector<COutput> per_step_output;
        NRHeap<CScoredState, CScoredState_Compare> beam;
        beam.resize(pOpts->beam);
        actions.resize(pOpts->beam);

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
            if (pcg->train) answer = (*goldAC)[step];

            for (int idx = 0; idx < lastStates.size(); idx++) {
                pGenerator = lastStates[idx];
                pGenerator->getCandidateActions(actions[idx], pOpts, pModel);
                if (pcg->train && nerOnly && pGenerator->_next_dist > 0) {
                    actions[idx].clear();
                    if (pGenerator->_bGold)actions[idx].push_back(answer);
                    else {
                        action.set(CAction::REL, 0);
                        actions[idx].push_back(action);
                    }
                }
                pGenerator->computeNextScore(pcg, actions[idx], true);
            }

            pcg->compute(); //must compute here, or we can not obtain the scores
            beam.clear();
            for (int idx = 0; idx < lastStates.size(); idx++) {
                pGenerator = lastStates[idx];
                scored_action.item = pGenerator;
                for (int idy = 0; idy < actions[idx].size(); ++idy) {
                    scored_action.ac.set(actions[idx][idy]); //TODO:
                    if (pGenerator->_bGold && actions[idx][idy] == answer) {
                        scored_action.bGold = true;
                        correct_action_scored = true;
                        output.bGold = true;
                    } else {
                        scored_action.bGold = false;
                        output.bGold = false;
                        if (pcg->train)pGenerator->_nextscores.outputs[idy].val[0] += pOpts->delta;
                    }
                    scored_action.score = pGenerator->_nextscores.outputs[idy].val[0];
                    scored_action.position = idy;
                    output.in = &(pGenerator->_nextscores.outputs[idy]);
                    beam.add_elem(scored_action);
                    per_step_output.push_back(output);
                }
            }

            outputs.push_back(per_step_output);

            if (pcg->train && !correct_action_scored) { //training
                std::cout << "error during training, gold-standard action is filtered: " << step << std::endl;
                std::cout << answer.str(pOpts) << std::endl;
                for (int idx = 0; idx < lastStates.size(); idx++) {
                    pGenerator = lastStates[idx];
                    if (pGenerator->_bGold) {
                        pGenerator->getCandidateActions(actions[idx], pOpts, pModel);
                        for (int idy = 0; idy < actions[idx].size(); ++idy) {
                            std::cout << actions[idx][idy].str(pOpts) << " ";
                        }
                        std::cout << std::endl;
                    }
                }
                return;
            }

            offset = beam.elemsize();
            if (offset == 0) { // judge correctiveness
                std::cout << "error, reach no output here, please find why" << std::endl;
                std::cout << "" << std::endl;
                return;
            }

            beam.sort_elem();
            for (int idx = 0; idx < offset; idx++) {
                pGenerator = beam[idx].item;
                action.set(beam[idx].ac);
                pGenerator->move(&(states[step][idx]), action);
                states[step][idx]._bGold = beam[idx].bGold;
                states[step][idx]._score = &(pGenerator->_nextscores.outputs[beam[idx].position]);
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

            if (pcg->train && !correct_in_beam) {
                break;
            }

            step++;
        }

        return;
    }

};

#endif /* SRC_BeamGraph_H_ */