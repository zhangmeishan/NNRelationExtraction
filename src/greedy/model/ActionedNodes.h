#ifndef SRC_ActionedNodes_H_
#define SRC_ActionedNodes_H_

#include "ModelParams.h"
#include "AtomFeatures.h"

struct ActionedNodes {
    LookupNode last_ner_input;
    IncLSTM1Builder ner_lstm;
    vector<IncLSTM1Builder*> p_ner_lstms;

    PSubNode left_lstm_entity;
    PSubNode right_lstm_entity;
    PSubNode left_ner_entity;

    ConcatNode ner_state_represent;
    UniNode ner_state_hidden;


    PSubNode left_lstm_middle;
    PSubNode left_lstm_end;
    PSubNode left_lstm_pointi;
    PSubNode left_lstm_pointj;

    PSubNode right_lstm_middle;
    PSubNode right_lstm_start;
    PSubNode right_lstm_pointi;
    PSubNode right_lstm_pointj;

    PSubNode left_ner_entityi;
    PSubNode left_ner_entityj;


    ConcatNode rel_state_represent;
    UniNode rel_state_hidden;

    vector<LookupNode> current_action_input;
    vector<PDotNode> action_score;
    vector<PAddNode> outputs;

    BucketNode bucket_ner;
    BucketNode bucket_word;
    BucketNode bucket_state;
    HyperParams *opt;

  public:
    inline void initial(ModelParams &params, HyperParams &hyparams, AlignedMemoryPool *mem) {
        opt = &hyparams;

        last_ner_input.setParam(&(params.ner_table));
        last_ner_input.init(hyparams.ner_dim, hyparams.dropProb, mem);
        ner_lstm.init(&(params.ner_lstm), hyparams.dropProb, mem); //already allocated here
        p_ner_lstms.resize(max_token_size);

        left_lstm_entity.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm_entity.init(hyparams.word_lstm_dim, -1, mem);
        left_ner_entity.init(hyparams.ner_lstm_dim, -1, mem);

        ner_state_represent.init(hyparams.ner_state_concat_dim, -1, mem);
        ner_state_hidden.setParam(&(params.ner_state_hidden));
        ner_state_hidden.init(hyparams.state_hidden_dim, -1, mem);

        left_lstm_middle.init(hyparams.word_lstm_dim, -1, mem);
        left_lstm_end.init(hyparams.word_lstm_dim, -1, mem);
        left_lstm_pointi.init(hyparams.word_lstm_dim, -1, mem);
        left_lstm_pointj.init(hyparams.word_lstm_dim, -1, mem);

        right_lstm_middle.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm_start.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm_pointi.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm_pointj.init(hyparams.word_lstm_dim, -1, mem);

        left_ner_entityi.init(hyparams.ner_lstm_dim, -1, mem);
        left_ner_entityj.init(hyparams.ner_lstm_dim, -1, mem);

        rel_state_represent.init(hyparams.rel_state_concat_dim, -1, mem);
        rel_state_hidden.setParam(&(params.rel_state_hidden));
        rel_state_hidden.init(hyparams.state_hidden_dim, -1, mem);

        current_action_input.resize(hyparams.action_num);
        action_score.resize(hyparams.action_num);
        outputs.resize(hyparams.action_num);

        //neural features
        for (int idx = 0; idx < hyparams.action_num; idx++) {
            current_action_input[idx].setParam(&(params.scored_action_table));
            current_action_input[idx].init(hyparams.state_hidden_dim, -1, mem);

            action_score[idx].init(1, -1, mem);
            outputs[idx].init(1, -1, mem);
        }

        bucket_ner.init(hyparams.ner_lstm_dim, -1, mem);
        bucket_word.init(hyparams.word_lstm_dim, -1, mem);
        bucket_state.init(hyparams.state_hidden_dim, -1, mem);
    }


  public:
    inline void forward(Graph *cg, const vector<CAction> &actions, const AtomFeatures &atomFeat, PNode prevStateNode) {
        vector<PNode> sumNodes;
        CAction ac;
        int ac_num;
        int position;
        vector<PNode> states, pools_left, pools_middle, pools_right;
        ac_num = actions.size();

        bucket_ner.forward(cg, 0);
        bucket_word.forward(cg, 0);
        bucket_state.forward(cg, 0);
        PNode pseudo_ner = &(bucket_ner);
        PNode pseudo_word = &(bucket_word);
        PNode pseudo_state = &(bucket_state);

        for (int idx = 0; idx < atomFeat.word_size; idx++) {
            p_ner_lstms[idx] = atomFeat.p_ner_lstms[idx];
        }

        if (!atomFeat.bRel) {
            if (atomFeat.ner_last_end >= 0 && p_ner_lstms[atomFeat.ner_last_end] == NULL) {
                last_ner_input.forward(cg, atomFeat.ner_last_label);
                if (atomFeat.ner_last_end > 0) {
                    ner_lstm.forward(cg, &last_ner_input, p_ner_lstms[atomFeat.ner_last_end - 1]);
                    p_ner_lstms[atomFeat.ner_last_end] = &ner_lstm;
                } else if (atomFeat.ner_last_end == 0) {
                    ner_lstm.forward(cg, &last_ner_input, NULL);
                    p_ner_lstms[atomFeat.ner_last_end] = &ner_lstm;
                }
            }

            states.clear();
            if (atomFeat.ner_last_end >= 0 && !p_ner_lstms[atomFeat.ner_last_end]) {
                states.push_back(&(p_ner_lstms[atomFeat.ner_last_end]->_hidden));
                PNode ner_lstm_start = atomFeat.ner_last_start > 0 ? &(p_ner_lstms[atomFeat.ner_last_start - 1]->_hidden) : pseudo_ner;
                left_ner_entity.forward(cg, &(p_ner_lstms[atomFeat.ner_last_end]->_hidden), ner_lstm_start);
                states.push_back(&left_ner_entity);
            } else {
                states.push_back(pseudo_ner);
                states.push_back(pseudo_ner);
            }

            PNode  p_word_context = &(atomFeat.p_word_left_lstm->_hiddens[atomFeat.ner_next_position]);
            states.push_back(p_word_context);
            p_word_context = &(atomFeat.p_word_right_lstm->_hiddens[atomFeat.ner_next_position]);
            states.push_back(p_word_context);

            //entity-level
            PNode left_lstm_node_end = (atomFeat.ner_last_end >= 0) ? &(atomFeat.p_word_left_lstm->_hiddens[atomFeat.ner_last_end]) : pseudo_word;
            PNode left_lstm_node_start = (atomFeat.ner_last_start > 0) ? &(atomFeat.p_word_left_lstm->_hiddens[atomFeat.ner_last_start - 1]) : pseudo_word;
            left_lstm_entity.forward(cg, left_lstm_node_end, left_lstm_node_start);
            states.push_back(&left_lstm_entity);

            PNode right_lstm_node_end = (atomFeat.ner_last_end >= 0) ? &(atomFeat.p_word_right_lstm->_hiddens[atomFeat.ner_last_end]) : pseudo_word;
            PNode right_lstm_node_start = (atomFeat.ner_last_start > 0) ? &(atomFeat.p_word_right_lstm->_hiddens[atomFeat.ner_last_start - 1]) : pseudo_word;
            right_lstm_entity.forward(cg, right_lstm_node_end, right_lstm_node_start);
            states.push_back(&right_lstm_entity);

            ner_state_represent.forward(cg, states);
            ner_state_hidden.forward(cg, &ner_state_represent);
        } else if (atomFeat.rel_must_o == 0) {
            int i = atomFeat.rel_i;
            int start_i = atomFeat.rel_i_start;
            int j = atomFeat.rel_j;
            int start_j = atomFeat.rel_j_start;

            if (start_i >= 0 && i >= start_i && start_j > i && j >= start_j && j < atomFeat.word_size) {

            } else {
                std::cout << "" << std::endl;
            }


            states.clear();

            PNode left_lstm_left = (start_i >= 1) ? &(atomFeat.p_word_left_lstm->_hiddens[start_i - 1]) : pseudo_word;
            states.push_back(left_lstm_left);

            PNode left_lstm_node_i = &(atomFeat.p_word_left_lstm->_hiddens[i]);
            left_lstm_pointi.forward(cg, left_lstm_node_i, left_lstm_left);
            states.push_back(&left_lstm_pointi);

            left_lstm_middle.forward(cg, &(atomFeat.p_word_left_lstm->_hiddens[start_j - 1]), left_lstm_node_i);
            states.push_back(&left_lstm_middle);

            PNode left_lstm_node_j = &(atomFeat.p_word_left_lstm->_hiddens[j]);
            left_lstm_pointj.forward(cg, left_lstm_node_j, &(atomFeat.p_word_left_lstm->_hiddens[start_j - 1]));
            states.push_back(&left_lstm_pointj);

            left_lstm_end.forward(cg, &(atomFeat.p_word_left_lstm->_hiddens[atomFeat.word_size - 1]), left_lstm_node_j);
            states.push_back(&left_lstm_end);


            PNode right_lstm_right = (j < atomFeat.word_size - 1) ? &(atomFeat.p_word_right_lstm->_hiddens[j + 1]) : pseudo_word;
            states.push_back(right_lstm_right);

            PNode right_lstm_node_j = &(atomFeat.p_word_right_lstm->_hiddens[j]);
            right_lstm_pointj.forward(cg, right_lstm_node_j, right_lstm_right);
            states.push_back(&right_lstm_pointj);

            right_lstm_middle.forward(cg, &(atomFeat.p_word_right_lstm->_hiddens[i + 1]), right_lstm_node_j);
            states.push_back(&right_lstm_middle);

            PNode right_lstm_node_i = &(atomFeat.p_word_right_lstm->_hiddens[i]);
            right_lstm_pointi.forward(cg, right_lstm_node_i, &(atomFeat.p_word_right_lstm->_hiddens[i + 1]));
            states.push_back(&right_lstm_pointi);

            right_lstm_start.forward(cg, &(atomFeat.p_word_right_lstm->_hiddens[0]), right_lstm_node_i);
            states.push_back(&right_lstm_start);

            if (p_ner_lstms[j] == NULL) {
                last_ner_input.forward(cg, atomFeat.rel_j_nerlabel);
                ner_lstm.forward(cg, &last_ner_input, p_ner_lstms[j - 1]);
                p_ner_lstms[j] = &ner_lstm;
            }

            PNode ner_lstm_start = start_j > 0 ? &(p_ner_lstms[start_j - 1]->_hidden) : pseudo_ner;
            left_ner_entityj.forward(cg, &(p_ner_lstms[j]->_hidden), ner_lstm_start);
            states.push_back(&left_ner_entityj);

            ner_lstm_start = start_i > 0 ? &(p_ner_lstms[start_i - 1]->_hidden) : pseudo_ner;
            left_ner_entityi.forward(cg, &(p_ner_lstms[i]->_hidden), ner_lstm_start);
            states.push_back(&left_ner_entityi);


            rel_state_represent.forward(cg, states);
            rel_state_hidden.forward(cg, &rel_state_represent);
        } else {
            //nothing do to
        }




        for (int idx = 0; idx < ac_num; idx++) {
            ac.set(actions[idx]);

            sumNodes.clear();

            string action_name = ac.str(opt);
            current_action_input[idx].forward(cg, action_name);
            if (!atomFeat.bRel) {
                action_score[idx].forward(cg, &current_action_input[idx], &ner_state_hidden);
            } else if (atomFeat.rel_must_o == 0) {
                action_score[idx].forward(cg, &current_action_input[idx], &rel_state_hidden);
            } else {
                action_score[idx].forward(cg, &current_action_input[idx], pseudo_state);
            }
            sumNodes.push_back(&action_score[idx]);

            if (prevStateNode != NULL) {
                sumNodes.push_back(prevStateNode);
            }

            outputs[idx].forward(cg, sumNodes);
        }
    }
};


#endif /* SRC_ActionedNodes_H_ */
