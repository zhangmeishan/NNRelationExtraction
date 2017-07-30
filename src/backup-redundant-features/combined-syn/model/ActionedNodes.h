#ifndef SRC_ActionedNodes_H_
#define SRC_ActionedNodes_H_

#include "ModelParams.h"
#include "AtomFeatures.h"

struct ActionedNodes {
    LookupNode last_action_input;
    LookupNode last2_action_input;
    BiNode action_conv;
    IncLSTM1Builder action_lstm;

    ConcatNode ner_state_represent;
    UniNode ner_state_hidden;

    PSubNode left_lstm1_entity;
    PSubNode right_lstm1_entity;

    PSubNode left_lstm1_middle;
    PSubNode left_lstm1_end;
    PSubNode left_lstm1_pointi;
    PSubNode left_lstm1_pointj;

    PSubNode right_lstm1_middle;
    PSubNode right_lstm1_start;
    PSubNode right_lstm1_pointi;
    PSubNode right_lstm1_pointj;


    LookupNode ner_input_i;
    LookupNode ner_input_j;
    ConcatNode rel_state_represent;
    UniNode rel_state_hidden;

    vector<LookupNode> current_action_input;
    vector<PDotNode> action_score;
    vector<SPAddNode> outputs;

    Node bucket;
    Node bucket_word_hidden;
    Node bucket_state_hidden;
    HyperParams *opt;

    //extern features
    PSubNode ext_left_lstm_entity;
    PSubNode ext_right_lstm_entity;

    PSubNode ext_left_lstm_middle;
    PSubNode ext_left_lstm_end;
    PSubNode ext_left_lstm_pointi;
    PSubNode ext_left_lstm_pointj;

    PSubNode ext_right_lstm_middle;
    PSubNode ext_right_lstm_start;
    PSubNode ext_right_lstm_pointi;
    PSubNode ext_right_lstm_pointj;
  public:
    inline void initial(ModelParams &params, HyperParams &hyparams, AlignedMemoryPool *mem) {
        opt = &hyparams;

        last_action_input.setParam(&(params.action_table));
        last_action_input.init(hyparams.action_dim, hyparams.dropProb, mem);
        last2_action_input.setParam(&(params.action_table));
        last2_action_input.init(hyparams.action_dim, hyparams.dropProb, mem);
        action_conv.setParam(&(params.action_conv));
        action_conv.init(hyparams.action_hidden_dim, hyparams.dropProb, mem);
        action_lstm.init(&(params.action_lstm), hyparams.dropProb, mem); //already allocated here

        ner_state_represent.init(hyparams.ner_state_concat_dim, -1, mem);
        ner_state_hidden.setParam(&(params.ner_state_hidden));
        ner_state_hidden.init(hyparams.state_hidden_dim, -1, mem);

        left_lstm1_entity.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm1_entity.init(hyparams.word_lstm_dim, -1, mem);

        ner_input_i.setParam(&(params.ner_table));
        ner_input_i.init(hyparams.ner_dim, hyparams.dropProb, mem);
        ner_input_j.setParam(&(params.ner_table));
        ner_input_j.init(hyparams.ner_dim, hyparams.dropProb, mem);

        left_lstm1_middle.init(hyparams.word_lstm_dim, -1, mem);
        left_lstm1_end.init(hyparams.word_lstm_dim, -1, mem);
        left_lstm1_pointi.init(hyparams.word_lstm_dim, -1, mem);
        left_lstm1_pointj.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm1_middle.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm1_start.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm1_pointi.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm1_pointj.init(hyparams.word_lstm_dim, -1, mem);

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

        bucket.init(hyparams.word_lstm_dim, -1, mem);
        bucket.set_bucket();
        bucket_word_hidden.init(hyparams.word_hidden_dim, -1, mem);
        bucket_word_hidden.set_bucket();
        bucket_state_hidden.init(hyparams.state_hidden_dim, -1, mem);
        bucket_state_hidden.set_bucket();
        //extern features
        ext_left_lstm_entity.init(hyparams.word_lstm_dim, -1, mem);
        ext_right_lstm_entity.init(hyparams.word_lstm_dim, -1, mem);

        ext_left_lstm_middle.init(hyparams.word_lstm_dim, -1, mem);
        ext_left_lstm_end.init(hyparams.word_lstm_dim, -1, mem);
        ext_left_lstm_pointi.init(hyparams.word_lstm_dim, -1, mem);
        ext_left_lstm_pointj.init(hyparams.word_lstm_dim, -1, mem);
        ext_right_lstm_middle.init(hyparams.word_lstm_dim, -1, mem);
        ext_right_lstm_start.init(hyparams.word_lstm_dim, -1, mem);
        ext_right_lstm_pointi.init(hyparams.word_lstm_dim, -1, mem);
        ext_right_lstm_pointj.init(hyparams.word_lstm_dim, -1, mem);
    }


  public:
    inline void forward(Graph *cg, const vector<CAction> &actions, const AtomFeatures &atomFeat, PNode prevStateNode) {
        static vector<PNode> sumNodes;
        static CAction ac;
        static int ac_num;
        static int position;
        static vector<PNode> states, pools_left, pools_middle, pools_right;
        ac_num = actions.size();

        if (atomFeat.next_dist == 0) {
            last2_action_input.forward(cg, atomFeat.str_2AC);
            last_action_input.forward(cg, atomFeat.str_1AC);

            action_conv.forward(cg, &last2_action_input, &last_action_input);
            action_lstm.forward(cg, &action_conv, atomFeat.p_action_lstm);

            states.clear();
            states.push_back(&(action_lstm._hidden));

            PNode  p_word_tanh_conv_context = &(atomFeat.p_word_tanh_conv2->at(atomFeat.next_i));
            states.push_back(p_word_tanh_conv_context);

            for (int context = 1; context <= opt->word_context; context++) {
                position = atomFeat.next_i + context;
                p_word_tanh_conv_context = (position >= 0 && position < atomFeat.word_size) ? &(atomFeat.p_word_tanh_conv2->at(position)) : &bucket_word_hidden;
                states.push_back(p_word_tanh_conv_context);

                position = atomFeat.next_i - context;
                p_word_tanh_conv_context = (position >= 0 && position < atomFeat.word_size) ? &(atomFeat.p_word_tanh_conv2->at(position)) : &bucket_word_hidden;
                states.push_back(p_word_tanh_conv_context);
            }

            //extern features
            p_word_tanh_conv_context = &(atomFeat.p_ext_word_tanh_conv->at(atomFeat.next_i));
            states.push_back(p_word_tanh_conv_context);

            for (int context = 1; context <= opt->word_context; context++) {
                position = atomFeat.next_i + context;
                p_word_tanh_conv_context = (position >= 0 && position < atomFeat.word_size) ? &(atomFeat.p_ext_word_tanh_conv->at(position)) : &bucket_word_hidden;
                states.push_back(p_word_tanh_conv_context);

                position = atomFeat.next_i - context;
                p_word_tanh_conv_context = (position >= 0 && position < atomFeat.word_size) ? &(atomFeat.p_ext_word_tanh_conv->at(position)) : &bucket_word_hidden;
                states.push_back(p_word_tanh_conv_context);
            }

            //entity-level
            PNode left_lstm1_node_end = (atomFeat.next_i > 0) ? &(atomFeat.p_word_left_lstm1->_hiddens[atomFeat.next_i - 1]) : &bucket;
            PNode left_lstm1_node_start = (atomFeat.last_start > 0) ? &(atomFeat.p_word_left_lstm1->_hiddens[atomFeat.last_start - 1]) : &bucket;
            left_lstm1_entity.forward(cg, left_lstm1_node_end, left_lstm1_node_start);
            states.push_back(&left_lstm1_entity);

            PNode right_lstm1_node_end = (atomFeat.last_start >= 0) ? &(atomFeat.p_word_right_lstm1->_hiddens[atomFeat.last_start]) : &bucket;
            PNode right_lstm1_node_start = (atomFeat.next_i > 0) ? &(atomFeat.p_word_right_lstm1->_hiddens[atomFeat.next_i]) : &bucket;
            right_lstm1_entity.forward(cg, right_lstm1_node_end, right_lstm1_node_start);
            states.push_back(&right_lstm1_entity);

            left_lstm1_node_end = (atomFeat.next_i > 0) ? &(atomFeat.p_ext_word_left_lstm->_hiddens[atomFeat.next_i - 1]) : &bucket;
            left_lstm1_node_start = (atomFeat.last_start > 0) ? &(atomFeat.p_ext_word_left_lstm->_hiddens[atomFeat.last_start - 1]) : &bucket;
            ext_left_lstm_entity.forward(cg, left_lstm1_node_end, left_lstm1_node_start);
            states.push_back(&ext_left_lstm_entity);

            right_lstm1_node_end = (atomFeat.last_start >= 0) ? &(atomFeat.p_ext_word_right_lstm->_hiddens[atomFeat.last_start]) : &bucket;
            right_lstm1_node_start = (atomFeat.next_i > 0) ? &(atomFeat.p_ext_word_right_lstm->_hiddens[atomFeat.next_i]) : &bucket;
            ext_right_lstm_entity.forward(cg, right_lstm1_node_end, right_lstm1_node_start);
            states.push_back(&ext_right_lstm_entity);

            ner_state_represent.forward(cg, states);
            ner_state_hidden.forward(cg, &ner_state_represent);
        } else if (atomFeat.rel_must_o == 0) {
            int i = atomFeat.next_i;
            int start_i = atomFeat.next_i_start;
            int j = atomFeat.next_i + atomFeat.next_dist;
            int start_j = atomFeat.next_j_start;

            states.clear();

            PNode left_lstm1_left = (start_i >= 1) ? &(atomFeat.p_word_left_lstm1->_hiddens[start_i - 1]) : &bucket;
            states.push_back(left_lstm1_left);

            PNode left_lstm1_node_i = &(atomFeat.p_word_left_lstm1->_hiddens[i]);
            left_lstm1_pointi.forward(cg, left_lstm1_node_i, left_lstm1_left);
            states.push_back(&left_lstm1_pointi);

            left_lstm1_middle.forward(cg, &(atomFeat.p_word_left_lstm1->_hiddens[start_j - 1]), left_lstm1_node_i);
            states.push_back(&left_lstm1_middle);

            PNode left_lstm1_node_j = &(atomFeat.p_word_left_lstm1->_hiddens[j]);
            left_lstm1_pointj.forward(cg, left_lstm1_node_j, &(atomFeat.p_word_left_lstm1->_hiddens[start_j - 1]));
            states.push_back(&left_lstm1_pointj);

            left_lstm1_end.forward(cg, &(atomFeat.p_word_left_lstm1->_hiddens[atomFeat.word_size - 1]), left_lstm1_node_j);
            states.push_back(&left_lstm1_end);


            PNode right_lstm1_right = (j < atomFeat.word_size - 1) ? &(atomFeat.p_word_right_lstm1->_hiddens[j + 1]) : &bucket;
            states.push_back(right_lstm1_right);

            PNode right_lstm1_node_j = &(atomFeat.p_word_right_lstm1->_hiddens[j]);
            right_lstm1_pointj.forward(cg, right_lstm1_node_j, right_lstm1_right);
            states.push_back(&right_lstm1_pointj);

            right_lstm1_middle.forward(cg, &(atomFeat.p_word_right_lstm1->_hiddens[i + 1]), right_lstm1_node_j);
            states.push_back(&right_lstm1_middle);

            PNode right_lstm1_node_i = &(atomFeat.p_word_right_lstm1->_hiddens[i]);
            right_lstm1_pointi.forward(cg, right_lstm1_node_i, &(atomFeat.p_word_right_lstm1->_hiddens[i + 1]));
            states.push_back(&right_lstm1_pointi);

            right_lstm1_start.forward(cg, &(atomFeat.p_word_right_lstm1->_hiddens[0]), right_lstm1_node_i);
            states.push_back(&right_lstm1_start);


            ner_input_i.forward(cg, atomFeat.label_i);
            states.push_back(&ner_input_i);
            ner_input_j.forward(cg, atomFeat.label_j);
            states.push_back(&ner_input_j);

            //extern features
            PNode ext_left_lstm_left = (start_i >= 1) ? &(atomFeat.p_ext_word_left_lstm->_hiddens[start_i - 1]) : &bucket;
            states.push_back(ext_left_lstm_left);

            PNode ext_left_lstm_node_i = &(atomFeat.p_ext_word_left_lstm->_hiddens[i]);
            ext_left_lstm_pointi.forward(cg, ext_left_lstm_node_i, ext_left_lstm_left);
            states.push_back(&ext_left_lstm_pointi);

            ext_left_lstm_middle.forward(cg, &(atomFeat.p_ext_word_left_lstm->_hiddens[start_j - 1]), ext_left_lstm_node_i);
            states.push_back(&ext_left_lstm_middle);

            PNode ext_left_lstm_node_j = &(atomFeat.p_ext_word_left_lstm->_hiddens[j]);
            ext_left_lstm_pointj.forward(cg, ext_left_lstm_node_j, &(atomFeat.p_ext_word_left_lstm->_hiddens[start_j - 1]));
            states.push_back(&ext_left_lstm_pointj);

            ext_left_lstm_end.forward(cg, &(atomFeat.p_ext_word_left_lstm->_hiddens[atomFeat.word_size - 1]), ext_left_lstm_node_j);
            states.push_back(&ext_left_lstm_end);

            PNode ext_right_lstm_right = (j < atomFeat.word_size - 1) ? &(atomFeat.p_ext_word_right_lstm->_hiddens[j + 1]) : &bucket;
            states.push_back(ext_right_lstm_right);

            PNode ext_right_lstm_node_j = &(atomFeat.p_ext_word_right_lstm->_hiddens[j]);
            ext_right_lstm_pointj.forward(cg, ext_right_lstm_node_j, ext_right_lstm_right);
            states.push_back(&ext_right_lstm_pointj);

            ext_right_lstm_middle.forward(cg, &(atomFeat.p_ext_word_right_lstm->_hiddens[i + 1]), ext_right_lstm_node_j);
            states.push_back(&ext_right_lstm_middle);

            PNode ext_right_lstm_node_i = &(atomFeat.p_ext_word_right_lstm->_hiddens[i]);
            ext_right_lstm_pointi.forward(cg, ext_right_lstm_node_i, &(atomFeat.p_ext_word_right_lstm->_hiddens[i + 1]));
            states.push_back(&ext_right_lstm_pointi);

            ext_right_lstm_start.forward(cg, &(atomFeat.p_ext_word_right_lstm->_hiddens[0]), ext_right_lstm_node_i);
            states.push_back(&ext_right_lstm_start);
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
            if (atomFeat.next_dist == 0) {
                action_score[idx].forward(cg, &current_action_input[idx], &ner_state_hidden);
            } else if (atomFeat.rel_must_o == 0) {
                action_score[idx].forward(cg, &current_action_input[idx], &rel_state_hidden);
            } else {
                action_score[idx].forward(cg, &current_action_input[idx], &bucket_state_hidden);
            }
            sumNodes.push_back(&action_score[idx]);

            if (prevStateNode != NULL) {
                sumNodes.push_back(prevStateNode);
            }

            outputs[idx].forward(cg, sumNodes, 0);
        }
    }
};


#endif /* SRC_ActionedNodes_H_ */
