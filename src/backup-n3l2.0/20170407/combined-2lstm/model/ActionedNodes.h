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
    PSubNode left_lstm2_entity;
    PSubNode right_lstm2_entity;

    //MaxPoolNode pool_left;
    //MaxPoolNode pool_middle;
    //MaxPoolNode pool_right;
    PSubNode left_lstm1_middle;
    PSubNode left_lstm1_end;
    PSubNode left_lstm1_pointi;
    PSubNode left_lstm1_pointj;

    PSubNode right_lstm1_middle;
    PSubNode right_lstm1_start;
    PSubNode right_lstm1_pointi;
    PSubNode right_lstm1_pointj;

    PSubNode left_lstm2_middle;
    PSubNode left_lstm2_end;
    PSubNode left_lstm2_pointi;
    PSubNode left_lstm2_pointj;

    PSubNode right_lstm2_middle;
    PSubNode right_lstm2_start;
    PSubNode right_lstm2_pointi;
    PSubNode right_lstm2_pointj;


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
        left_lstm2_entity.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm2_entity.init(hyparams.word_lstm_dim, -1, mem);
        ner_input_i.setParam(&(params.ner_table));
        ner_input_i.init(hyparams.ner_dim, hyparams.dropProb, mem);
        ner_input_j.setParam(&(params.ner_table));
        ner_input_j.init(hyparams.ner_dim, hyparams.dropProb, mem);
        /*
        pool_left.setParam(max_token_size);
        pool_left.init(hyparams.word_hidden2_dim, -1, mem);
        pool_middle.setParam(max_token_size);
        pool_middle.init(hyparams.word_hidden2_dim, -1, mem);
        pool_right.setParam(max_token_size);
        pool_right.init(hyparams.word_hidden2_dim, -1, mem);
        */

        left_lstm1_middle.init(hyparams.word_lstm_dim, -1, mem);
        left_lstm1_end.init(hyparams.word_lstm_dim, -1, mem);
        left_lstm1_pointi.init(hyparams.word_lstm_dim, -1, mem);
        left_lstm1_pointj.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm1_middle.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm1_start.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm1_pointi.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm1_pointj.init(hyparams.word_lstm_dim, -1, mem);

        left_lstm2_middle.init(hyparams.word_lstm_dim, -1, mem);
        left_lstm2_end.init(hyparams.word_lstm_dim, -1, mem);
        left_lstm2_pointi.init(hyparams.word_lstm_dim, -1, mem);
        left_lstm2_pointj.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm2_middle.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm2_start.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm2_pointi.init(hyparams.word_lstm_dim, -1, mem);
        right_lstm2_pointj.init(hyparams.word_lstm_dim, -1, mem);

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

            PNode  p_word_tanh_conv_context = &(atomFeat.p_word_tanh_conv1->at(atomFeat.next_i));
            states.push_back(p_word_tanh_conv_context);
            p_word_tanh_conv_context = &(atomFeat.p_word_tanh_conv2->at(atomFeat.next_i));
            states.push_back(p_word_tanh_conv_context);
            p_word_tanh_conv_context = &(atomFeat.p_word_tanh_conv3->at(atomFeat.next_i));
            states.push_back(p_word_tanh_conv_context);
            for (int context = 1; context <= opt->word_context; context++) {                              
                position = atomFeat.next_i + context;
                p_word_tanh_conv_context = (position >= 0 && position < atomFeat.word_size) ? &(atomFeat.p_word_tanh_conv1->at(position)) : &bucket_word_hidden;
                states.push_back(p_word_tanh_conv_context);
                p_word_tanh_conv_context = (position >= 0 && position < atomFeat.word_size) ? &(atomFeat.p_word_tanh_conv2->at(position)) : &bucket_word_hidden;
                states.push_back(p_word_tanh_conv_context);
                p_word_tanh_conv_context = (position >= 0 && position < atomFeat.word_size) ? &(atomFeat.p_word_tanh_conv3->at(position)) : &bucket_word_hidden;
                states.push_back(p_word_tanh_conv_context);


                position = atomFeat.next_i - context;
                p_word_tanh_conv_context = (position >= 0 && position < atomFeat.word_size) ? &(atomFeat.p_word_tanh_conv1->at(position)) : &bucket_word_hidden;
                states.push_back(p_word_tanh_conv_context);
                p_word_tanh_conv_context = (position >= 0 && position < atomFeat.word_size) ? &(atomFeat.p_word_tanh_conv2->at(position)) : &bucket_word_hidden;
                states.push_back(p_word_tanh_conv_context);
                p_word_tanh_conv_context = (position >= 0 && position < atomFeat.word_size) ? &(atomFeat.p_word_tanh_conv3->at(position)) : &bucket_word_hidden;
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

            PNode left_lstm2_node_end = (atomFeat.next_i > 0) ? &(atomFeat.p_word_left_lstm2->_hiddens[atomFeat.next_i - 1]) : &bucket;
            PNode left_lstm2_node_start = (atomFeat.last_start > 0) ? &(atomFeat.p_word_left_lstm2->_hiddens[atomFeat.last_start - 1]) : &bucket;
            left_lstm2_entity.forward(cg, left_lstm2_node_end, left_lstm2_node_start);
            states.push_back(&left_lstm2_entity);

            PNode right_lstm2_node_end = (atomFeat.last_start >= 0) ? &(atomFeat.p_word_right_lstm2->_hiddens[atomFeat.last_start]) : &bucket;
            PNode right_lstm2_node_start = (atomFeat.next_i > 0) ? &(atomFeat.p_word_right_lstm2->_hiddens[atomFeat.next_i]) : &bucket;
            right_lstm2_entity.forward(cg, right_lstm2_node_end, right_lstm2_node_start);
            states.push_back(&right_lstm2_entity);

            ner_state_represent.forward(cg, states);
            ner_state_hidden.forward(cg, &ner_state_represent);
        }
        else if (atomFeat.rel_must_o == 0) {
            int i = atomFeat.next_i;
            int start_i = atomFeat.next_i_start;
            int j = atomFeat.next_i + atomFeat.next_dist;
            int start_j = atomFeat.next_j_start;
           
            states.clear();
            /*
            PNode dep_lstm_node_i = &(atomFeat.p_dep_b2t_lstm->_hiddens[i]);
            states.push_back(dep_lstm_node_i);
            PNode dep_lstm_node_j = &(atomFeat.p_dep_b2t_lstm->_hiddens[j]);
            states.push_back(dep_lstm_node_j);
            PNode dep_lstm_node_anc = &(atomFeat.p_dep_b2t_lstm->_hiddens[anc]);
            states.push_back(dep_lstm_node_anc);
            b2t_ancestor_exclude_nodei.forward(cg, dep_lstm_node_anc, dep_lstm_node_i);
            states.push_back(&b2t_ancestor_exclude_nodei);
            b2t_ancestor_exclude_nodej.forward(cg, dep_lstm_node_anc, dep_lstm_node_j);
            states.push_back(&b2t_ancestor_exclude_nodej);

            dep_lstm_node_i = &(atomFeat.p_dep_t2b_lstm->_hiddens[i]);
            states.push_back(dep_lstm_node_i);
            dep_lstm_node_j = &(atomFeat.p_dep_t2b_lstm->_hiddens[j]);
            states.push_back(dep_lstm_node_j);
            dep_lstm_node_anc = &(atomFeat.p_dep_t2b_lstm->_hiddens[anc]);
            states.push_back(dep_lstm_node_anc);
            t2b_ancestor_exclude_nodei.forward(cg, dep_lstm_node_i, dep_lstm_node_anc);
            states.push_back(&t2b_ancestor_exclude_nodei);
            t2b_ancestor_exclude_nodej.forward(cg, dep_lstm_node_j, dep_lstm_node_anc);
            states.push_back(&t2b_ancestor_exclude_nodej);
            */

            states.clear();
           
            PNode left_lstm1_left = (start_i >= 1) ? &(atomFeat.p_word_left_lstm1->_hiddens[start_i - 1]) : &bucket;
            states.push_back(left_lstm1_left);

            PNode left_lstm1_node_i = &(atomFeat.p_word_left_lstm1->_hiddens[i]);
            left_lstm1_pointi.forward(cg, left_lstm1_node_i, left_lstm1_left);
            states.push_back(&left_lstm1_pointi);

            left_lstm1_middle.forward(cg, &(atomFeat.p_word_left_lstm1->_hiddens[start_j -1]), left_lstm1_node_i);
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

            PNode left_lstm2_left = (start_i >= 1) ? &(atomFeat.p_word_left_lstm2->_hiddens[start_i - 1]) : &bucket;
            states.push_back(left_lstm2_left);

            PNode left_lstm2_node_i = &(atomFeat.p_word_left_lstm2->_hiddens[i]);
            left_lstm2_pointi.forward(cg, left_lstm2_node_i, left_lstm2_left);
            states.push_back(&left_lstm2_pointi);

            left_lstm2_middle.forward(cg, &(atomFeat.p_word_left_lstm2->_hiddens[start_j - 1]), left_lstm2_node_i);
            states.push_back(&left_lstm2_middle);

            PNode left_lstm2_node_j = &(atomFeat.p_word_left_lstm2->_hiddens[j]);
            left_lstm2_pointj.forward(cg, left_lstm2_node_j, &(atomFeat.p_word_left_lstm2->_hiddens[start_j - 1]));
            states.push_back(&left_lstm2_pointj);

            left_lstm2_end.forward(cg, &(atomFeat.p_word_left_lstm2->_hiddens[atomFeat.word_size - 1]), left_lstm2_node_j);
            states.push_back(&left_lstm2_end);


            PNode right_lstm2_right = (j < atomFeat.word_size - 1) ? &(atomFeat.p_word_right_lstm2->_hiddens[j + 1]) : &bucket;
            states.push_back(right_lstm2_right);

            PNode right_lstm2_node_j = &(atomFeat.p_word_right_lstm2->_hiddens[j]);
            right_lstm2_pointj.forward(cg, right_lstm2_node_j, right_lstm2_right);
            states.push_back(&right_lstm2_pointj);

            right_lstm2_middle.forward(cg, &(atomFeat.p_word_right_lstm2->_hiddens[i + 1]), right_lstm2_node_j);
            states.push_back(&right_lstm2_middle);

            PNode right_lstm2_node_i = &(atomFeat.p_word_right_lstm2->_hiddens[i]);
            right_lstm2_pointi.forward(cg, right_lstm2_node_i, &(atomFeat.p_word_right_lstm2->_hiddens[i + 1]));
            states.push_back(&right_lstm2_pointi);

            right_lstm2_start.forward(cg, &(atomFeat.p_word_right_lstm2->_hiddens[0]), right_lstm2_node_i);
            states.push_back(&right_lstm2_start);

            /*
            PNode node_rep_i = &(atomFeat.p_word_tanh_conv2->at(i));
            states.push_back(node_rep_i);
            PNode node_rep_j = &(atomFeat.p_word_tanh_conv2->at(j));
            states.push_back(node_rep_j);

            pools_left.clear(); pools_middle.clear(); pools_right.clear();
            for (int idx = 0; idx < atomFeat.word_size; idx++) {
                if (idx < i) {
                    pools_left.push_back(&(atomFeat.p_word_tanh_conv2->at(idx)));
                }
                if (idx > j) {
                    pools_right.push_back(&(atomFeat.p_word_tanh_conv2->at(idx)));
                }
                if (idx > i && idx < j) {
                    pools_middle.push_back(&(atomFeat.p_word_tanh_conv2->at(idx)));
                }
            }

            if (pools_left.size() > 0) {
                pool_left.forward(cg, pools_left);
                states.push_back(&pool_left);
            }
            else {
                states.push_back(&bucket_word_hidden);
            }

            if (pools_middle.size() > 0) {
                pool_middle.forward(cg, pools_middle);
                states.push_back(&pool_middle);
            }
            else {
                states.push_back(&bucket_word_hidden);
            }

            if (pools_right.size() > 0) {
                pool_right.forward(cg, pools_right);
                states.push_back(&pool_right);
            }
            else {
                states.push_back(&bucket_word_hidden);
            }
            */

            ner_input_i.forward(cg, atomFeat.label_i);
            states.push_back(&ner_input_i);
            ner_input_j.forward(cg, atomFeat.label_j);
            states.push_back(&ner_input_j);
                   
            rel_state_represent.forward(cg, states);
            rel_state_hidden.forward(cg, &rel_state_represent);
        }
        else {
            //nothing do to
        }




		for (int idx = 0; idx < ac_num; idx++) {
			ac.set(actions[idx]);
		
			sumNodes.clear();

			string action_name = ac.str(opt);
			current_action_input[idx].forward(cg, action_name);
            if (atomFeat.next_dist == 0) {
                action_score[idx].forward(cg, &current_action_input[idx], &ner_state_hidden);
            }
            else if(atomFeat.rel_must_o == 0){
                action_score[idx].forward(cg, &current_action_input[idx], &rel_state_hidden);
            }
            else {
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
