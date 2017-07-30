#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3L.h"
#include "Options.h"
#include <unordered_set>

struct HyperParams {
    Alphabet ner_labels;
    Alphabet rel_labels;
    unordered_map<string, unordered_set<int> > rel_dir;
    int ner_noprefix_num;
    unordered_map<string, int> word_stat;
    int maxlength;
    int action_num;
    dtype delta;
    int beam;

    dtype nnRegular; // for optimization
    dtype adaAlpha;  // for optimization
    dtype adaEps; // for optimization
    dtype dropProb;

    int char_dim;
    int word_dim;
    int word_ext_dim;
    int tag_dim;
    int action_dim;
    int ner_dim;
    int label_dim;

    int char_context;
    int char_represent_dim;
    int char_hidden_dim;

    int word_context;
    int word_represent_dim;

    int word_hidden_dim;
    int action_hidden_dim;

    int word_lstm_dim;
    int action_lstm_dim;

    int tree_input_dim;
    int tree_lstm_dim;


    int ner_state_concat_dim;
    int rel_state_concat_dim;
    int state_hidden_dim;

  public:
    HyperParams() {
        beam = 1; // TODO:
        maxlength = max_step_size;
        bAssigned = false;
    }

    void setRequared(Options &opt) {
        //please specify dictionary outside
        //please sepcify char_dim, word_dim and action_dim outside.
        beam = opt.beam;
        delta = opt.delta;
        bAssigned = true;


        ner_noprefix_num = (ner_labels.size() - 1) / 4;

        nnRegular = opt.regParameter;
        adaAlpha = opt.adaAlpha;
        adaEps = opt.adaEps;
        dropProb = opt.dropProb;

        char_dim = opt.charEmbSize;
        word_dim = opt.wordEmbSize;
        word_ext_dim = opt.wordExtEmbSize;
        tag_dim = opt.tagEmbSize;
        action_dim = opt.actionEmbSize;
        ner_dim = opt.nerEmbSize;
        label_dim = opt.labelEmbSize;

        char_context = opt.charContext;
        char_represent_dim = (2 * char_context + 1) * char_dim;
        char_hidden_dim = opt.charHiddenSize;

        word_context = opt.wordContext;
        word_represent_dim = word_dim + word_ext_dim + tag_dim + char_hidden_dim;
        word_hidden_dim = opt.wordHiddenSize;
        word_lstm_dim = opt.wordRNNHiddenSize;

        tree_input_dim = word_hidden_dim + label_dim;
        tree_lstm_dim = word_lstm_dim;


        action_hidden_dim = opt.actionHiddenSize;
        action_lstm_dim = opt.actionRNNHiddenSize;

        ner_state_concat_dim = action_lstm_dim + 2 * (2 * word_context + 1) * word_hidden_dim + 4 * word_lstm_dim;
        //rel_state_concat_dim = 10 * tree_lstm_dim;
        rel_state_concat_dim = 20 * word_lstm_dim + 2 * ner_dim;

        state_hidden_dim = opt.state_hidden_dim; //TODO:

    }

    void clear() {
        bAssigned = false;
    }

    bool bValid() {
        return bAssigned;
    }


  public:

    void print() {

    }

  private:
    bool bAssigned;
};


#endif /* SRC_HyperParams_H_ */
