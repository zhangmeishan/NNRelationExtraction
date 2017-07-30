#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3LDG.h"
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
    int batch;

    dtype nnRegular; // for optimization
    dtype adaAlpha;  // for optimization
    dtype adaEps; // for optimization
    dtype dropProb;

    int char_dim;
    int word_dim;
    int word_ext_dim;
    int tag_dim;
    int ner_dim;

    int char_context;
    int char_represent_dim;
    int char_hidden_dim;

    int word_context;
    int word_represent_dim;

    int word_hidden_dim;

    int word_lstm_dim;
    int ner_lstm_dim;


    int ner_state_concat_dim;
    int rel_state_concat_dim;
    int state_hidden_dim;

    int ext_lstm_dim;

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
        batch = opt.batchSize;


        //ner_noprefix_num = (ner_labels.size() - 1) / 4;

        nnRegular = opt.regParameter;
        adaAlpha = opt.adaAlpha;
        adaEps = opt.adaEps;
        dropProb = opt.dropProb;

        char_dim = opt.charEmbSize;
        word_dim = opt.wordEmbSize;
        word_ext_dim = opt.wordExtEmbSize;
        tag_dim = opt.tagEmbSize;
        ner_dim = opt.nerEmbSize;

        char_context = opt.charContext;
        char_represent_dim = (2 * char_context + 1) * char_dim;
        char_hidden_dim = opt.charHiddenSize;

        word_context = opt.wordContext;
        word_represent_dim = word_dim + word_ext_dim + tag_dim + char_hidden_dim;
        word_hidden_dim = opt.wordHiddenSize;
        word_lstm_dim = opt.wordRNNHiddenSize;

        ner_lstm_dim = opt.nerRNNHiddenSize;

        ner_state_concat_dim = 2 * ner_lstm_dim + 4 * word_lstm_dim + 4 * word_lstm_dim;
        //rel_state_concat_dim = 10 * tree_lstm_dim;
        rel_state_concat_dim = 20 * word_lstm_dim + 2 * ner_lstm_dim;

        state_hidden_dim = opt.state_hidden_dim; //TODO:

        ext_lstm_dim = 300; //fixed
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
