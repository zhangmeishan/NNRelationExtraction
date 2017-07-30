#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_

#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams {

  public:
    //neural parameters
    Alphabet embeded_chars; // chars
    LookupTable char_table; // should be initialized outside
    Alphabet embeded_words; // words
    LookupTable word_table; // should be initialized outside
    Alphabet embeded_ext_words;
    LookupTable word_ext_table;
    Alphabet embeded_tags; // tags
    LookupTable tag_table; // should be initialized outside
    Alphabet embeded_actions;
    LookupTable action_table; // should be initialized outside
    Alphabet embeded_ners;
    LookupTable ner_table; // should be initialized outside

    UniParams char_tanh_conv;

    UniParams word_tanh_conv1;

    LSTM1Params word_left_lstm1; //left lstm
    LSTM1Params word_right_lstm1; //right lstm
    BiParams word_tanh_conv2;


    BiParams action_conv;
    LSTM1Params action_lstm;

    UniParams ner_state_hidden;
    UniParams rel_state_hidden;
    LookupTable scored_action_table;

  public:
    bool initial(HyperParams &opts, AlignedMemoryPool *mem) {
        char_tanh_conv.initial(opts.char_hidden_dim, opts.char_represent_dim, true, mem);

        word_tanh_conv1.initial(opts.word_hidden_dim, opts.word_represent_dim, true, mem);
        word_left_lstm1.initial(opts.word_lstm_dim, opts.word_hidden_dim, mem); //left lstm
        word_right_lstm1.initial(opts.word_lstm_dim, opts.word_hidden_dim, mem); //right lstm
        word_tanh_conv2.initial(opts.word_hidden_dim, opts.word_lstm_dim, opts.word_lstm_dim, true, mem);


        action_conv.initial(opts.action_hidden_dim, opts.action_dim, opts.action_dim, true, mem);
        action_lstm.initial(opts.action_lstm_dim, opts.action_hidden_dim, mem);

        ner_state_hidden.initial(opts.state_hidden_dim, opts.ner_state_concat_dim, true, mem);
        rel_state_hidden.initial(opts.state_hidden_dim, opts.rel_state_concat_dim, true, mem);
        scored_action_table.initial(&embeded_actions, opts.state_hidden_dim, true);
        scored_action_table.E.val.random(0.01);

        return true;
    }


    void exportModelParams(ModelUpdate &ada) {
        //neural features
        char_table.exportAdaParams(ada);
        char_tanh_conv.exportAdaParams(ada);
        word_table.exportAdaParams(ada);
        //word_ext_table.exportAdaParams(ada);
        tag_table.exportAdaParams(ada);
        action_table.exportAdaParams(ada);
        ner_table.exportAdaParams(ada);
        word_tanh_conv1.exportAdaParams(ada);
        word_left_lstm1.exportAdaParams(ada);
        word_right_lstm1.exportAdaParams(ada);
        word_tanh_conv2.exportAdaParams(ada);
        action_conv.exportAdaParams(ada);
        action_lstm.exportAdaParams(ada);
        ner_state_hidden.exportAdaParams(ada);
        rel_state_hidden.exportAdaParams(ada);
        scored_action_table.exportAdaParams(ada);
    }

    // will add it later
    void saveModel() {

    }

    void loadModel(const string &inFile) {

    }

};

#endif /* SRC_ModelParams_H_ */
