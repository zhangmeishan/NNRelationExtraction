#ifndef SRC_AtomFeatures_H_
#define SRC_AtomFeatures_H_


#include "ModelParams.h"
struct AtomFeatures {
  public:
    //ner
    short ner_next_position;
    short ner_last_start;
    short ner_last_end;
    string ner_last_label;

    //rel
    short rel_i;
    short rel_i_start;

    short rel_j;
    short rel_j_start;
    string rel_j_nerlabel;
    short rel_must_o;

    //all
    short word_size;
    bool bRel;

  public:
    //all
    LSTM1Builder* p_word_left_lstm;
    LSTM1Builder* p_word_right_lstm;
    vector<IncLSTM1Builder*> p_ner_lstms;

  public:
    void clear() {
        ner_next_position = -1;
        ner_last_start = -1;
        ner_last_end = -1;
        ner_last_label = "";

        rel_i = -1;
        rel_i_start = -1;
        rel_j = -1;
        rel_j_start = -1;
        rel_j_nerlabel = "";
        rel_must_o = -1;


        word_size = -1;
        bRel = false;

        p_word_left_lstm = NULL;
        p_word_right_lstm = NULL;
        p_ner_lstms.clear();
    }

};

#endif /* SRC_AtomFeatures_H_ */
