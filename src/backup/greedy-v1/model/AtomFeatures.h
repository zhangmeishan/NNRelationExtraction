#ifndef SRC_AtomFeatures_H_
#define SRC_AtomFeatures_H_


#include "ModelParams.h"
struct AtomFeatures {
public:
  string str_1AC;
  string str_2AC;
  short next_i;
  short next_i_start;
  short next_j;
  short next_j_start;
  short next_dist;
  short word_size;
  short rel_must_o;

public:
  IncLSTM1Builder* p_action_lstm;
  LSTM1Builder* p_word_left_lstm;
  LSTM1Builder* p_word_right_lstm;
  vector<BiNode>* p_word_tanh_conv1;

public:
  void clear(){
    str_1AC = "";
    str_2AC = "";
    word_size = -1;
    next_i = -1;
    next_i_start = -1;
    next_j = -1;
    next_j_start = -1;
    next_dist = -1;
    rel_must_o = -1;
    p_action_lstm = NULL;
    p_word_left_lstm = NULL;
    p_word_right_lstm = NULL;
    p_word_tanh_conv1 = NULL;
  }

};

#endif /* SRC_AtomFeatures_H_ */
