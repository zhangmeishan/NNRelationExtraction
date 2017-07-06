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
  short last_start;
  string label_i;
  string label_j;
  short word_size;
  short rel_must_o;

public:
  IncLSTM1Builder* p_action_lstm;
  LSTM1Builder* p_word_left_lstm1;
  LSTM1Builder* p_word_right_lstm1;
  LSTM1Builder* p_word_left_lstm2;
  LSTM1Builder* p_word_right_lstm2;
  vector<UniNode>* p_word_tanh_conv1;
  vector<BiNode>* p_word_tanh_conv2;
  vector<BiNode>* p_word_tanh_conv3;

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
    last_start = -1;
    rel_must_o = -1;
    label_i = "";
    label_j = "";
    p_action_lstm = NULL;
    p_word_left_lstm1 = NULL;
    p_word_right_lstm1 = NULL;
    p_word_left_lstm2 = NULL;
    p_word_right_lstm2 = NULL;
    p_word_tanh_conv1 = NULL;
    p_word_tanh_conv2 = NULL;
    p_word_tanh_conv3 = NULL;
  }

};

#endif /* SRC_AtomFeatures_H_ */
