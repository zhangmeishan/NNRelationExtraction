#ifndef SRC_IncrementalNodes_H_
#define SRC_IncrementalNodes_H_

#include "ModelParams.h"

struct IncrementalNodes {
	IncLSTM1Builder* word_lstm;

public:
	IncrementalNodes() : word_lstm(NULL) {

	}

};

#endif /* SRC_IncrementalNodes_H_ */
