#ifndef STATE_H_
#define STATE_H_

#include "ModelParams.h"
#include "Action.h"
#include "ActionedNodes.h"
#include "AtomFeatures.h"
#include "Utf.h"
#include "Instance.h"
#include "GlobalNodes.h"
#include "IncrementalNodes.h"

class CStateItem {
public:
    short _label;
    short _next_i; // 
    short _next_dist;
    short _next_j; // _next_j = _next_i + _next_dist;
    short _step;

    CStateItem *_prevState;
    CStateItem *_lastNERState;
    CStateItem *_lastRELState;
    Instance *_inst;
    int _word_size;

    CAction _lastAction;
    PNode _score;

    // features
    ActionedNodes _nextscores;  // features current used
    AtomFeatures _atomFeat;  //features will be used for future
    IncrementalNodes _inc_nodes;


public:
    bool _bStart; // whether it is a start state
    bool _bGold; // for train
    bool _bEnd; // whether it is an end state
    short _nGLabel;
    short _nPLabel;
    short _nCLabel;

public:
    CStateItem() {
        clear();
    }


    virtual ~CStateItem() {
        clear();
    }

    void initial(ModelParams &params, HyperParams &hyparams, AlignedMemoryPool *mem) {
        _nextscores.initial(params, hyparams, mem);
    }

    void setInput(Instance& inst) {
        _inst = &inst;
        _word_size = _inst->size();
    }

    void clear() {
        _next_i = 0;
        _next_j = 0;
        _next_dist = 0;
        _step = 0;
        //clearProperty();
        _label = invalid_label;

        _prevState = 0;
        _lastNERState = 0;
        _lastRELState = 0;
        _lastAction.clear();

        _inst = 0;
        _word_size = 0;

        _score = NULL;
        _bStart = true;
        _bGold = true;
        _bEnd = false;
        _nGLabel = 0;
        _nPLabel = 0;
        _nCLabel = 0;
    }



protected:
    inline void copyProperty2Next(CStateItem *next) {
        //memcpy(next->_labels, _labels, sizeof(short) * _step);

        if (_next_j == _word_size - 1) {
            next->_next_i = 0;
            next->_next_dist = _next_dist + 1;
            next->_next_j = _next_dist + 1;
        }
        else {
            next->_next_i = _next_i + 1;
            next->_next_dist = _next_dist;
            next->_next_j = _next_j + 1;
        }

        if (next->_next_j != next->_next_i + next->_next_dist) {
            std::cout << "State shift error" << std::endl;
            exit(0);
        }

        next->_step = _step + 1;
        next->_inst = _inst;
        next->_word_size = _word_size;
        next->_prevState = this;

        if (next->_next_dist == _word_size) {
            next->_bEnd = true;
        }
    }

    //inline void clearProperty() {
        //_labels[_step] = invalid_label; // impossible number
    //}

    // conditions
public:
    bool allow_ner() const {
        if (_next_dist == 0 && _next_i < _word_size) {
            return true;
        }
        return false;
    }

    bool allow_rel() const {
        if (_next_dist > 0 && _next_dist < _word_size) {
            return true;
        }
        return false;
    }


    bool IsTerminated() const {
        return _bEnd;
    }

public:
    short getNERId(const int& i) const {
        if (_next_dist == 0 && i >= _next_i) {
            return -1;
        }

        CStateItem *lastNERState = _lastNERState;
        while (lastNERState != NULL && lastNERState->_step >= i + 1) {
            //cannot be _next_i because of the last word of one sentence may be an entity
            if (lastNERState->_step == i + 1) {
                return lastNERState->_label;
            }
            lastNERState = lastNERState->_lastNERState;
        }
        return 0;
    }

    short getSpanStart(const int& i) const {
        if (_next_dist == 0 && i >= _next_i) {
            return -1;
        }

        CStateItem *lastNERState = _lastNERState;
        while (lastNERState != NULL && lastNERState->_step >= i + 1) {
            //cannot be _next_i because of the last word of one sentence may be an entity
            if (lastNERState->_step == i + 1) {
                break;
            }
            lastNERState = lastNERState->_lastNERState;
        }

        if (lastNERState == NULL || lastNERState->_step <= i || lastNERState->_label % 4 == 0) {
            return i;
        }
        else {
            lastNERState = lastNERState->_lastNERState;
            while (lastNERState->_label % 4 != 1) {
                lastNERState = lastNERState->_lastNERState;
            }
            return lastNERState->_step - 1;
        }

    }

    //actions
public:
    void ner(CStateItem *next, short ner_id) {
        if (!allow_ner()) {
            std::cout << "assign ner error" << std::endl;
            return;
        }

        copyProperty2Next(next);
        next->_label = ner_id;
        if (_label > 0) {
            next->_lastNERState = this;
        }
        else {
            next->_lastNERState = _lastNERState;
        }

        next->_lastRELState = _lastRELState;


        //next->clearProperty();
        next->_lastAction.set(CAction::NER, ner_id); //TODO:
    }

    void rel(CStateItem *next, short rel_id) {
        if (!allow_rel()) {
            std::cout << "assign relation error" << std::endl;
            return;
        }

        copyProperty2Next(next);
        next->_label = rel_id;

        if (_next_dist == 1 && _next_i == 0 && _label > 0) { // first relation operation
            next->_lastNERState = this;
        }
        else {
            next->_lastNERState = _lastNERState;
        }


        if ((_next_dist == 1 && _next_i == 0) || _label == 0) {
            next->_lastRELState = _lastRELState;
        }
        else {
            next->_lastRELState = this;
            if (_label == invalid_label) {
                std::cout << "strange relation" << std::endl;
            }
        }

        // next->clearProperty();
        next->_lastAction.set(CAction::REL, rel_id); //TODO:
    }

    //move, orcale
public:
    void move(CStateItem *next, const CAction &ac) {
        next->_bStart = false;
        next->_bEnd = false;
        next->_bGold = false;
        if (ac.isNER()) {
            ner(next, ac._label);
        }
        else if (ac.isREL()) {
            rel(next, ac._label); //TODO:
        }
        else {
            std::cout << "error action" << std::endl;
        }
    }

    //partial results
    void getResults(CResult &result, HyperParams &opts) const {
        result.clear();
        result.allocate(_word_size);
        const CStateItem* prev = this->_prevState;
        const CStateItem* curr = this;
        int count = 0;
        while (prev != NULL) {
            if (prev->_next_i == prev->_next_j) {
                result.ners[prev->_next_i] = opts.ner_labels.from_id(curr->_label);
            }
            else {
                short labelId = curr->_label;
                if (labelId > 0) {
                    result.relations[prev->_next_i][prev->_next_dist - 1] = opts.rel_labels.from_id(labelId);
                    result.directions[prev->_next_i][prev->_next_dist - 1] = 1;
                }
                else if (labelId < 0) {
                    result.relations[prev->_next_i][prev->_next_dist - 1] = opts.rel_labels.from_id(-labelId);
                    result.directions[prev->_next_i][prev->_next_dist - 1] = -1;
                }
                else {
                    result.relations[prev->_next_i][prev->_next_dist - 1] = opts.rel_labels.from_id(0);
                    result.directions[prev->_next_i][prev->_next_dist - 1] = 0;
                }
            }
            curr = prev;
            prev = prev->_prevState;
            count++;
        }

        if (count != _step) {
            std::cout << "step number not equal count of historical states, please check." << std::endl;
        }

        result.words = &_inst->words;
        result.tags = &_inst->tags;
        result.heads = &_inst->heads;
        result.labels = &_inst->labels;
    }

    // TODO:
    void getGoldAction(HyperParams &opts, const CResult &result, CAction &ac) const {
        if (allow_ner()) {
            ac.set(CAction::NER, opts.ner_labels.from_string(result.ners[_next_i]));
            return;
        }

        if (allow_rel()) {
            int rel_labelId = opts.rel_labels.from_string(result.relations[_next_i][_next_dist - 1]);
            if (result.directions[_next_i][_next_dist - 1] < 0) {
                rel_labelId = -rel_labelId;
            }
            ac.set(CAction::REL, rel_labelId);
            return;
        }



        ac.set(CAction::NO_ACTION, invalid_label);
        return;
    }
    //
    //	// we did not judge whether history actions are match with current state.
    void getGoldAction(const CStateItem* goldState, CAction& ac) const {
        if (_step > goldState->_step || _step < 0) {
            ac.set(CAction::NO_ACTION, -1);
            return;
        }
        const CStateItem *prevState = goldState->_prevState;
        CAction curAction = goldState->_lastAction;
        while (_step < prevState->_step) {
            curAction = prevState->_lastAction;
            prevState = prevState->_prevState;
        }
        return ac.set(curAction._code, curAction._label);
    }

    void getCandidateActions(vector<CAction> &actions, HyperParams* opts, ModelParams* params) const {
        actions.clear();
        CAction ac;

        if (IsTerminated()) {
            std::cout << "terminated, error" << std::endl;
            return;
        }

        if (allow_ner()) {
            int modvalue = _lastAction._label % 4;
            if (modvalue == 0 || modvalue == 3) { //o, e-xx, s-xx
                ac.set(CAction::NER, 0);
                if (params->embeded_actions.from_string(ac.str(opts)) >= 0)actions.push_back(ac);
                for (int i = 0; i < opts->ner_noprefix_num; i++) {
                    ac.set(CAction::NER, 4 * i + 1);  //b-xx
                    if (params->embeded_actions.from_string(ac.str(opts)) >= 0)actions.push_back(ac);
                    ac.set(CAction::NER, 4 * i + 4);  //s-xx
                    if (params->embeded_actions.from_string(ac.str(opts)) >= 0)actions.push_back(ac);
                }
            }
            else if (modvalue == 1) { //b-xx
                ac.set(CAction::NER, _lastAction._label + 1);  //m-xx
                if (params->embeded_actions.from_string(ac.str(opts)) >= 0)actions.push_back(ac);
                ac.set(CAction::NER, _lastAction._label + 2);  //e-xx
                if (params->embeded_actions.from_string(ac.str(opts)) >= 0)actions.push_back(ac);
            }
            else { // m-xx
                ac.set(CAction::NER, _lastAction._label);  //m-xx
                if (params->embeded_actions.from_string(ac.str(opts)) >= 0)actions.push_back(ac);
                ac.set(CAction::NER, _lastAction._label + 1);  //e-xx
                if (params->embeded_actions.from_string(ac.str(opts)) >= 0)actions.push_back(ac);
            }

            //gold
            //ac.set(CAction::NER, opts->ner_labels.from_string(_inst->result.ners[_next_i]));
            //actions.push_back(ac);
            return;
        }

        if (allow_rel()) {
            short label_i = getNERId(_next_i);
            short label_j = getNERId(_next_j);

            int modvalue_i = label_i % 4;
            int modvalue_j = label_j % 4;

            if (label_i > 0 && label_j > 0
                && (modvalue_i == 0 || modvalue_i == 3)
                && (modvalue_j == 0 || modvalue_j == 3)) {
                for (int idx = 0; idx < opts->rel_labels.size(); idx++) {
                    if (opts->rel_dir[opts->rel_labels.from_id(idx)].find(1) != opts->rel_dir[opts->rel_labels.from_id(idx)].end()) {
                        ac.set(CAction::REL, idx);
                        actions.push_back(ac);
                    }
                    if (opts->rel_dir[opts->rel_labels.from_id(idx)].find(-1) != opts->rel_dir[opts->rel_labels.from_id(idx)].end()) {
                        ac.set(CAction::REL, -idx);
                        actions.push_back(ac);
                    }
                }
            }
            else {
                ac.set(CAction::REL, 0);
                actions.push_back(ac);
            }


        }

    }

    //TODO: debug
    inline std::string str(HyperParams* opts) const {
        stringstream curoutstr;
        curoutstr << "score: " << _score->val[0] << " ";

        curoutstr << "actions:";
        vector<string> allacs;

        const CStateItem * curState;
        curState = this;
        while (!curState->_bStart) {
            allacs.insert(allacs.begin(), curState->_lastAction.str(opts));
            curState = curState->_prevState;
        }
        for (int idx = 0; idx < allacs.size(); idx++) {
            curoutstr << " " << allacs[idx];
        }
        return curoutstr.str();
    }


public:

    inline void computeNextScore(Graph *cg, const vector<CAction>& acs, bool useBeam) {
        if (_bStart || !useBeam) {
            _nextscores.forward(cg, acs, _atomFeat, NULL);
        }
        else {
            _nextscores.forward(cg, acs, _atomFeat, _score);
        }
    }

    inline void prepare(HyperParams* hyper_params, ModelParams* model_params, GlobalNodes* global_nodes) {
        _atomFeat.str_1AC = _bStart ? nullkey : _lastAction.str(hyper_params);
        _atomFeat.str_2AC = _prevState == 0 || _prevState->_bStart ? nullkey : _prevState->_lastAction.str(hyper_params);
        _atomFeat.p_action_lstm = _prevState == 0 ? NULL : &(_prevState->_nextscores.action_lstm);
        _atomFeat.word_size = _word_size;
        _atomFeat.last_start = -1;
        if (_next_dist == 0) {
            _atomFeat.last_start = getSpanStart(_next_i - 1);
        }

        _atomFeat.next_i = _next_i;

        _atomFeat.next_j = _next_j;
        _atomFeat.next_dist = _next_dist;

        _atomFeat.p_word_left_lstm = global_nodes == NULL ? NULL : &(global_nodes->word_left_lstm);
        _atomFeat.p_word_right_lstm = global_nodes == NULL ? NULL : &(global_nodes->word_right_lstm);
        _atomFeat.p_word_tanh_conv2 = global_nodes == NULL ? NULL : &(global_nodes->word_tanh_conv2);

        _atomFeat.rel_must_o = -1;
        _atomFeat.next_i_start = -1;
        _atomFeat.next_j_start = -1;
        _atomFeat.label_i = nullkey;
        _atomFeat.label_j = nullkey;
        if (allow_rel()) {
            short label_i = getNERId(_next_i);
            short label_j = getNERId(_next_j);


            int modvalue_i = label_i % 4;
            int modvalue_j = label_j % 4;

            if (label_i > 0 && label_j > 0
                && (modvalue_i == 0 || modvalue_i == 3)
                && (modvalue_j == 0 || modvalue_j == 3)) {
                _atomFeat.rel_must_o = 0;
                _atomFeat.next_i_start = getSpanStart(_next_i);
                _atomFeat.next_j_start = getSpanStart(_next_j);
                _atomFeat.label_i = cleanLabel(hyper_params->ner_labels.from_id(label_i));
                _atomFeat.label_j = cleanLabel(hyper_params->ner_labels.from_id(label_j));
            }
            else {
                _atomFeat.rel_must_o = 1;
            }
        }
    }
};

class CScoredState {
public:
    CStateItem *item;
    CAction ac;
    dtype score;
    bool bGold;
    int position;

public:
    CScoredState() : item(0), score(0), ac(0, -1), bGold(0), position(-1) {
    }

    CScoredState(const CScoredState &other) : item(other.item), score(other.score), ac(other.ac), bGold(other.bGold),
        position(other.position) {

    }

public:
    bool operator<(const CScoredState &a1) const {
        return score < a1.score;
    }

    bool operator>(const CScoredState &a1) const {
        return score > a1.score;
    }

    bool operator<=(const CScoredState &a1) const {
        return score <= a1.score;
    }

    bool operator>=(const CScoredState &a1) const {
        return score >= a1.score;
    }
};

class CScoredState_Compare {
public:
    int operator()(const CScoredState &o1, const CScoredState &o2) const {
        if (o1.score < o2.score)
            return -1;
        else if (o1.score > o2.score)
            return 1;
        else
            return 0;
    }
};


struct COutput {
    PNode in;
    short nGLabel;
    short nPLabel;
    short nCLabel;
    bool bGold;

    COutput() : in(NULL), nGLabel(0), nPLabel(0), nCLabel(0), bGold(false) {
    }

    COutput(const COutput &other) : in(other.in), nGLabel(other.nGLabel), nPLabel(other.nPLabel), nCLabel(other.nCLabel), bGold(other.bGold) {
    }
};

#endif /* STATE_H_ */