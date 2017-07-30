#ifndef STATE_H_
#define STATE_H_

#include "ModelParams.h"
#include "Action.h"
#include "ActionedNodes.h"
#include "AtomFeatures.h"
#include "Utf.h"
#include "Instance.h"
#include "GlobalNodes.h"

class CStateItem {
  public:
    short _label;
    short _current_i; //
    short _current_j; //
    short _step;
    short _labels[max_token_size][max_token_size];

    CStateItem *_prevState;
    Instance *_inst;
    int _word_size;

    CAction _lastAction;
    PNode _score;

    // features
    ActionedNodes _nextscores;  // features current used
    AtomFeatures _atomFeat;  //features will be used for future


  public:
    bool _bStart; // whether it is a start state
    bool _bGold; // for train
    bool _bEnd; // whether it is an end state


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
        _current_i = -1;
        _current_j = -1;
        _step = 0;
        _label = invalid_label;

        for (int idx = 0; idx < max_token_size; idx++) {
            for (int idy = 0; idy < max_token_size; idy++) {
                _labels[idx][idy] = -1;
            }
        }

        _prevState = 0;
        _lastAction.clear();

        _inst = 0;
        _word_size = 0;

        _score = NULL;
        _bStart = true;
        _bGold = true;
        _bEnd = false;
    }



  protected:
    inline void copyProperty2Next(CStateItem *next) {
        next->_current_i = _current_i;
        next->_current_j = _current_j;

        for (int idx = 0; idx < max_token_size; idx++) {
            memcpy(next->_labels[idx], _labels[idx], sizeof(short) * (max_token_size));
        }

        // do not need modification any more
        next->_inst = _inst;
        next->_word_size = _word_size;
        next->_step = _step + 1;
        next->_prevState = this;
    }

    // conditions
  public:
    bool allow_ner() const {
        short next_j = _current_j + 1;
        short next_i = _current_i + 1;
        short next_dist = _current_j - _current_i;

        if (next_j == _word_size) {
            next_i = 0;
            next_dist = next_dist + 1;
            next_j = next_dist;
        }

        if (next_dist == 0 && next_j < _word_size && next_j >= 0) {
            return true;
        }
        return false;
    }

    bool allow_rel() const {
        short next_j = _current_j + 1;
        short next_i = _current_i + 1;
        short next_dist = _current_j - _current_i;

        if (next_j == _word_size) {
            next_i = 0;
            next_dist = next_dist + 1;
            next_j = next_dist;
        }

        if (next_dist > 0 && next_j < _word_size && next_i >= 0) {
            return true;
        }
        return false;
    }


    bool IsTerminated() const {
        return _bEnd;
    }

  public:
    // please check the specified index has been annotated with ner label first
    short getNERId(const int& i) const {
        if (i >= _word_size || i < 0) {
            return -1;
        }

        return _labels[i][i];
    }

    // please check the specified index has been annotated with ner label first
    short getSpanStart(const int& i) const {
        if (i >= _word_size || i < 0) {
            return -1;
        }

        if (i == 0 || _labels[i][i] % 4 == 0) {
            return i;
        }

        int j = i - 1;
        while (j > 0) {
            if (_labels[j][j] % 4 == 1) {
                break;
            }
            j--;
        }

        return j;
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
        next->_current_j = _current_j + 1;
        next->_current_i = _current_i + 1;
        if (next->_current_j == _word_size) {
            next->_current_i = 0;
            next->_current_j = _current_j - _current_i + 1;
        }

        next->_labels[next->_current_i][next->_current_j] = ner_id;

        next->_lastAction.set(CAction::NER, ner_id); //TODO:
    }

    void rel(CStateItem *next, short rel_id) {
        if (!allow_rel()) {
            std::cout << "assign relation error" << std::endl;
            return;
        }

        copyProperty2Next(next);
        next->_label = rel_id;
        next->_current_j = _current_j + 1;
        next->_current_i = _current_i + 1;

        if (next->_current_j == _word_size) {
            next->_current_i = 0;
            next->_current_j = _current_j - _current_i + 1;
        }

        next->_labels[next->_current_i][next->_current_j] = rel_id;

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
        } else if (ac.isREL()) {
            rel(next, ac._label); //TODO:
        } else {
            std::cout << "error action" << std::endl;
        }

        if (next->_current_i == 0 && next->_current_j == _word_size - 1) {
            next->_bEnd = true;
        }
    }

    //partial results
    void getResults(CResult &result, HyperParams &opts) const {
        result.clear();
        const CStateItem* curr = this;
        result.words = &(_inst->words);
        result.tags = &(_inst->tags);
        result.heads = &(_inst->heads);
        result.labels = &(_inst->labels);
        result.allocate(_word_size);
        int count = 0;

        while (true) {
            if (curr->_current_i == curr->_current_j) {
                result.ners[curr->_current_i] = opts.ner_labels.from_id(curr->_label);
            } else {
                short labelId = curr->_label;
                int dist = curr->_current_j - curr->_current_i;
                if (labelId > 0) {
                    result.relations[curr->_current_i][dist - 1] = opts.rel_labels.from_id(labelId);
                    result.directions[curr->_current_i][dist - 1] = 1;
                } else if (labelId < 0) {
                    result.relations[curr->_current_i][dist - 1] = opts.rel_labels.from_id(-labelId);
                    result.directions[curr->_current_i][dist - 1] = -1;
                } else {
                    result.relations[curr->_current_i][dist - 1] = opts.rel_labels.from_id(0);
                    result.directions[curr->_current_i][dist - 1] = 0;
                }
            }
            count++;
            curr = curr->_prevState;
            if (curr->_bStart) {
                break;
            }
        }

        if (count != _step) {
            std::cout << "step number not equal count of historical states, please check." << std::endl;
        }
    }

    // TODO:
    void getGoldAction(HyperParams &opts, const CResult &result, CAction &ac) const {
        short next_j = _current_j + 1;
        short next_i = _current_i + 1;
        short next_dist = _current_j - _current_i;

        if (next_j == _word_size) {
            next_i = 0;
            next_dist = next_dist + 1;
            next_j = next_dist;
        }

        if (allow_ner()) {
            ac.set(CAction::NER, opts.ner_labels.from_string(result.ners[next_i]));
            return;
        }

        if (allow_rel()) {
            int rel_labelId = opts.rel_labels.from_string(result.relations[next_i][next_dist - 1]);
            if (result.directions[next_i][next_dist - 1] < 0) {
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
            } else if (modvalue == 1) { //b-xx
                ac.set(CAction::NER, _lastAction._label + 1);  //m-xx
                if (params->embeded_actions.from_string(ac.str(opts)) >= 0)actions.push_back(ac);
                ac.set(CAction::NER, _lastAction._label + 2);  //e-xx
                if (params->embeded_actions.from_string(ac.str(opts)) >= 0)actions.push_back(ac);
            } else { // m-xx
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
            short next_j = _current_j + 1;
            short next_i = _current_i + 1;
            short next_dist = _current_j - _current_i;

            if (next_j == _word_size) {
                next_i = 0;
                next_dist = next_dist + 1;
                next_j = next_dist;
            }

            short label_i = getNERId(next_i);
            short label_j = getNERId(next_j);

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
            } else {
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
        } else {
            _nextscores.forward(cg, acs, _atomFeat, _score);
        }
    }

    inline void prepare(HyperParams* hyper_params, ModelParams* model_params, GlobalNodes* global_nodes) {
        _atomFeat.clear();
        short next_j = _current_j + 1;
        short next_i = _current_i + 1;
        short next_dist = _current_j - _current_i;

        if (next_j == _word_size) {
            next_i = 0;
            next_dist = next_dist + 1;
            next_j = next_dist;
        }

        if (allow_ner()) {
            _atomFeat.ner_next_position = next_j;
            _atomFeat.ner_last_end = _current_j;
            if (_current_j + 1 != next_j) {
                std::cout << "musth have an error here" << std::endl;
            }
            _atomFeat.ner_last_start = getSpanStart(_atomFeat.ner_last_end);
            short label_j = getNERId(_current_j);
            _atomFeat.ner_last_label = hyper_params->ner_labels.from_id(label_j);
            _atomFeat.bRel = false;
        } else if (allow_rel()) {
            _atomFeat.rel_i = next_i;
            _atomFeat.rel_j = next_j;

            _atomFeat.rel_i_start = getSpanStart(_atomFeat.rel_i);
            _atomFeat.rel_j_start = getSpanStart(_atomFeat.rel_j);

            short label_i = getNERId(_atomFeat.rel_i);
            short label_j = getNERId(_atomFeat.rel_j);

            _atomFeat.rel_j_nerlabel = hyper_params->ner_labels.from_id(label_j);

            int modvalue_i = label_i % 4;
            int modvalue_j = label_j % 4;

            if (label_i > 0 && label_j > 0
                    && (modvalue_i == 0 || modvalue_i == 3)
                    && (modvalue_j == 0 || modvalue_j == 3)) {
                _atomFeat.rel_must_o = 0;
            } else {
                _atomFeat.rel_must_o = 1;
            }
            _atomFeat.bRel = true;
        } else {
            std::cout << "error for next step!" << std::endl;
        }

        _atomFeat.word_size = _word_size;
        _atomFeat.p_word_left_lstm = global_nodes == NULL ? NULL : &(global_nodes->word_left_lstm);
        _atomFeat.p_word_right_lstm = global_nodes == NULL ? NULL : &(global_nodes->word_right_lstm);
        _atomFeat.p_ner_lstms.resize(_word_size);
        if (_bStart) {
            for (int idx = 0; idx < _atomFeat.word_size; idx++) {
                _atomFeat.p_ner_lstms[idx] = NULL;
            }
        } else {
            for (int idx = 0; idx < _atomFeat.word_size; idx++) {
                _atomFeat.p_ner_lstms[idx] = _prevState->_nextscores.p_ner_lstms[idx];
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