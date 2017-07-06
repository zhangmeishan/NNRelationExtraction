#ifndef BASIC_CAction_H_
#define BASIC_CAction_H_

#include <sstream>
const static short invalid_label = -1000;
class CAction {
public:
  enum CODE {
    NER = 0,
    REL = 1,
    NO_ACTION = 2
  };

  unsigned long _code;
  short _label;

public:
  CAction() : _code(NO_ACTION), _label(invalid_label) {
  }

  CAction(int code, short label) : _code(code), _label(label) {
  }

  CAction(const CAction &ac) : _code(ac._code), _label(ac._label) {
  }

public:
  inline void clear() {
    _code = NO_ACTION;
    _label = invalid_label;
  }

  inline void set(int code, short label) {
    _code = code;
    _label = label;
  }

  inline void set(const CAction &ac) {
    _code = ac._code;
    _label = ac._label;
  }

  inline bool isNone() const { return _code == NO_ACTION; }

  inline bool isNER() const { return _code == NER; }

  inline bool isREL() const { return _code == REL; }

public:
//  TODO: add label?
  inline std::string str(HyperParams *opts) const {
    if (_code == NER) return "NER_" + opts->ner_labels.from_id(_label);
    else if (_code == REL) {
        if (_label >= 0) {
            return "REL_" + opts->rel_labels.from_id(_label);
        }
        else {
            return "RREL_" + opts->rel_labels.from_id(-_label);
        }
    }

    return "NO_ACTION";
  }

public:
  bool operator==(const CAction &a1) const { return (_code == a1._code) && (_label == a1._label); }

  bool operator!=(const CAction &a1) const { return (_code != a1._code) || (_label != a1._label); }
};


#endif /* BASIC_CAction_H_ */
