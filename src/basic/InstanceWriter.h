#ifndef _CONLL_WRITER_
#define _CONLL_WRITER_

#include "Writer.h"
#include "MyLib.h"
#include <sstream>

class InstanceWriter : public Writer {
  public:
    InstanceWriter() {}

    ~InstanceWriter() {}

    int write(const Instance *pInstance) {
        if (!m_outf.is_open()) return -1;
        m_outf << pInstance->result.str() << endl;
        return 0;
    }

    //FIXME:
    int write(const CResult &result) {
        if (!m_outf.is_open())
            return -1;
        m_outf << result.str() << endl;
        return 0;
    }
};

#endif

