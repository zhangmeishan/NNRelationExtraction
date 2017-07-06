#ifndef BASIC_CRESULT_H_
#define BASIC_CRESULT_H_

#include <string>
#include <vector>
#include <fstream>
#include "N3LDG.h"
#include "Alphabet.h"
#include "Utf.h"


class CResult {
public:
  vector<string> ners;
  //relation[i][j]; i, i+j+1 has relation
  vector<vector<string> > relations;
  vector<vector<int> > directions;


  const vector<string> *words;
  const vector<string> *tags;
  const vector<int> *heads;
  const vector<string> *labels;


public:
  inline void clear() {
    words = NULL;
    tags = NULL;
    heads = NULL;
    labels = NULL;

    ners.clear();
    clearVec(relations);
    clearVec(directions);
  }

  inline void allocate(const int &size) {
    if (ners.size() != size) {
      ners.resize(size);
      relations.resize(size - 1);
      directions.resize(size - 1);
      for (int idx = 0; idx < size; idx++) {
        ners[idx] = "o";
        if (size - idx - 1 > 0) {
          relations[idx].resize(size - idx - 1);
          directions[idx].resize(size - idx - 1);
          for (int idy = 0; idy < size - idx - 1; idy++) {
            relations[idx][idy] = "noRel";
            directions[idx][idy] = 0;
          }
        }
      }
    }
  }

  inline void extractNERs(unordered_set<string>& entities) {
    static int idx, idy, endpos;
    entities.clear();
    idx = 0;
    while (idx < ners.size()) {
      if (is_start_label(ners[idx])) {
        idy = idx;
        endpos = -1;
        while (idy < ners.size()) {
          if (!is_continue_label(ners[idy], ners[idx], idy - idx)) {
            endpos = idy - 1;
            break;
          }
          endpos = idy;
          idy++;
        }
        stringstream ss;
        ss << "[" << idx << "," << endpos << "]";
        entities.insert(cleanLabel(ners[idx]) + ss.str());
        idx = endpos;
      }
      idx++;
    }
  }

  inline void extractRelations(unordered_set<string>& entity_relations) {
    static int idx, idy, endpos;
    unordered_map<int, string> marked_entities;
    idx = 0;
    int size = ners.size();
    while (idx < size) {
      if (is_start_label(ners[idx])) {
        idy = idx;
        endpos = -1;
        while (idy < ners.size()) {
          if (!is_continue_label(ners[idy], ners[idx], idy - idx)) {
            endpos = idy - 1;
            break;
          }
          endpos = idy;
          idy++;
        }
        stringstream ss;
        ss << "[" << idx << "," << endpos << "]";
        marked_entities[endpos] = ss.str();
        idx = endpos;
      }
      idx++;
    }
    entity_relations.clear();
    for (int i = 0; i < size - 1; ++i) {
      for (int j = 0; j < size - i - 1; j++) {
        if (relations[i][j] != "noRel") {
          stringstream ss;
          ss << "(" << marked_entities[i] << "," << marked_entities[i + j + 1] << ")=[" << directions[i][j] << "]" << relations[i][j] << std::endl;
          entity_relations.insert(ss.str());
        }
      }
    }
  }

  inline int size() const {
    return ners.size();
  }

  inline void copyValuesFrom(const CResult &result) {
    static int size;
    size = result.size();
    allocate(size);

    for (int i = 0; i < size; i++) {
      ners[i] = result.ners[i];
      if (i == size - 1) continue;
      for (int j = 0; j < size - i - 1; j++) {
        relations[i][j] = result.relations[i][j];
        directions[i][j] = result.directions[i][j];
      }
    }

    words = result.words;
    tags = result.tags;
    heads = result.heads;
    labels = result.labels;
  }

  inline void copyValuesFrom(const CResult &result, const vector<string> *pwords,
    const vector<string> *ptags, const vector<int> *pheads, const vector<string> *plabels) {
    static int size;
    size = result.size();
    allocate(size);

    for (int i = 0; i < size; i++) {
      ners[i] = result.ners[i];
      if (i == size - 1) continue;
      for (int j = 0; j < size - i - 1; j++) {
        relations[i][j] = result.relations[i][j];
        directions[i][j] = result.directions[i][j];
      }
    }
    words = pwords;
    tags = ptags;
    heads = pheads;
    labels = plabels;
  }


  inline std::string str() const {
    stringstream ss;
    int size = ners.size();
    for (int i = 0; i < size; ++i) {
      ss << "token " << (*words)[i] << " " << (*tags)[i] << " " << (*heads)[i] << " " << (*labels)[i] << " " << ners[i] << std::endl;
    }
    for (int i = 0; i < size - 1; ++i) {
      for (int j = 0; j < size - i - 1; j++) {
        if (relations[i][j] != "noRel") {
          ss << "rel " << i << " " << i + j + 1 << " " << directions[i][j] << " " << relations[i][j] << std::endl;
        }
      }
    }

    return ss.str();
  }

};


#endif
