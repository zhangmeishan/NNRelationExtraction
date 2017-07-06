#ifndef _JST_INSTANCE_
#define _JST_INSTANCE_

#include "N3LDG.h"
#include "Metric.h"
#include "Result.h"

class Instance {
public:
  Instance() {
  }

  Instance(const Instance &other) {
    copyValuesFrom(other);
  }

  ~Instance() {
  }

public:

  int size() const {
    return words.size();
  }


  void clear() {
    words.clear();
    tags.clear();
    heads.clear();
    labels.clear();
    result.clear();
    clearVec(chars);
    clearVec(syn_feats);
  }

  void allocate(const int &size) {
    if (words.size() != size) {
      words.resize(size);
    }
    if (tags.size() != size) {
      tags.resize(size);
    }
    if (heads.size() != size) {
      heads.resize(size);
    }
    if (labels.size() != size) {
      labels.resize(size);
    }

    syn_feats.resize(size);
    for (int idx = 0; idx < size; idx++) {
      syn_feats[idx].clear();
    }

    result.allocate(size);
  }


  void copyValuesFrom(const Instance &anInstance) {
    allocate(anInstance.size());
    for (int i = 0; i < anInstance.size(); i++) {
      words[i] = anInstance.words[i];
    }

    for (int i = 0; i < anInstance.size(); i++) {
      tags[i] = anInstance.tags[i];
    }

    for (int i = 0; i < anInstance.size(); i++) {
      heads[i] = anInstance.heads[i];
    }

    for (int i = 0; i < anInstance.size(); i++) {
      labels[i] = anInstance.labels[i];
    }

    result.copyValuesFrom(anInstance.result, &words, &tags, &heads, &labels);

    getChars();

    for (int i = 0; i < anInstance.size(); i++) {
      syn_feats[i].clear();
      for (int j = 0; j < anInstance.syn_feats[i].size(); j++) {
        syn_feats[i].push_back(anInstance.syn_feats[i][j]);
      }
    }

  }


  void evaluate(CResult &other, Metric &nerEval, Metric &relEval) {
    unordered_set<string>::iterator iter;

    unordered_set<string> gold_entities, pred_entities;
    result.extractNERs(gold_entities);
    other.extractNERs(pred_entities);

    nerEval.overall_label_count += gold_entities.size();
    nerEval.predicated_label_count += pred_entities.size();
    for (iter = pred_entities.begin(); iter != pred_entities.end(); iter++) {
      if (gold_entities.find(*iter) != gold_entities.end()) {
        nerEval.correct_label_count++;
      }
    }

    unordered_set<string> gold_entity_relations, pred_entity_relations;
    result.extractRelations(gold_entity_relations);
    other.extractRelations(pred_entity_relations);

    relEval.overall_label_count += gold_entity_relations.size();
    relEval.predicated_label_count += pred_entity_relations.size();
    for (iter = pred_entity_relations.begin(); iter != pred_entity_relations.end(); iter++) {
      if (gold_entity_relations.find(*iter) != gold_entity_relations.end()) {
        relEval.correct_label_count++;
      }
    }
  }


public:
  vector<string> words;
  vector<vector<string> > chars;
  vector<string> tags;
  vector<string> labels;
  vector<int> heads;
  CResult result;

  //extenral features
  vector<vector<PNode> > syn_feats;

public:
  inline void getChars() {
    int size = words.size();
    chars.resize(size);
    for (int idx = 0; idx < size; idx++) {
      getCharactersFromUTF8String(words[idx], chars[idx]);
    }

  }

};

#endif
