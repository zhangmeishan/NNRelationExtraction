#include "RelationClassifier.h"
#include <chrono>
#include <omp.h>
#include <set>

#include "Argument_helper.h"

Extractor::Extractor(size_t memsize) : m_driver(memsize) {
    srand(0);
}


Extractor::~Extractor() {
}

int Extractor::createAlphabet(vector<Instance> &vecInsts) {
    cout << "Creating Alphabet..." << endl;

    int totalInstance = vecInsts.size();

    unordered_map<string, int> word_stat;
    unordered_map<string, int> tag_stat;
    unordered_map<string, int> dep_stat;
    unordered_map<string, int> ner_stat;
    unordered_map<string, int> rel_stat;
    unordered_map<string, int> char_stat;

    string root = "ROOT";


    assert(totalInstance > 0);
    for (int numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
        const Instance &instance = vecInsts[numInstance];

        for (int idx = 0; idx < instance.words.size(); idx++) {
            string curWord = normalize_to_lower(instance.words[idx]);
            m_driver._hyperparams.word_stat[curWord]++;
            word_stat[curWord]++;
            tag_stat[instance.tags[idx]]++;
            dep_stat[instance.labels[idx]]++;
            if (instance.heads[idx] == -1)
                root = instance.labels[idx];

            string curner = instance.result.ners[idx];
            if (is_start_label(curner)) {
                ner_stat[cleanLabel(curner)]++;
            }

            for (int idy = 0; idy < instance.chars[idx].size(); idy++) {
                char_stat[instance.chars[idx][idy]]++;
            }
        }

        for (int idx = 0; idx < instance.result.relations.size(); idx++) {
            for (int idy = 0; idy < instance.result.relations[idx].size(); idy++) {
                if (instance.result.relations[idx][idy] != "noRel") {
                    rel_stat[instance.result.relations[idx][idy]]++;
                    m_driver._hyperparams.rel_dir[instance.result.relations[idx][idy]].insert(instance.result.directions[idx][idy]);
                }
            }
        }
    }

    if (m_options.wordEmbFile != "") {
        m_driver._modelparams.embeded_ext_words.initial(m_options.wordEmbFile);
    } else {
        std::cerr << "missing embedding file! \n";
        exit(0);
    }

    word_stat[unknownkey] = m_options.wordCutOff + 1;
    m_driver._modelparams.embeded_words.initial(word_stat, m_options.wordCutOff);
    m_driver._modelparams.embeded_tags.initial(tag_stat, 0);
    //m_driver._modelparams.embeded_labels.initial(dep_stat, 0);
    char_stat[unknownkey] = 1;
    m_driver._modelparams.embeded_chars.initial(char_stat, 0);


    // TODO:

    m_driver._hyperparams.ner_labels.clear();
    m_driver._hyperparams.ner_labels.from_string("o");
    static unordered_map<string, int>::const_iterator iter;
    for (iter = ner_stat.begin(); iter != ner_stat.end(); iter++) {
        m_driver._hyperparams.ner_labels.from_string("b-" + iter->first);
        m_driver._hyperparams.ner_labels.from_string("m-" + iter->first);
        m_driver._hyperparams.ner_labels.from_string("e-" + iter->first);
        m_driver._hyperparams.ner_labels.from_string("s-" + iter->first);
    }
    m_driver._hyperparams.ner_labels.set_fixed_flag(true);
    int ner_count = m_driver._hyperparams.ner_labels.size();

    m_driver._hyperparams.rel_labels.clear();
    m_driver._hyperparams.rel_labels.from_string("noRel");
    m_driver._hyperparams.rel_dir["noRel"].insert(1);
    for (iter = rel_stat.begin(); iter != rel_stat.end(); iter++) {
        m_driver._hyperparams.rel_labels.from_string(iter->first);
    }
    m_driver._hyperparams.rel_labels.set_fixed_flag(true);
    int rel_count = m_driver._hyperparams.rel_labels.size();


    m_driver._hyperparams.action_num = ner_count > 2 * rel_count ? ner_count : 2 * rel_count;

    unordered_map<string, int> action_stat;
    vector<CStateItem> state(m_driver._hyperparams.maxlength + 1);
    CResult output;
    CAction answer;
    Metric ner, rel, rel_punc;
    ner.reset();
    rel.reset();
    rel_punc.reset();

    int stepNum;

    for (int numInstance = 0; numInstance < totalInstance; numInstance++) {
        Instance &instance = vecInsts[numInstance];
        stepNum = 0;
        state[stepNum].clear();
        state[stepNum].setInput(instance);
        while (!state[stepNum].IsTerminated()) {
            state[stepNum].getGoldAction(m_driver._hyperparams, instance.result, answer);
            //       std::cout << answer.str(&(m_driver._hyperparams)) << " ";

            action_stat[answer.str(&(m_driver._hyperparams))]++;
            //      TODO: state? answer(gold action)?
            state[stepNum].prepare(&m_driver._hyperparams, NULL, NULL);
            state[stepNum].move(&(state[stepNum + 1]), answer);
            stepNum++;
        }
        //
        state[stepNum].getResults(output, m_driver._hyperparams);
        //    std::cout << endl;
        //    std::cout << output.str();
        ////
        instance.evaluate(output, ner, rel); //TODO: 不唯一? //FIXME:

        if (!ner.bIdentical() || !rel.bIdentical()) {
            std::cout << "error state conversion!" << std::endl;
            exit(0);
        }


        if ((numInstance + 1) % m_options.verboseIter == 0) {
            std::cout << numInstance + 1 << " ";
            if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
                std::cout << std::endl;
            std::cout.flush();
        }
        if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
            break;
    }
    action_stat[nullkey] = 1;
    m_driver._modelparams.embeded_actions.initial(action_stat, 0);

    return 0;
}

void Extractor::getGoldActions(vector<Instance>& vecInsts, vector<vector<CAction> >& vecActions) {
    vecActions.clear();

    static vector<CAction> acs;
    static bool bFindGold;
    Metric ner, rel, rel_punc;
    vector<CStateItem> state(m_driver._hyperparams.maxlength + 1);
    CResult output;
    CAction answer;
    ner.reset();
    rel.reset();
    rel_punc.reset();
    static int numInstance, stepNum;
    vecActions.resize(vecInsts.size());
    for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
        Instance &instance = vecInsts[numInstance];

        stepNum = 0;
        state[stepNum].clear();
        state[stepNum].setInput(instance);
        while (!state[stepNum].IsTerminated()) {
            state[stepNum].getGoldAction(m_driver._hyperparams, instance.result, answer);
            // std::cout << answer.str(&(m_driver._hyperparams)) << " ";
            state[stepNum].getCandidateActions(acs, &m_driver._hyperparams, &m_driver._modelparams);

            bFindGold = false;
            for (int idz = 0; idz < acs.size(); idz++) {
                if (acs[idz] == answer) {
                    bFindGold = true;
                    break;
                }
            }
            if (!bFindGold) {
                state[stepNum].getCandidateActions(acs, &m_driver._hyperparams, &m_driver._modelparams);
                std::cout << "gold action has been filtered" << std::endl;
                exit(0);
            }

            vecActions[numInstance].push_back(answer);
            state[stepNum].move(&state[stepNum + 1], answer);
            stepNum++;
        }

        state[stepNum].getResults(output, m_driver._hyperparams);
        //FIXME: 内存错误
        //    std::cout << endl;
        //    std::cout << output.str();
        instance.evaluate(output, ner, rel);

        if (!ner.bIdentical() || !rel.bIdentical()) {
            std::cout << "error state conversion!" << std::endl;
            exit(0);
        }

        if ((numInstance + 1) % m_options.verboseIter == 0) {
            cout << numInstance + 1 << " ";
            if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
                cout << std::endl;
            cout.flush();
        }
        if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
            break;
    }
}

void Extractor::train(const string &trainFile, const string &devFile, const string &testFile, const string &modelFile,
                      const string &optionFile) {
    if (optionFile != "")
        m_options.load(optionFile);

    m_options.showOptions();
    vector<Instance> trainInsts, devInsts, testInsts;
    m_pipe.readInstances(trainFile, trainInsts, m_options.maxInstance);
    if (devFile != "")
        m_pipe.readInstances(devFile, devInsts, m_options.maxInstance);
    if (testFile != "")
        m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);

    vector<vector<Instance> > otherInsts(m_options.testFiles.size());
    for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
        m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_options.maxInstance);
    }

    createAlphabet(trainInsts);

    m_driver._modelparams.word_ext_table.initial(&m_driver._modelparams.embeded_ext_words, m_options.wordEmbFile, false, m_options.wordEmbNormalize);
    m_options.wordExtEmbSize = m_driver._modelparams.word_ext_table.nDim;

    m_driver._modelparams.word_table.initial(&m_driver._modelparams.embeded_words, m_options.wordEmbSize, true);

    m_driver._modelparams.tag_table.initial(&m_driver._modelparams.embeded_tags, m_options.tagEmbSize, true);

    m_driver._modelparams.action_table.initial(&m_driver._modelparams.embeded_actions, m_options.actionEmbSize, true);

    //m_driver._modelparams.label_table.initial(&m_driver._modelparams.embeded_labels, m_options.labelEmbSize, true);
    m_driver._modelparams.char_table.initial(&m_driver._modelparams.embeded_chars, m_options.charEmbSize, true);

    m_driver._hyperparams.setRequared(m_options);
    m_driver.initial();

    vector<vector<CAction> > trainInstGoldactions, devInstGoldactions, testInstGoldactions;
    getGoldActions(trainInsts, trainInstGoldactions);
    //getGoldActions(devInsts, devInstGoldactions);
    //getGoldActions(testInsts, testInstGoldactions);
    double bestFmeasure = -1;

    int inputSize = trainInsts.size();

    std::vector<int> indexes;
    for (int i = 0; i < inputSize; ++i)
        indexes.push_back(i);

    static Metric eval;
    static Metric dev_ner, dev_rel;
    static Metric test_ner, test_rel;

    int maxIter = m_options.maxIter;
    int oneIterMaxRound = (inputSize + m_options.batchSize - 1) / m_options.batchSize;
    std::cout << "\nmaxIter = " << maxIter << std::endl;
    int devNum = devInsts.size(), testNum = testInsts.size();

    vector<CResult > decodeInstResults;
    bool bCurIterBetter;
    vector<Instance> subInstances;
    vector<vector<CAction> > subInstGoldActions;
    NRVec<bool> decays;
    decays.resize(maxIter);
    decays = false;
    //decays[5] = true; decays[15] = true; decays[25] = true;
    for (int iter = 0; iter < maxIter; ++iter) {
        std::cout << "##### Iteration " << iter << std::endl;
        srand(iter);
        bool bEvaluate = false;

        if (m_options.batchSize == 1) {
            auto t_start_train = std::chrono::high_resolution_clock::now();
            eval.reset();
            bEvaluate = true;
            random_shuffle(indexes.begin(), indexes.end());
            std::cout << "random: " << indexes[0] << ", " << indexes[indexes.size() - 1] << std::endl;
            for (int idy = 0; idy < inputSize; idy++) {
                subInstances.clear();
                subInstGoldActions.clear();
                subInstances.push_back(trainInsts[indexes[idy]]);
                subInstGoldActions.push_back(trainInstGoldactions[indexes[idy]]);
                double cost = m_driver.train(subInstances, subInstGoldActions);

                eval.overall_label_count += m_driver._eval.overall_label_count;
                eval.correct_label_count += m_driver._eval.correct_label_count;

                if ((idy + 1) % (m_options.verboseIter * 10) == 0) {
                    auto t_end_train = std::chrono::high_resolution_clock::now();
                    std::cout << "current: " << idy + 1 << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy()
                              << ", time = " << std::chrono::duration<double>(t_end_train - t_start_train).count() << std::endl;
                }

                //if (m_driver._batch >= m_options.batchSize) {
                //    m_driver.updateModel();
                //}
                m_driver.updateModel();
            }
            {
                auto t_end_train = std::chrono::high_resolution_clock::now();
                std::cout << "current: " << iter + 1 << ", Correct(%) = " << eval.getAccuracy()
                          << ", time = " << std::chrono::duration<double>(t_end_train - t_start_train).count() << std::endl;
            }
        } else {
            eval.reset();
            auto t_start_train = std::chrono::high_resolution_clock::now();
            bEvaluate = true;
            for (int idk = 0; idk < (inputSize + m_options.batchSize - 1) / m_options.batchSize; idk++) {
                random_shuffle(indexes.begin(), indexes.end());
                subInstances.clear();
                subInstGoldActions.clear();
                for (int idy = 0; idy < m_options.batchSize; idy++) {
                    subInstances.push_back(trainInsts[indexes[idy]]);
                    subInstGoldActions.push_back(trainInstGoldactions[indexes[idy]]);
                }
                double cost = m_driver.train(subInstances, subInstGoldActions);

                eval.overall_label_count += m_driver._eval.overall_label_count;
                eval.correct_label_count += m_driver._eval.correct_label_count;

                if ((idk + 1) % (m_options.verboseIter * 10) == 0) {
                    auto t_end_train = std::chrono::high_resolution_clock::now();
                    std::cout << "current: " << idk + 1 << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy()
                              << ", time = " << std::chrono::duration<double>(t_end_train - t_start_train).count() << std::endl;
                }

                m_driver.updateModel();
            }

            {
                auto t_end_train = std::chrono::high_resolution_clock::now();
                std::cout << "current: " << iter + 1 << ", Correct(%) = " << eval.getAccuracy()
                          << ", time = " << std::chrono::duration<double>(t_end_train - t_start_train).count() << std::endl;
            }
        }

        if (decays[iter]) {
            m_options.adaAlpha = 0.5 * m_options.adaAlpha;
            m_driver.setUpdateParameters(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);
        }

        if (bEvaluate && devNum > 0) {
            auto t_start_dev = std::chrono::high_resolution_clock::now();
            std::cout << "Dev start." << std::endl;
            bCurIterBetter = false;
            if (!m_options.outBest.empty())
                decodeInstResults.clear();
            dev_ner.reset();
            dev_rel.reset();
            predict(devInsts, decodeInstResults);
            for (int idx = 0; idx < devInsts.size(); idx++) {
                devInsts[idx].evaluate(decodeInstResults[idx], dev_ner, dev_rel);
            }
            auto t_end_dev = std::chrono::high_resolution_clock::now();
            std::cout << "Dev finished. Total time taken is: " << std::chrono::duration<double>(t_end_dev - t_start_dev).count() << std::endl;
            std::cout << "dev:" << std::endl;
            dev_ner.print();
            dev_rel.print();

            if (!m_options.outBest.empty() && dev_rel.getAccuracy() > bestFmeasure) {
                m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
                bCurIterBetter = true;
            }
        }

        if (testNum > 0) {
            clock_t time_start = clock();
            auto t_start_test = std::chrono::high_resolution_clock::now();
            if (!m_options.outBest.empty())
                decodeInstResults.clear();
            test_ner.reset();
            test_rel.reset();
            predict(testInsts, decodeInstResults);
            for (int idx = 0; idx < testInsts.size(); idx++) {
                testInsts[idx].evaluate(decodeInstResults[idx], test_ner, test_rel);
            }
            auto t_end_test = std::chrono::high_resolution_clock::now();
            std::cout << "Test finished. Total time taken is: " << std::chrono::duration<double>(t_end_test - t_start_test).count() << std::endl;
            std::cout << "test:" << std::endl;
            test_ner.print();
            test_rel.print();

            if (!m_options.outBest.empty() && bCurIterBetter) {
                m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
            }
        }

        for (int idx = 0; idx < otherInsts.size(); idx++) {
            auto t_start_other = std::chrono::high_resolution_clock::now();
            std::cout << "processing " << m_options.testFiles[idx] << std::endl;
            if (!m_options.outBest.empty())
                decodeInstResults.clear();
            test_ner.reset();
            test_rel.reset();
            predict(otherInsts[idx], decodeInstResults);
            for (int idy = 0; idy < otherInsts[idx].size(); idy++) {
                otherInsts[idx][idy].evaluate(decodeInstResults[idy], test_ner, test_rel);
            }
            auto t_end_other = std::chrono::high_resolution_clock::now();
            std::cout << "Test finished. Total time taken is: " << std::chrono::duration<double>(t_end_other - t_start_other).count() << std::endl;
            std::cout << "test:" << std::endl;
            test_ner.print();
            test_rel.print();

            if (!m_options.outBest.empty() && bCurIterBetter) {
                m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
            }
        }


        if (m_options.saveIntermediate && dev_rel.getAccuracy() > bestFmeasure) {
            std::cout << "Exceeds best previous DIS of " << bestFmeasure << ". Saving model file.." << std::endl;
            bestFmeasure = dev_rel.getAccuracy();
            writeModelFile(modelFile);
        }

    }
}

void Extractor::predict(vector<Instance>& inputs, vector<CResult> &outputs) {
    int sentNum = inputs.size();
    if (sentNum <= 0) return;
    vector<Instance> batch_sentences;
    vector<CResult> batch_outputs;
    outputs.resize(sentNum);
    int sent_count = 0;
    for (int idx = 0; idx < sentNum; idx++) {
        batch_sentences.push_back(inputs[idx]);
        if (batch_sentences.size() == m_options.batchSize || idx == sentNum - 1) {
            m_driver.decode(batch_sentences, batch_outputs);
            batch_sentences.clear();
            for (int idy = 0; idy < batch_outputs.size(); idy++) {
                outputs[sent_count].copyValuesFrom(batch_outputs[idy]);
                outputs[sent_count].words = &(inputs[sent_count].words);
                outputs[sent_count].tags = &(inputs[sent_count].tags);
                outputs[sent_count].heads = &(inputs[sent_count].heads);
                outputs[sent_count].labels = &(inputs[sent_count].labels);
                sent_count++;
            }
        }
    }

    if (outputs.size() != sentNum) {
        std::cout << "decoded number not match" << std::endl;
    }

}

void Extractor::test(const string &testFile, const string &outputFile, const string &modelFile) {

}


void Extractor::loadModelFile(const string &inputModelFile) {

}

void Extractor::writeModelFile(const string &outputModelFile) {

}

int main(int argc, char* argv[]) {
    std::string trainFile = "", devFile = "", testFile = "", modelFile = "";
    std::string wordEmbFile = "", optionFile = "";
    std::string outputFile = "";
    bool bTrain = false;
    dsr::Argument_helper ah;
    int memsize = 0;
    int threads = 1;


    ah.new_flag("l", "learn", "train or test", bTrain);
    ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
    ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
    ah.new_named_string("test", "testCorpus", "named_string",
                        "testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
    ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
    ah.new_named_string("word", "wordEmbFile", "named_string", "pretrained word embedding file to train a model, optional when training", wordEmbFile);
    ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
    ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);
    ah.new_named_int("mem", "memsize", "named_int", "memory allocated for tensor nodes", memsize);
    ah.new_named_int("th", "thread", "named_int", "number of threads for openmp", threads);

    ah.process(argc, argv);

    omp_set_num_threads(threads);
    //  Eigen::setNbThreads(threads);
    //  mkl_set_num_threads(4);
    //  mkl_set_dynamic(false);
    //  omp_set_nested(false);
    //  omp_set_dynamic(false);
    Extractor extractor(memsize);
    if (bTrain) {
        extractor.train(trainFile, devFile, testFile, modelFile, optionFile);
    } else {
        extractor.test(testFile, outputFile, modelFile);
    }

    //test(argv);
    //ah.write_values(std::cout);

}
