#include "CombinedExtractor.h"
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
    }
    else {
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
    ner_stat[nullkey] = 1;
    m_driver._modelparams.embeded_ners.initial(ner_stat, 0);


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
    ner.reset(); rel.reset(); rel_punc.reset();
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

    m_driver._modelparams.ner_table.initial(&m_driver._modelparams.embeded_ners, m_options.nerEmbSize, true);

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

    static vector<CResult > decodeInstResults;
    static CResult curDecodeInst;
    static bool bCurIterBetter;
    static vector<Instance> subInstances;
    static vector<vector<CAction> > subInstGoldActions;
    NRVec<bool> decays;
    decays.resize(maxIter);
    decays = false;
    decays[5] = true; decays[15] = true; decays[30] = true;
    int maxNERIter = 0;
    int startBeam = 30;
    for (int iter = 0; iter < maxIter; ++iter) {
        std::cout << "##### Iteration " << iter << std::endl;
        srand(iter);
        random_shuffle(indexes.begin(), indexes.end());
        std::cout << "random: " << indexes[0] << ", " << indexes[indexes.size() - 1] << std::endl;
        bool bEvaluate = false;

        eval.reset();
        bEvaluate = true;
        for (int idy = 0; idy < inputSize; idy++) {
            subInstances.clear();
            subInstGoldActions.clear();
            subInstances.push_back(trainInsts[indexes[idy]]);
            subInstGoldActions.push_back(trainInstGoldactions[indexes[idy]]);
            double cost = m_driver.train(subInstances, subInstGoldActions, iter < maxNERIter);

            eval.overall_label_count += m_driver._eval.overall_label_count;
            eval.correct_label_count += m_driver._eval.correct_label_count;

            if ((idy + 1) % (m_options.verboseIter) == 0) {
                std::cout << "current: " << idy + 1 << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy() << std::endl;
            }
            if (m_driver._batch >= m_options.batchSize) {
                m_driver.updateModel();
            }
        }
        if (m_driver._batch > 0) {
            m_driver.updateModel();
        }

        if (decays[iter]) {
            m_options.adaAlpha = 0.5 * m_options.adaAlpha;
            m_driver.setUpdateParameters(m_options.regParameter, m_options.adaAlpha, m_options.adaEps);
        }

        std::cout << "current: " << iter + 1 << ", Correct(%) = " << eval.getAccuracy() << std::endl;

        if (bEvaluate && devNum > 0) {
            clock_t time_start = clock();
            std::cout << "Dev start." << std::endl;
            bCurIterBetter = false;
            if (!m_options.outBest.empty())
                decodeInstResults.clear();
            dev_ner.reset();
            dev_rel.reset();
            for (int idx = 0; idx < devInsts.size(); idx++) {
                predict(devInsts[idx], curDecodeInst);
                devInsts[idx].evaluate(curDecodeInst, dev_ner, dev_rel);
                if (!m_options.outBest.empty()) {
                    decodeInstResults.push_back(curDecodeInst);
                }
            }
            std::cout << "Dev finished. Total time taken is: " << double(clock() - time_start) / CLOCKS_PER_SEC << std::endl;
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
            std::cout << "Test start." << std::endl;
            if (!m_options.outBest.empty())
                decodeInstResults.clear();
            test_ner.reset(); test_rel.reset();
            for (int idx = 0; idx < testInsts.size(); idx++) {
                predict(testInsts[idx], curDecodeInst);
                testInsts[idx].evaluate(curDecodeInst, test_ner, test_rel);
                if (bCurIterBetter && !m_options.outBest.empty()) {
                    decodeInstResults.push_back(curDecodeInst);
                }
            }
            std::cout << "Test finished. Total time taken is: " << double(clock() - time_start) / CLOCKS_PER_SEC << std::endl;
            std::cout << "test:" << std::endl;
            test_ner.print();
            test_rel.print();

            if (!m_options.outBest.empty() && bCurIterBetter) {
                m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
            }
        }

        for (int idx = 0; idx < otherInsts.size(); idx++) {
            std::cout << "processing " << m_options.testFiles[idx] << std::endl;
            clock_t time_start = clock();
            if (!m_options.outBest.empty())
                decodeInstResults.clear();
            test_ner.reset(); test_rel.reset();
            for (int idy = 0; idy < otherInsts[idx].size(); idy++) {
                predict(otherInsts[idx][idy], curDecodeInst);
                otherInsts[idx][idy].evaluate(curDecodeInst, test_ner, test_rel);
                if (bCurIterBetter && !m_options.outBest.empty()) {
                    decodeInstResults.push_back(curDecodeInst);
                }
            }
            std::cout << m_options.testFiles[idx] << " finished. Total time taken is: " << double(clock() - time_start) / CLOCKS_PER_SEC << std::endl;
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

        if (iter >= startBeam) {
            m_driver.setGraph(true);
        }

    }
}

void Extractor::predict(Instance &input, CResult &output) {
    m_driver.decode(input, output);
}

void Extractor::test(const string &testFile, const string &outputFile, const string &modelFile) {

}


void Extractor::loadModelFile(const string &inputModelFile) {

}

void Extractor::writeModelFile(const string &outputModelFile) {

}

int main(int argc, char *argv[]) {
    std::string trainFile = "", devFile = "", testFile = "", modelFile = "";
    std::string wordEmbFile = "", optionFile = "";
    std::string outputFile = "";
    bool bTrain = false;
    dsr::Argument_helper ah;
    int memsize = 0;

    ah.new_flag("l", "learn", "train or test", bTrain);
    ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training",
        trainFile);
    ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training",
        devFile);
    ah.new_named_string("test", "testCorpus", "named_string",
        "testing corpus to train a model or input file to test a model, optional when training and must when testing",
        testFile);
    ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
    ah.new_named_string("word", "wordEmbFile", "named_string",
        "pretrained word embedding file to train a model, optional when training", wordEmbFile);
    ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training",
        optionFile);
    ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);
    ah.new_named_int("mem", "memsize", "named_int", "memory allocated for tensor nodes", memsize);

    ah.process(argc, argv);

    Extractor extractor(memsize);
    if (bTrain) {
        extractor.train(trainFile, devFile, testFile, modelFile, optionFile);
    }
    else {
        extractor.test(testFile, outputFile, modelFile);
    }

    //test(argv);
    //ah.write_values(std::cout);

}
