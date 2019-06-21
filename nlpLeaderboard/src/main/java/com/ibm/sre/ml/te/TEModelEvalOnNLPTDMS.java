/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.ibm.sre.ml.te;

import com.ibm.sre.DocTAET;
import com.ibm.sre.NLPResult;
import com.ibm.sre.evaluation.MultiLabelEvaluationMetrics;
import com.ibm.sre.pdfparser.GrobidPDFProcessor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import org.slf4j.LoggerFactory;

/**
 *
 * The code illustrates per_label and per_sample evaluation for the few_shot_setup experiment on NLP-TDMS
 * @author yhou
*/
public class TEModelEvalOnNLPTDMS {

    private Properties prop;
    static org.slf4j.Logger logger = LoggerFactory.getLogger(TEModelEvalOnNLPTDMS.class);

    public TEModelEvalOnNLPTDMS() throws IOException, Exception {
        prop = new Properties();
        prop.load(new FileReader("config.properties"));
    }



    public Map<String, String> getPredictedSore() throws IOException, Exception {
        Map<String, String> scorePrediction = new HashMap();
        String file1 = prop.getProperty("projectPath") + "/" + "data/exp/few-shot-setup/NLP-TDMS/paperVersion/test_score.tsv";
        String file2 = prop.getProperty("projectPath") + "/" + "data/exp/few-shot-setup/NLP-TDMS/paperVersion/test_score_results.tsv";
        BufferedReader br1 = new BufferedReader(new FileReader(file1));
        BufferedReader br2 = new BufferedReader(new FileReader(file2));
        List<String> f1 = new ArrayList();
        List<String> f2 = new ArrayList();
        String line = "";
        while ((line = br1.readLine()) != null) {
            f1.add(line);
        }
        while ((line = br2.readLine()) != null) {
            f2.add(line);
        }
        for (int i = 0; i < f1.size(); i++) {
            String filename = f1.get(i).split("\t")[1].split("#")[0];
            String board = f1.get(i).split("\t")[2];
            String dataset = board.split(",")[0].trim();
            String eval = board.split(",")[1].trim();
            String scoreStr = f1.get(i).split("\t")[1].split("#")[1];
            if (Double.valueOf(f2.get(i).split("\t")[0]) > 0.0) {
                if (scorePrediction.containsKey(filename + "#" + dataset + ":::" + eval)) {
                    String oldScoreStr = scorePrediction.get(filename + "#" + dataset + ":::" + eval).split("#")[0];
                    String oldConfidenceStr = scorePrediction.get(filename + "#" + dataset + ":::" + eval).split("#")[1];
                    Double newConfidenceScore = Double.valueOf(f2.get(i).split("\t")[0]);
                    Double oldConfidenceScore = Double.valueOf(oldConfidenceStr);
                    if (newConfidenceScore > oldConfidenceScore) {
                        scorePrediction.put(filename + "#" + dataset + ":::" + eval, scoreStr + "#" + f2.get(i).split("\t")[0]);
                    }
                } else {
                    scorePrediction.put(filename + "#" + dataset + ":::" + eval, scoreStr + "#" + f2.get(i).split("\t")[0]);
                }

            }
        }
        return scorePrediction;
    }
    

    public void evaluateTDMSExtraction() throws IOException, Exception {
        Map<String, String> scorePrediction = getPredictedSore();
        String file1 = prop.getProperty("projectPath") + "/" + "data/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv";
        String file2 = prop.getProperty("projectPath") + "/" + "data/exp/few-shot-setup/NLP-TDMS/paperVersion/test_results.tsv";

        
        BufferedReader br1 = new BufferedReader(new FileReader(file1));
        BufferedReader br2 = new BufferedReader(new FileReader(file2));
        
        Set<String> excludeTestFiles = getTestFilesInTrainsetWithDifferentName();


        MultiLabelEvaluationMetrics evalMatrix = new MultiLabelEvaluationMetrics();
        Map<String, Set<NLPResult>> resultsPredictionsTestPapers = new HashMap();
        List<String> f1 = new ArrayList();
        List<String> f2 = new ArrayList();
        String line = "";
        while ((line = br1.readLine()) != null) {
            f1.add(line);
        }
        while ((line = br2.readLine()) != null) {
            f2.add(line);
        }
        //
        for (int i = 0; i < f1.size(); i++) {
            String filename = f1.get(i).split("\t")[1];
            if(excludeTestFiles.contains(filename)) continue;
            String leaderboard = f1.get(i).split("\t")[2];
            if (!resultsPredictionsTestPapers.containsKey(filename)) {
                Set<NLPResult> results = new HashSet();
                resultsPredictionsTestPapers.put(filename, results);
            }
            if (Double.valueOf(f2.get(i).split("\t")[0]) > 0.5) {
                if (leaderboard.equalsIgnoreCase("unknow")) {
                    NLPResult result = new NLPResult(filename, "unknow", "unknow");
                    result.setEvaluationMetric("unknow");
                    result.setEvaluationScore("unknow");
                    resultsPredictionsTestPapers.get(filename).add(result);
                } else {
                    String task = leaderboard.split(",")[0].replace(" ", "_").trim();
                    String dataset = leaderboard.split(",")[1].trim();
                    String eval = leaderboard.split(",")[2].trim();
                    NLPResult result = new NLPResult(filename, task, dataset);
                    result.setEvaluationMetric(eval);
                    if (scorePrediction.containsKey(filename + "#" + dataset + ":::" + eval)) {
                        result.setEvaluationScore(scorePrediction.get(filename + "#" + dataset + ":::" + eval).split("#")[0]);
                    }
                    resultsPredictionsTestPapers.get(filename).add(result);
                }
            }
        }
        //collect evaluation labels seen in the train.tsv
        Set<String> evaluatedLabels = new HashSet();
        String file3 = prop.getProperty("projectPath") + "/" + "data/exp/few-shot-setup/NLP-TDMS/paperVersion/train.tsv";
        BufferedReader br3 = new BufferedReader(new FileReader(file3));
        String line3 = "";
        while ((line3 = br3.readLine()) != null) {
            String leaderboard = line3.split("\t")[2];
            if (leaderboard.equalsIgnoreCase("unknow")) {
                continue;
            } else {
                String task = leaderboard.split(",")[0].replace(" ", "_");
                String dataset = leaderboard.split(",")[1];
                String eval = leaderboard.split(",")[2];
                evaluatedLabels.add(task.trim() + ":::" + dataset.trim() + ":::" + eval.trim());
            }
        }
        logger.info("leaderboard evaluation:");
        logger.info("per_label:");
        logger.info(evalMatrix.perLabelEvaluation_Leaderboard_TaskDatasetEvaluationMatrix(resultsPredictionsTestPapers, false, evaluatedLabels));
        logger.info("per_sample:");
        logger.info(evalMatrix.perSampleEvaluation_Leaderboard(resultsPredictionsTestPapers, file1));
    }
    

        public Set<String> getTestFilesInTrainsetWithDifferentName() throws IOException, Exception {
        Map<String, Set<String>> trainTitle = new HashMap();
        Map<String, Set<String>> testTitle = new HashMap();
        Set<String> excludeFiles = new HashSet();
        String file10 = prop.getProperty("projectPath") + "/" + "data/exp/few-shot-setup/NLP-TDMS/paperVersion/train.tsv";
        String file20 = prop.getProperty("projectPath") + "/" + "data/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv";
        String line0 = "";
        BufferedReader br10 = new BufferedReader(new FileReader(file10));
        BufferedReader br20 = new BufferedReader(new FileReader(file20));
        while ((line0 = br10.readLine()) != null) {
            String filename = line0.split("\t")[1];
            String title = line0.split("\t")[3].substring(0, 50);
            if (trainTitle.containsKey(title)) {
                trainTitle.get(title).add(filename);
            } else {
                Set<String> files = new HashSet();
                files.add(filename);
                trainTitle.put(title, files);
            }
        }
        while ((line0 = br20.readLine()) != null) {
            String filename = line0.split("\t")[1];
            String title = line0.split("\t")[3].substring(0, 50);
            if (testTitle.containsKey(title)) {
                testTitle.get(title).add(filename);
            } else {
                Set<String> files = new HashSet();
                files.add(filename);
                testTitle.put(title, files);
            }
        }

        for (String title : testTitle.keySet()) {
            if (trainTitle.keySet().contains(title)) {
                excludeFiles.addAll(testTitle.get(title));
            }
        }
        return excludeFiles;
    }



    
    public static void main(String[] args) throws IOException, Exception{
        TEModelEvalOnNLPTDMS teEval = new TEModelEvalOnNLPTDMS();
        teEval.evaluateTDMSExtraction();
    }

}
