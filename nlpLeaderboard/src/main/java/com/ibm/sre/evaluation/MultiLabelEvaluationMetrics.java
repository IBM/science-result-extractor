/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.ibm.sre.evaluation;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.slf4j.LoggerFactory;
//import scala.collection.parallel.IterableSplitter$class;

import com.ibm.sre.NLPResult;

/**
 *
 * @author yhou
 */
public class MultiLabelEvaluationMetrics {

    static Map<String, Set<String>> taskAnnotation = new HashMap();
    static Map<String, Set<String>> datasetAnnotation = new HashMap();
    static Map<String, Set<NLPResult>> resultAnnotation = new HashMap();
    private Properties prop;
    static org.slf4j.Logger logger = LoggerFactory.getLogger(MultiLabelEvaluationMetrics.class);

    private static final int TP = 0;
    private static final int FP = 1;
    private static final int FN = 2;
    private static final int TPScore = 3;
    private static final int TP_prediction = 4;

    private Map<String, int[]> taskDatasetMap;
    private Map<String, int[]> leaderboardMap;
    private Map<String, Map<Integer, Set<NLPResult>>> leaderboardMapDebug;

    public MultiLabelEvaluationMetrics() throws IOException {
        taskDatasetMap = new HashMap();
        leaderboardMap = new HashMap();
        leaderboardMapDebug = new HashMap();
        prop = new Properties();
        prop.load(new FileReader("config.properties"));
        loadTaskAnnotation(prop.getProperty("projectPath") + "/" + prop.getProperty("task_annotation"));
        loadDatasetAnnotation(prop.getProperty("projectPath") + "/" + prop.getProperty("dataset_annotation"));
        loadResultsAnnotation(prop.getProperty("projectPath") + "/" + prop.getProperty("result_annotation"));
    }

    public void loadTaskAnnotation(String file) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line = "";
        while ((line = br.readLine()) != null) {
            String pdfFileName = line.split("\t")[0];
            String taskname = line.split("\t")[1];
            Set<String> tasks = new HashSet();
            tasks.add(taskname);
            taskAnnotation.put(pdfFileName, tasks);
        }
        br.close();
    }

    public void loadDatasetAnnotation(String file) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line = "";
        while ((line = br.readLine()) != null) {
            String pdfFileName = line.split("\t")[0];
            String dataset = line.split("\t")[1];
            Set<String> datasetNames = new HashSet();
            for (int i = 0; i < dataset.split("#").length; i++) {
                datasetNames.add(dataset.split("#")[i]);
            }
            datasetAnnotation.put(pdfFileName, datasetNames);
        }
        br.close();
    }

    //sometimes a paper contain two results for the same leaderboard
    public static void loadResultsAnnotation(String file) throws IOException {
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line = "";
        while ((line = br.readLine()) != null) {
            String pdfFileName = line.split("\t")[0];

            String resultsstr = line.split("\t")[1];
            Set<NLPResult> results = new HashSet();
            for (int i = 0; i < resultsstr.split("\\$").length; i++) {
                if (resultsstr.split("\\$")[i].split("#").length < 4) {
                    continue;//the paper doesn't report score
                }
                String task = resultsstr.split("\\$")[i].split("#")[0];
                String dataset = resultsstr.split("\\$")[i].split("#")[1];
                String evaluationMatrix = resultsstr.split("\\$")[i].split("#")[2];
                String scoreStr = resultsstr.split("\\$")[i].split("#")[3];
                NLPResult result = new NLPResult(pdfFileName, task, dataset);
                result.setEvaluationMetric(evaluationMatrix);
                //extract score
                Pattern pattern = Pattern.compile("(\\d+\\.\\d+|\\d+|\\d+\\%|\\d+\\.\\d+\\%|\\d+\\.)");
                Matcher matcher = pattern.matcher(scoreStr);
                if (matcher.find()) {
//                    double score = Double.valueOf(matcher.group(1));
//                    result.setEvaluationScore(score);
                    result.setEvaluationScore(scoreStr.trim());
//                        if(scoreStr.equalsIgnoreCase("﻿39.6"))
//                           result.setEvaluationScore("39.6"); 
                } else {
                    continue; // the paper doesn't report numbers, skip to collect this as a result
                }
                results.add(result);
            }
            resultAnnotation.put(pdfFileName, results);
        }
        br.close();
    }

    private void addPaper_Result(Set<NLPResult> gold, Set<NLPResult> predictions) {
        for (NLPResult re : gold) {
            String leaderboardName = re.taskName + ":::" + re.datasetName + ":::" + re.evaluationMetric;
            int[] counts = leaderboardMap.get(leaderboardName);
            Map<Integer, Set<NLPResult>> debugInfo = leaderboardMapDebug.get(leaderboardName);
            if (debugInfo == null) {
                debugInfo = new HashMap();
                debugInfo.put(TP, new HashSet<NLPResult>());
                debugInfo.put(FP, new HashSet<NLPResult>());
                debugInfo.put(FN, new HashSet<NLPResult>());
                debugInfo.put(TPScore, new HashSet<NLPResult>());
                debugInfo.put(TP_prediction, new HashSet<NLPResult>());
                leaderboardMapDebug.put(leaderboardName, debugInfo);
            }
            if (counts == null) {
                counts = new int[4]; //counts for tp, fp, fn, and TPScore
                leaderboardMap.put(leaderboardName, counts);
            }
            boolean reInPrediction = false;

            for (NLPResult prediction : predictions) {
                if (prediction.equals_relax(re)) {
                    reInPrediction = true;
                    counts[TP]++;
                    debugInfo.get(TP).add(re);
                    debugInfo.get(TP_prediction).add(prediction);
//                    if (prediction.score == re.score) {
//                    if(re.score.contains(prediction.score)&&!prediction.score.isEmpty()&&!re.score.equalsIgnoreCase(prediction.score)){
//                        System.err.println(re.paperName + re.score + ":" + prediction.score);
//                    }

                    if (re.score.matches("\\d+|\\d+\\.\\d+") && prediction.score.matches("\\d+|\\d+\\.\\d+")) {
                        double d1 = Double.valueOf(re.score);
                        double d2 = Double.valueOf(prediction.score);
                        if (d1 == d2) {
                            counts[TPScore]++;
                            debugInfo.get(TPScore).add(re);
                        }
                    } else if (re.score.equalsIgnoreCase(prediction.score) && !prediction.score.isEmpty()) {
                        counts[TPScore]++;
                        debugInfo.get(TPScore).add(re);
                    }
                }
            }

            if (!reInPrediction) {
                counts[FN]++;
                debugInfo.get(FN).add(re);
            }
        }
        for (NLPResult re : predictions) {
            String leaderboardName = re.taskName + ":::" + re.datasetName + ":::" + re.evaluationMetric;
            int[] counts = leaderboardMap.get(leaderboardName);
            Map<Integer, Set<NLPResult>> debugInfo = leaderboardMapDebug.get(leaderboardName);
            if (debugInfo == null) {
                debugInfo = new HashMap();
                debugInfo.put(TP, new HashSet<NLPResult>());
                debugInfo.put(FP, new HashSet<NLPResult>());
                debugInfo.put(FN, new HashSet<NLPResult>());
                debugInfo.put(TPScore, new HashSet<NLPResult>());
                debugInfo.put(TP_prediction, new HashSet<NLPResult>());
                leaderboardMapDebug.put(leaderboardName, debugInfo);
            }
            if (counts == null) {
                counts = new int[4];
                leaderboardMap.put(leaderboardName, counts);
            }
            boolean reInGold = false;
            for (NLPResult goldlabel : gold) {
                if (re.equals_relax(goldlabel)) {
                    reInGold = true;
                    break;
                }
            }
            if (counts != null && !reInGold) {
                counts[FP]++;
                debugInfo.get(FP).add(re);
            }
        }
    }

    private void addPaper(Set<String> gold, Set<String> predictions) {
        for (String datasetTask : gold) {
            int[] counts = taskDatasetMap.get(datasetTask);
            if (counts == null) {
                counts = new int[4];  // counts for tp, fp, fn, tpScore=0 
                taskDatasetMap.put(datasetTask, counts);
            }
            if (predictions.contains(datasetTask)) {
                counts[TP]++; // increment tp
            } else {
                counts[FN]++; // increment fn
            }
        }
        for (String datasetTask : predictions) {
            int[] counts = taskDatasetMap.get(datasetTask);
            if (counts == null) {
                counts = new int[4];  // counts for tp, fp, fn
                taskDatasetMap.put(datasetTask, counts);
            }
            if (!gold.contains(datasetTask)) {
                counts[FP]++;  // increment fp
            }
        }
    }

    //only evaluated labeled appearing in the set of evaluatedLables
    private String perLabelEvaluation(Map<String, int[]> labelMap, Set<String> evaluatedLabels) {
        double f1Acc = 0.0;
        double precAcc = 0.0;
        double recAcc = 0.0;
        double scoreAcc = 0.0;
        int tp_all = 0;
        int fp_all = 0;
        int fn_all = 0;
        int tp_score_all = 0;
        StringBuffer sb = new StringBuffer();
        sb.append("Class\tTP\tFP\tFN\tTPScore\tPrec\tRec\tF1\tscoreAcc\n");
        for (Entry<String, int[]> entry : labelMap.entrySet()) {
            String taskDataset = entry.getKey();
            if (!evaluatedLabels.contains(taskDataset)) {
                continue;
            }
            int[] counts = entry.getValue();
            tp_all=tp_all + counts[TP];
            fp_all=fp_all + counts[FP];
            fn_all=fn_all + counts[FN];
            tp_score_all = tp_score_all +counts[TPScore];
            sb.append(taskDataset);
            sb.append('\t');
            for (int i = 0; i < 4; i++) {
                sb.append(counts[i]);
                sb.append('\t');
            }
            double prec = precision(counts[TP], counts[FP]);
            double accScore = precision(counts[TPScore], counts[TP] - counts[TPScore]);
//            double accScore = precision(counts[TPScore], counts[TP] - counts[TPScore] + counts[FP]);
            double rec = recall(counts[TP], counts[FN]);
            double f1 = f1Score(prec, rec);
            f1Acc += f1;
            precAcc += prec;
            recAcc += rec;
            scoreAcc += accScore;
            sb.append(String.format("%.3f\t%.3f\t%.3f\t%.3f\n", prec, rec, f1, accScore));
        }
//        sb.append(String.format("Macro-averaged precision: %.4f", precAcc / labelMap.size()));
//        sb.append(String.format("Macro-averaged recall: %.4f", recAcc / labelMap.size()));
//        sb.append(String.format("Macro-averaged F1: %.4f", f1Acc / labelMap.size()));
//        sb.append(String.format("Macro-averaged score extraction acc: %.4f", scoreAcc / labelMap.size()));
        sb.append(String.format("Macro-averaged precision: %.4f ", precAcc / evaluatedLabels.size()));
        sb.append(String.format("Macro-averaged recall: %.4f ", recAcc / evaluatedLabels.size()));
        sb.append(String.format("Macro-averaged F1: %.4f ", f1Acc / evaluatedLabels.size()));
        sb.append(String.format("Macro-averaged score extraction acc: %.4f ", scoreAcc / evaluatedLabels.size()));
        sb.append("\n");
        double micro_p = precision(tp_all, fp_all);
        double micro_r = recall(tp_all, fn_all);
        double micro_scoreAcc = tp_score_all/(tp_all+0.0);
        sb.append(String.format("Micro-averaged precision: %.4f ", micro_p));
        sb.append(String.format("Micro-averaged recall: %.4f ", micro_r));
        sb.append(String.format("Micro-averaged F1: %.4f ", f1Score(micro_p, micro_r)));
        sb.append(String.format("Micro-averaged score extraction acc: %.4f ", micro_scoreAcc));
        return sb.toString();
    }

    private double recall(int tp, int fn) {
        return precision(tp, fn);
    }

    private double precision(int tp, int fp) {
        if (tp + fp <= 0) {
            return 0.0;
        } else {
            return ((double) tp) / (tp + fp);
        }
    }

    private double f1Score(double prec, double recall) {
        if (prec + recall <= 0.0) {
            return 0.0;
        } else {
            return 2 * prec * recall / (prec + recall);
        }
    }

    private String perSampleEvaluation(Map<String, Set<String>> prediction, Map<String, Set<String>> gold) {
        double macroF1 = 0.0;
        double macroP = 0.0;
        double macroR = 0.0;
        StringBuffer sb = new StringBuffer();
        sb.append("Paper\tTP\tFP\tFN\tPrec\tRec\tF1\n");
        for (Entry<String, Set<String>> item : prediction.entrySet()) {
            int tp = 0;
            int fp = 0;
            int fn = 0;
            String paperID = item.getKey();
            Set<String> paperPrediction = item.getValue();
            if (paperPrediction.isEmpty()) {
                sb.append(paperID + "\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\n");
                continue;
            }
            if (gold.containsKey(paperID)) {
                Set<String> goldAnnotation = gold.get(paperID);
                for (String predicted : paperPrediction) {
                    if (goldAnnotation.contains(predicted)) {
                        tp++;
                    } else {
                        fp++;
                    }
                }
                for (String goldStr : goldAnnotation) {
                    if (!paperPrediction.contains(goldStr)) {
                        fn++;
                    }
                }
            }
            double p = precision(tp, fp);
            double r = recall(tp, fn);
            double f = f1Score(p, r);
            macroP += p;
            macroR += r;
            macroF1 += f;
            sb.append(paperID + "\t" + tp + "\t" + fp + "\t" + fn + "\t");
            sb.append(String.format("%.3f\t%.3f\t%.3f\n", p, r, f));
        }
        sb.append(String.format("Macro-averaged Precision: %.4f ", macroP / prediction.size()));
        sb.append(String.format("Macro-averaged Recall: %.4f ", macroR / prediction.size()));
        sb.append(String.format("Macro-averaged fscore: %.4f ", macroF1 / prediction.size()));
        return sb.toString();
    }

    private String perSampleEvaluation_leaderboardRelax(Map<String, Set<NLPResult>> prediction, Map<String, Set<NLPResult>> gold) {
        double macroF1 = 0.0;
        double macroP = 0.0;
        double macroR = 0.0;
        double microF1 = 0.0;
        double microP = 0.0;
        double microR = 0.0;
        int tp_all = 0;
        int fp_all = 0;
        int fn_all = 0;
        StringBuffer sb = new StringBuffer();
        sb.append("Paper\tTP\tFP\tFN\tPrec\tRec\tF1\n");
        for (Entry<String, Set<NLPResult>> item : prediction.entrySet()) {
            int tp = 0;
            int fp = 0;
            int fn = 0;
            String paperID = item.getKey();
            Set<NLPResult> paperPrediction = item.getValue();
            if (gold.containsKey(paperID)) {
                Set<NLPResult> goldAnnotation = gold.get(paperID);
                for (NLPResult predicted : paperPrediction) {
                    boolean goldContainPredicted = false;
                    for(NLPResult goldAnno: goldAnnotation){
                        if(goldAnno.equals_relax(predicted)){
                          goldContainPredicted = true;
                        break;
                      }
                    }
                    if (goldContainPredicted) {
                        tp++;
                        tp_all++;
                    } else {
                        fp++;
                        fp_all++;
                    }
                }
                for (NLPResult goldAnno : goldAnnotation) {
                    boolean predictionContaingold = false;
                    for(NLPResult predict: paperPrediction){ 
                        if(predict.equals_relax(goldAnno)){
                            predictionContaingold = true;
                            break;
                        }
                    }
                    if (!predictionContaingold) {
                        fn++;
                        fn_all++;
                    }
                }
            }
            double p = precision(tp, fp);
            double r = recall(tp, fn);
            double f = f1Score(p, r);
            macroP += p;
            macroR += r;
            macroF1 += f;
//            sb.append(paperID + "\t" + tp + "\t" + fp + "\t" + fn + "\t");
//            sb.append(String.format("%.3f\t%.3f\t%.3f\n", p, r, f));
        }
        double p_micro = precision(tp_all, fp_all);
        double r_micro = recall(tp_all, fn_all);
        double f_micro = f1Score(p_micro, r_micro);
        sb.append(String.format("Macro-averaged Precision: %.4f ", macroP / prediction.size()));
        sb.append(String.format("Macro-averaged Recall: %.4f ", macroR / prediction.size()));
        sb.append(String.format("Macro-averaged fscore: %.4f ", macroF1 / prediction.size()));
        sb.append(String.format("Micro-averaged Precision: %.4f ", p_micro));
        sb.append(String.format("Micro-averaged Recall: %.4f ", r_micro));
        sb.append(String.format("Micro-averaged fscore: %.4f ", f_micro));
        return sb.toString();
    }
    
        private String perSampleEvaluation_leaderboardStrict(Map<String, Set<NLPResult>> prediction, Map<String, Set<NLPResult>> gold) {
        double macroF1 = 0.0;
        double macroP = 0.0;
        double macroR = 0.0;
        double microF1 = 0.0;
        double microP = 0.0;
        double microR = 0.0;
        int tp_all = 0;
        int fp_all = 0;
        int fn_all = 0;
        StringBuffer sb = new StringBuffer();
        sb.append("Paper\tTP\tFP\tFN\tPrec\tRec\tF1\n");
        for (Entry<String, Set<NLPResult>> item : prediction.entrySet()) {
            int tp = 0;
            int fp = 0;
            int fn = 0;
            String paperID = item.getKey();
            Set<NLPResult> paperPrediction = item.getValue();
            if (gold.containsKey(paperID)) {
                Set<NLPResult> goldAnnotation = gold.get(paperID);
                for (NLPResult predicted : paperPrediction) {
                    boolean goldContainPredicted = false;
                    for(NLPResult goldAnno: goldAnnotation){
                        if(goldAnno.equals_strict(predicted)){
                          goldContainPredicted = true;
                        break;
                      }
                    }
                    if (goldContainPredicted) {
                        tp++;
                        tp_all++;
                    } else {
                        fp++;
                        fp_all++;
                    }
                }
                for (NLPResult goldAnno : goldAnnotation) {
                    boolean predictionContaingold = false;
                    for(NLPResult predict: paperPrediction){ 
                        if(predict.equals_strict(goldAnno)){
                            predictionContaingold = true;
                            break;
                        }
                    }
                    if (!predictionContaingold) {
                        fn++;
                        fn_all++;
                    }
                }
            }
            double p = precision(tp, fp);
            double r = recall(tp, fn);
            double f = f1Score(p, r);
            macroP += p;
            macroR += r;
            macroF1 += f;
//            sb.append(paperID + "\t" + tp + "\t" + fp + "\t" + fn + "\t");
//            sb.append(String.format("%.3f\t%.3f\t%.3f\n", p, r, f));
        }
        double p_micro = precision(tp_all, fp_all);
        double r_micro = recall(tp_all, fn_all);
        double f_micro = f1Score(p_micro, r_micro);
        sb.append(String.format("Macro-averaged Precision: %.4f ", macroP / prediction.size()));
        sb.append(String.format("Macro-averaged Recall: %.4f ", macroR / prediction.size()));
        sb.append(String.format("Macro-averaged fscore: %.4f ", macroF1 / prediction.size()));
        sb.append(String.format("Micro-averaged Precision: %.4f ", p_micro));
        sb.append(String.format("Micro-averaged Recall: %.4f ", r_micro));
        sb.append(String.format("Micro-averaged fscore: %.4f ", f_micro));
        return sb.toString();
    }


    private String perLabelEvaluation(Map<String, Set<String>> prediction, Map<String, Set<String>> gold) {
        Set<String> evaluateLabels = new HashSet();
        for (Entry<String, Set<String>> item : gold.entrySet()) {
            evaluateLabels.addAll(item.getValue());
        }
        for (Entry<String, Set<String>> item : prediction.entrySet()) {
            String paperID = item.getKey();
            Set<String> paperPrediction = item.getValue();
            if (gold.containsKey(paperID)) {
                addPaper(gold.get(paperID), paperPrediction);
            }
        }
        return perLabelEvaluation(taskDatasetMap, evaluateLabels);
    }

    public String perLabelEvaluation_Task(Map<String, Set<String>> prediction) {
        return perLabelEvaluation(prediction, taskAnnotation);
    }

    public String perLabelEvaluation_Dataset(Map<String, Set<String>> prediction) {
        return perLabelEvaluation(prediction, datasetAnnotation);
    }

    public String perSampleEvaluation_Task(Map<String, Set<String>> prediction) {
        return perSampleEvaluation(prediction, taskAnnotation);
    }

    public String perSampleEvaluation_Dataset(Map<String, Set<String>> prediction) {
        return perSampleEvaluation(prediction, datasetAnnotation);
    }

    public String perSampleEvaluation_Leaderboard(Map<String, Set<NLPResult>> prediction, String testFile) throws IOException {
        BufferedReader br1 = new BufferedReader(new FileReader(testFile));
        Map<String, Set<NLPResult>> revisedGoldAnnotation = new HashMap();
        Map<String, Set<NLPResult>> revisedGoldAnnotation_wo_unknow = new HashMap();
        Map<String, Set<NLPResult>> prediction_on_known = new HashMap();
        for(String file: prediction.keySet())
            prediction_on_known.put(file, prediction.get(file));
        
        String line = "";
        while ((line = br1.readLine()) != null) {
            String goldLabel = line.split("\t")[0];
            String filename = line.split("\t")[1];
            String leaderboard = line.split("\t")[2];
            if (!revisedGoldAnnotation.containsKey(filename)) {
                Set<NLPResult> anno = new HashSet();
                Set<NLPResult> anno1 = new HashSet();
                revisedGoldAnnotation.put(filename, anno);
                revisedGoldAnnotation_wo_unknow.put(filename, anno1);
            }
            if (goldLabel.equalsIgnoreCase("true")) {
                if (leaderboard.equalsIgnoreCase("unknow")) {
                    NLPResult result = new NLPResult(filename, "unknow", "unknow");
                    result.setEvaluationMetric("unknow");
                    result.setEvaluationScore("unknow");
                    revisedGoldAnnotation.get(filename).add(result);
                    prediction_on_known.remove(filename);
                } else {
                    String task = leaderboard.split(",")[0].replace(" ", "_").trim();
                    String dataset = leaderboard.split(",")[1].trim();
                    String eval = leaderboard.split(",")[2].trim();
                    Set<NLPResult> originalAnnotatedResults = resultAnnotation.get(filename);
                    for (NLPResult re : originalAnnotatedResults) {
                        if (re.taskName.equals(task) && re.datasetName.equalsIgnoreCase(dataset) && re.evaluationMetric.equalsIgnoreCase(eval)) {
                            revisedGoldAnnotation.get(filename).add(re);
                            revisedGoldAnnotation_wo_unknow.get(filename).add(re);
                        }
                    }
                }
            }
        }
        br1.close();
        //for debug
//        for(String file: prediction.keySet()){
//            System.err.println(file);
//            System.err.println("prediction");
//            for(NLPResult predict: prediction.get(file))
//                System.err.println(predict.toString());
//            System.err.println("goldAnno");
//            for(NLPResult anno: revisedGoldAnnotation.get(file))
//                System.err.println(anno.toString());
//        }
        
        
        String relaxEvalResult = perSampleEvaluation_leaderboardRelax(prediction, revisedGoldAnnotation);
        String relaxEvalResult_wo_unknow = perSampleEvaluation_leaderboardRelax(prediction_on_known, revisedGoldAnnotation_wo_unknow);
        String strictEvalResult = perSampleEvaluation_leaderboardStrict(prediction, revisedGoldAnnotation);
        String strictEvalResult_wo_unknow = perSampleEvaluation_leaderboardStrict(prediction_on_known, revisedGoldAnnotation_wo_unknow);


        return "relaxEval + \n" + relaxEvalResult + "\n\nstrictEval\n" + strictEvalResult
               +"\n\nrelaxEval(wo_unknow)\n" + relaxEvalResult_wo_unknow
                + "\n\nstrictEval(wo_unknow)\n" + strictEvalResult_wo_unknow;

        //return relaxEvalResult_wo_unknow;
    }

    //Macro-average Precision/Recall/f1 for task+dataset+evaluationMetric; then accuracy of numeric score extraction
    //numeric score is the best score achieved by a paper for a task on a specific dataset
    private String perLabelEvaluation_Leaderboard_TaskDatasetEvaluationMatrix(Map<String, Set<NLPResult>> prediction, Map<String, Set<NLPResult>> gold, boolean printDebug) {
        Set<String> evaluateLabels = new HashSet();
        for (Entry<String, Set<NLPResult>> item : gold.entrySet()) {
            for (NLPResult re : item.getValue()) {
                String leaderboardName = re.taskName + ":::" + re.datasetName + ":::" + re.evaluationMetric;
                evaluateLabels.add(leaderboardName);
            }
        }
        for (Entry<String, Set<NLPResult>> item : prediction.entrySet()) {
            String paperID = item.getKey();
            Set<NLPResult> paperPrediction = item.getValue();
            if (gold.containsKey(paperID)) {
                addPaper_Result(gold.get(paperID), paperPrediction);
            }
        }
        if (printDebug) {
            return perLabelEvaluation(leaderboardMap, evaluateLabels) + "\n" + perLabelEvaluation_Leaderboard_debuginfo(leaderboardMapDebug);
        }
        return perLabelEvaluation(leaderboardMap, evaluateLabels);
    }

    private String perLabelEvaluation_Leaderboard_TaskDatasetEvaluationMatrix(Map<String, Set<NLPResult>> prediction, Map<String, Set<NLPResult>> gold, boolean printDebug, Set<String> evaluateLabels) {
//        Set<String> evaluateLabels = new HashSet();
//        for(Entry<String, Set<NLPResult>> item : gold.entrySet()){
//            for(NLPResult re: item.getValue()){
//                String leaderboardName = re.taskName + ":::" + re.datasetName + ":::" + re.evaluationMetric;
//                evaluateLabels.add(leaderboardName);
//            }
//        }
        for (Entry<String, Set<NLPResult>> item : prediction.entrySet()) {
            String paperID = item.getKey();
            Set<NLPResult> paperPrediction = item.getValue();
            if (gold.containsKey(paperID)) {
                addPaper_Result(gold.get(paperID), paperPrediction);
            }
        }
        if (printDebug) {
            return perLabelEvaluation(leaderboardMap, evaluateLabels) + "\n" + perLabelEvaluation_Leaderboard_debuginfo(leaderboardMapDebug);
        }
        return perLabelEvaluation(leaderboardMap, evaluateLabels);
    }

    private String perLabelEvaluation_Leaderboard_debuginfo(Map<String, Map<Integer, Set<NLPResult>>> debugMap) {
        StringBuffer sb = new StringBuffer();
        sb.append("Leaderboard prediction debug:\n");
        for (Entry<String, Map<Integer, Set<NLPResult>>> item : debugMap.entrySet()) {
            sb.append("leaderboard:" + item.getKey()).append("\n");
            sb.append("TP:\n");
            for (NLPResult result : item.getValue().get(TP)) {
                sb.append(result.toString()).append("\n");
            }
            sb.append("FN:\n");
            for (NLPResult result : item.getValue().get(FN)) {
                sb.append(result.toString()).append("\n");
            }
            sb.append("FP:\n");
            for (NLPResult result : item.getValue().get(FP)) {
                sb.append(result.toString()).append("\n");
            }
            sb.append("TP_predicted:\n");
            for (NLPResult result : item.getValue().get(TP_prediction)) {
                sb.append(result.toString()).append("\n");
            }
        }
        return sb.toString();

    }

    public String perLabelEvaluation_Leaderboard_TaskDatasetEvaluationMatrix(Map<String, Set<NLPResult>> prediction, boolean printDebug) {
        return perLabelEvaluation_Leaderboard_TaskDatasetEvaluationMatrix(prediction, resultAnnotation, printDebug);
    }

    public String perLabelEvaluation_Leaderboard_TaskDatasetEvaluationMatrix(Map<String, Set<NLPResult>> prediction, boolean printDebug, Set<String> evaluatedLabels) {
        return perLabelEvaluation_Leaderboard_TaskDatasetEvaluationMatrix(prediction, resultAnnotation, printDebug, evaluatedLabels);
    }

}
