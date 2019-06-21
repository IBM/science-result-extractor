package com.ibm.sre.evaluation;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.commons.lang3.StringUtils;

import com.ibm.sre.NLPResult;

import au.com.bytecode.opencsv.CSVReader;

public class EvaluateCsv {

    private static final String TRAIN_FILENAME = "../data/exp1/ablationfull/train.tsv";
    private static final String TEST_FILENAME = "../data/exp1/ablationfull/test.tsv";
    private static final String INPUT_CSV = "/tmp/output-onelabel.csv";
    private static final String INPUT_SCORE_CSV = "/tmp/score_output.csv";
    private static final String OUTPUT_CSV = "/tmp/eval_output.tsv";

    public static void main(String[] args) {
        if (args.length == 0) {
            args = new String[]{INPUT_CSV, INPUT_SCORE_CSV};
        }
        try {
            Map<String, Set<NLPResult>> prediction = new HashMap<>(170);
            Map<String, Set<String>> taskPrediction = new HashMap<>(170);
            Map<String, Set<String>> dsPrediction = new HashMap<>(170);
            
            // load file-score map (if present)
            File inpScore = new File(INPUT_SCORE_CSV);
            Map<String, String> fileScoreMap = null;
            if (inpScore.exists()) {
                fileScoreMap = loadFileScoreMap(inpScore);
            }
            
            // read CSV with predictions (in this case from scikit learn
            // format is paper name, task, dataset, evaluation metric, score
            CSVReader csvReader = new CSVReader(new FileReader(args[0]));
            String[] row = csvReader.readNext();  // skip header
            while ((row = csvReader.readNext()) != null) {
                String file = row[0];
                String score = null;
                if (fileScoreMap != null) {
                    score = fileScoreMap.get(file);
                }
                Set<String> taskSet;
                Set<String> dsSet;
                Set<String> metricSet;
                if (row.length == 3) {  // merged label
                    String labelStr = row[1];
                    if (score == null)
                        score = row[2];
                    // get sets from strings
                    Set<String>[] sets = parsePythonSetMerged(labelStr);
                    taskSet = sets[0];
                    dsSet = sets[1];
                    metricSet = sets[2];
                    
                } else {  // assuming at least 5 cols
                    String tasksStr = row[1];
                    String datasetsStr = row[2];
                    String metricsStr = row[3];
                    if (score == null)
                        score = row[4];

                    // get sets from strings
                    taskSet = parsePythonSet(tasksStr);
                    dsSet = parsePythonSet(datasetsStr);
                    metricSet = parsePythonSet(metricsStr);
                }
                

                Set<NLPResult> results = prediction.get(file);
                if (results == null) {
                    results = new HashSet<>();
                    prediction.put(file, results);
                }
                // TODO now assuming one row per file, might change
                Set<String> taskPred = new HashSet<>();
                taskPrediction.put(file, taskPred);
                Set<String> dsPred = new HashSet<>();
                dsPrediction.put(file, dsPred);                
                
                // get all combinations
                for (String task : taskSet) {
                    // task seems to need underscores instead of spaces
                    task = task.replace(' ', '_');
                    taskPred.add(task);
                    for (String dataset : dsSet) {
                        dsPred.add(dataset);
                        for (String metric : metricSet) {
                            NLPResult nlpResult = new NLPResult(file, task, dataset);
                            nlpResult.setEvaluationMetric(metric);
                            nlpResult.setEvaluationScore(score);
                            results.add(nlpResult);
                        }
                    }
                }
            }
            csvReader.close();
            
            MultiLabelEvaluationMetrics eval = new MultiLabelEvaluationMetrics();
            Set<String> labelsForEvaluation = getLabelsForEvalutation();
            BufferedWriter bw = new BufferedWriter(new FileWriter(OUTPUT_CSV));
            String evalString = "Per Label Full\n" + eval.perLabelEvaluation_Leaderboard_TaskDatasetEvaluationMatrix(prediction, false, labelsForEvaluation) + "\n\n";
            evalString += "Per Label Task\n" + eval.perLabelEvaluation_Task(taskPrediction) + "\n\n";
            evalString += "Per Label Dataset\n" + eval.perLabelEvaluation_Dataset(dsPrediction) + "\n\n";
            evalString += "Per Sample Full\n" + eval.perSampleEvaluation_Leaderboard(prediction, TEST_FILENAME) + "\n\n";
            evalString += "Per Sample Task\n" + eval.perSampleEvaluation_Task(taskPrediction) + "\n\n";
            evalString += "Per Sample Dataset\n" + eval.perSampleEvaluation_Dataset(dsPrediction) + "\n\n";
            bw.write(evalString);
            System.out.println(evalString);
            bw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        
    }

    private static Map<String, String> loadFileScoreMap(File inpScore) throws IOException {
        Map<String, String> retMap = new HashMap<>();
        CSVReader csvReader = new CSVReader(new FileReader(inpScore));
        String[] row = csvReader.readNext();  // skip header
        while ((row = csvReader.readNext()) != null) {
            String file = row[0];
            String score = row[1];
            retMap.put(file, score);
        }
        csvReader.close();
        return retMap;
    }

    private static Set<String>[] parsePythonSetMerged(String setString) {
        // split up "('relationship extraction, SemEval-2010 Task 8, F1',)"
        Set<String> tskSet = new HashSet<>();
        Set<String> dsSet = new HashSet<>();
        Set<String> metSet = new HashSet<>();
        setString = setString.substring(1, setString.length()-1);  // strip leading and trailing parentheses
        String[] entries = setString.split(",\\s*");
        if (entries.length > 3)
            throw new UnsupportedOperationException("Can't handle multiple 'merged' labels");
        if (entries[0].equals("'unknow'")) {
            tskSet.add(StringUtils.strip(entries[0], "'"));  // strip leading and trailing quotes
            dsSet.add(StringUtils.strip(entries[0], "'"));
            metSet.add(StringUtils.strip(entries[0], "'"));
        } else if (entries.length == 3) {
            tskSet.add(StringUtils.strip(entries[0], "'"));
            dsSet.add(StringUtils.strip(entries[1], "'"));
            metSet.add(StringUtils.strip(entries[2], "'"));
        } else {
            System.err.println("Unexpected label: " + setString);
        }
        return new Set[] {tskSet, dsSet, metSet};
    }

    private static Set<String> parsePythonSet(String setString) {
        // split up "('ROUGE-1', 'ROUGE-L')"
        // FIXME counting on tasks, dataset, and metric names not to have commas
        Set<String> retSet = new HashSet<>();
        setString = setString.substring(1, setString.length()-1);  // strip leading and trailing parentheses
        String[] entries = setString.split(",\\s*");
        for (String entry : entries) {
            entry = entry.substring(1, entry.length()-1);  // strip leading and trailing quotes
            retSet.add(entry);
        }
        return retSet;
    }

    private static Set<String> getLabelsForEvalutation() {
        //collect evaluation labels seen in the train.tsv
        Set<String> evaluatedLabels = new HashSet<>();
        try (BufferedReader br = new BufferedReader(new FileReader(TRAIN_FILENAME))) {
            String line = "";
            while ((line = br.readLine()) != null) {
                String leaderboard = line.split("\t")[2];
                if (leaderboard.equalsIgnoreCase("unknow")) {
                    continue;
                } else {
                    String[] cols = leaderboard.split(",");
                    String task = cols[0].replace(" ", "_").trim();
                    String dataset = cols[1].trim();
                    String eval = cols[2].trim();
                    evaluatedLabels.add(task + ":::" + dataset + ":::" + eval);
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return evaluatedLabels;
    }

}
