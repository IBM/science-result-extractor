package com.ibm.sre.data.annotation;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.ibm.sre.NLPResult;

/**
 * Loads gold standard TDMS labels per document from task, dataset, and TDMS annotations 
 * 
 * @author charlesj
 *
 */
public class TdmsGoldAnnotation {

    final private Properties prop;
    final private Map<String, Set<String>> taskAnnotation;
    final private Map<String, Set<String>> datasetAnnotation;
    final private Map<String, Set<NLPResult>> tdmsAnnotation;


    public TdmsGoldAnnotation() {
        prop = new Properties();
        try {
            prop.load(new FileReader("config.properties"));
        } catch (IOException e) {
            e.printStackTrace();
        }
        taskAnnotation = loadTaskAnnotation(prop.getProperty("projectPath") + "/" + prop.getProperty("task_annotation"));
        datasetAnnotation = loadDatasetAnnotation(prop.getProperty("projectPath") + "/" + prop.getProperty("dataset_annotation"));
        tdmsAnnotation = loadResultsAnnotation(prop.getProperty("projectPath") + "/" + prop.getProperty("result_annotation"));
    }

    private Map<String, Set<String>> loadTaskAnnotation(String file) {
        Map<String, Set<String>> ret = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line = "";
            while ((line = br.readLine()) != null) {
                String pdfFileName = line.split("\t")[0];
                String taskname = line.split("\t")[1];
                Set<String> tasks = new HashSet<>();
                tasks.add(taskname);
                ret.put(pdfFileName, tasks);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ret;
    }

    private Map<String, Set<String>> loadDatasetAnnotation(String file) {
        Map<String, Set<String>> ret = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line = "";
            while ((line = br.readLine()) != null) {
                String pdfFileName = line.split("\t")[0];
                String dataset = line.split("\t")[1];
                Set<String> datasetNames = new HashSet<>();
                for (int i = 0; i < dataset.split("#").length; i++) {
                    datasetNames.add(dataset.split("#")[i]);
                }
                ret.put(pdfFileName, datasetNames);
            }
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ret;
    }

    //sometimes a paper contain two results for the same leaderboard
    private Map<String, Set<NLPResult>> loadResultsAnnotation(String file) {
        Map<String, Set<NLPResult>> ret = new HashMap<>();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line = "";
            while ((line = br.readLine()) != null) {
                String pdfFileName = line.split("\t")[0];
    
                String resultsstr = line.split("\t")[1];
                Set<NLPResult> results = new HashSet<>();
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
                        result.setEvaluationScore(scoreStr.trim());
                    } else {
                        continue; // the paper doesn't report numbers, skip to collect this as a result
                    }
                    results.add(result);
                }
                ret.put(pdfFileName, results);
            }
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return ret;
    }

    
    public Map<String, Set<String>> getTaskAnnotation() {
        return taskAnnotation;
    }

    public Map<String, Set<String>> getDatasetAnnotation() {
        return datasetAnnotation;
    }

    public Map<String, Set<NLPResult>> getTdmsAnnotation() {
        return tdmsAnnotation;
    }

    public static void main(String[] args) {
        TdmsGoldAnnotation annotation = new TdmsGoldAnnotation();
        for (Entry<String, Set<String>> entry : annotation.getTaskAnnotation().entrySet()) {
            System.out.println(entry.getKey() + ":   " + String.join(",", entry.getValue()));
        }
    }

}
