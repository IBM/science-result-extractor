/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.ibm.sre.tdmsie;

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

import com.ibm.sre.DocTAET;
import com.ibm.sre.NLPResult;
import com.ibm.sre.evaluation.MultiLabelEvaluationMetrics;

/**
 * prepare testing data for TDM and TDMS extraction for the given PDF files
 *
 * @author yhou
 */
public class GenerateTestDataOnPDFPapers {

    private Properties prop;

    public GenerateTestDataOnPDFPapers() throws IOException, Exception {
        prop = new Properties();
        prop.load(new FileReader("config.properties"));
    }

    
    //generate test tsv data for TDM Pairs prediction for pdf papers
    //every evaluatedLabel has the format of "task, dataset, metric"
    //evaluatedLabels could be collected from the training data
    public void generateTestData4TDMPrediction(String PdfFileFolder, String OutputFile, Set<String> evaluatedLabels) throws IOException, Exception {
        FileWriter writer1 = new FileWriter(new File(OutputFile));
        StringBuffer sb1 = new StringBuffer();
        File PDFFile = new File(PdfFileFolder);
        for (File pdfFile : PDFFile.listFiles()) {
            String filename = pdfFile.getName();
            if (!filename.contains(".pdf")) {
                continue;
            }
            String docTEATStr = DocTAET.getDocTAETRepresentation(pdfFile.getAbsolutePath());
            for (String TDMLabel : evaluatedLabels) {
                sb1.append("true" + "\t" + filename + "\t" + TDMLabel + "\t" + docTEATStr).append("\n");
            }
        }
        writer1.write(sb1.toString());
        writer1.close();
    }

    //generate test tsv data for score prediction for pdf papers
    //every evaluatedLabel has the format of "dataset, metric"
    //evaluatedLabels should be collected from the TDM prediction results for the same pdf files
    //Alternatively, you can also collected them from the training data    
    public void generateTestData4ScorePrediction(String PdfFileFolder, String OutputFile, Set<String> evaluatedLabels) throws IOException, Exception {
        FileWriter writer1 = new FileWriter(new File(OutputFile));
        StringBuffer sb1 = new StringBuffer();
        File PDFFile = new File(PdfFileFolder);
        for (File pdfFile : PDFFile.listFiles()) {
            String filename = pdfFile.getName();
            if (!filename.contains(".pdf")) {
                continue;
            }
            List<String> numbersAndContext = DocTAET.getTableBoldNumberContext(pdfFile.getAbsolutePath());
            for (String DMLabel : evaluatedLabels) {
                for (String numberInfo : numbersAndContext) {
                    sb1.append("true" + "\t" + filename + "#" + numberInfo.split("#")[0] + "\t" + DMLabel + "\t" + numberInfo.split("#")[1]).append("\n");
                }
            }
        }
        writer1.write(sb1.toString());
        writer1.close();
    }

    //generate test tsv data for TDM Pairs prediction for pdf papers
    //every evaluatedLabel has the format of "task, dataset, metric"
    //evaluatedLabels is collected from the training data
    public void generateTestData4TDMPrediction(String PdfFileFolder, String OutputFile) throws IOException, Exception {
        //collect predicting labels seen in the train.tsv
        Set<String> evaluatedLabels = new HashSet();
        String file3 = prop.getProperty("projectPath") + "/" + "data/exp/few-shot-setup/NLP-TDMS/train.tsv";
        BufferedReader br3 = new BufferedReader(new FileReader(file3));
        String line3 = "";
        while ((line3 = br3.readLine()) != null) {
            String leaderboard = line3.split("\t")[2];
            evaluatedLabels.add(leaderboard);
        }

        FileWriter writer1 = new FileWriter(new File(OutputFile));
        StringBuffer sb1 = new StringBuffer();
        File PDFFile = new File(PdfFileFolder);
        for (File pdfFile : PDFFile.listFiles()) {
            String filename = pdfFile.getName();
            if (!filename.contains(".pdf")) {
                continue;
            }
            String docTEATStr = DocTAET.getDocTAETRepresentation(pdfFile.getAbsolutePath());
            for (String TDMLabel : evaluatedLabels) {
                sb1.append("true" + "\t" + filename + "\t" + TDMLabel + "\t" + docTEATStr).append("\n");
            }
        }
        writer1.write(sb1.toString());
        writer1.close();
    }

    //generate test tsv data for score prediction for pdf papers
    //every evaluatedLabel has the format of "dataset, metric"
    //evaluatedLabels is collected from the TDM triples in the training data
    public void generateTestData4ScorePrediction(String PdfFileFolder, String OutputFile) throws IOException, Exception {
        //collect predicting labels seen in the train.tsv
        Set<String> evaluatedLabels = new HashSet();
        String file3 = prop.getProperty("projectPath") + "/" + "data/exp/few-shot-setup/NLP-TDMS/train.tsv";
        BufferedReader br3 = new BufferedReader(new FileReader(file3));
        String line3 = "";
        while ((line3 = br3.readLine()) != null) {
            String leaderboard = line3.split("\t")[2];
            if (leaderboard.equalsIgnoreCase("unknow")) {
                continue;
            }
            String task = leaderboard.split(",")[0];
            String dataset = leaderboard.split(",")[1];
            String eval = leaderboard.split(",")[2];
            evaluatedLabels.add(dataset.trim() + ", " + eval.trim());
        }

        FileWriter writer1 = new FileWriter(new File(OutputFile));
        StringBuffer sb1 = new StringBuffer();
        File PDFFile = new File(PdfFileFolder);
        for (File pdfFile : PDFFile.listFiles()) {
            String filename = pdfFile.getName();
            if (!filename.contains(".pdf")) {
                continue;
            }
            List<String> numbersAndContext = DocTAET.getTableBoldNumberContext(pdfFile.getAbsolutePath());
            for (String DMLabel : evaluatedLabels) {
                for (String numberInfo : numbersAndContext) {
                    sb1.append("true" + "\t" + filename + "#" + numberInfo.split("#")[0] + "\t" + DMLabel + "\t" + numberInfo.split("#")[1]).append("\n");
                }
            }
        }
        writer1.write(sb1.toString());
        writer1.close();
    }
    
    
    //generate test tsv data for score prediction for pdf papers
    //every evaluatedLabel has the format of "dataset, metric"
    //evaluatedLabels is collected from the TDM prediction 
    public void generateTestData4ScorePrediction(String TDMTestFile, String TDMTestResultFile, String pdfFileFolder, String outputFile) throws IOException, Exception {

        BufferedReader br1 = new BufferedReader(new FileReader(TDMTestFile));
        BufferedReader br2 = new BufferedReader(new FileReader(TDMTestResultFile));

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
                    resultsPredictionsTestPapers.get(filename).add(result);
                }
            }
        }
        

        FileWriter writer1 = new FileWriter(new File(outputFile));
        StringBuffer sb1 = new StringBuffer();
        String dir_pdfFile = pdfFileFolder;
        for (Map.Entry<String, Set<NLPResult>> item : resultsPredictionsTestPapers.entrySet()) {
            String pdfFileName = item.getKey();
            String pdfPath = dir_pdfFile + "/" + pdfFileName;
            List<String> numbersAndContext = DocTAET.getTableBoldNumberContext(pdfPath);
            for (NLPResult result : item.getValue()) {
                String board = result.datasetName + ", " + result.evaluationMetric;
                for (String numberInfo : numbersAndContext) {
                    sb1.append("true" + "\t" + pdfFileName + "#" + numberInfo.split("#")[0] + "\t" + board + "\t" + numberInfo.split("#")[1]).append("\n");
                }
            }
        }
        writer1.write(sb1.toString());
        writer1.close();
    }

    public Map<String, Set<NLPResult>> getTDMPrediction(String testFile, String testResultFile) throws IOException, Exception {
        BufferedReader br1 = new BufferedReader(new FileReader(testFile));
        BufferedReader br2 = new BufferedReader(new FileReader(testResultFile));
        Map<String, Set<NLPResult>> resultsPredictions4TestPapers = new HashMap();
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
            String leaderboard = f1.get(i).split("\t")[2];
            String context = f1.get(i).split("\t")[3];
            if (leaderboard.equalsIgnoreCase("unknow")) {
                continue;
            }
            if (!resultsPredictions4TestPapers.containsKey(filename)) {
                Set<NLPResult> results = new HashSet();
                resultsPredictions4TestPapers.put(filename, results);
            }
            if (Double.valueOf(f2.get(i).split("\t")[0]) > 0.5) {
                String task = leaderboard.split(",")[0].replace(" ", "_").trim();
                String dataset = leaderboard.split(",")[1].trim();
                String eval = leaderboard.split(",")[2].trim();
                String title = context.substring(0, 50);
                NLPResult result = new NLPResult(filename, task, dataset);
                result.setEvaluationMetric(eval);
                resultsPredictions4TestPapers.get(filename).add(result);
            }
        }
        return resultsPredictions4TestPapers;
    }

    public static void main(String[] args) throws IOException, Exception {
        GenerateTestDataOnPDFPapers createTestdata = new GenerateTestDataOnPDFPapers();
        createTestdata.generateTestData4TDMPrediction("/Users/yhou/Downloads/tmp/test", "/Users/yhou/Downloads/tmp/test/test_TDM.tsv");
        createTestdata.generateTestData4ScorePrediction("/Users/yhou/Downloads/tmp/test", "/Users/yhou/Downloads/tmp/test/test_score.tsv");
    }

}
