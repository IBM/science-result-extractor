/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.ibm.sre.data.corpus;

import com.ibm.sre.NLPResult;
import com.ibm.sre.data.annotation.TdmsGoldAnnotation;
import com.ibm.sre.pdfparser.CachedTable;
import com.ibm.sre.pdfparser.GrobidPDFProcessor;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import sun.swing.SwingUtilities2;

/**
 * Illustration of reading the Section/Table information from the pre-parsed
 * xml/json files, also read TDMS annotation information for each file
 *
 * @author yhou
 */
public class NlpTDMSReader {

    public static void main(String[] args) throws IOException, Exception {
        TdmsGoldAnnotation annotation = new TdmsGoldAnnotation();
        GrobidPDFProcessor gp = GrobidPDFProcessor.getInstance();
        Properties prop = new Properties();
        prop.load(new FileReader("config.properties"));
        String dir_pdfFile = prop.getProperty("projectPath") + "/" + prop.getProperty("pdfFile");
        File pdfFiles = new File(dir_pdfFile);
        for (File pdfFile : pdfFiles.listFiles()) {
            String filename = pdfFile.getName();
            if (!filename.contains(".pdf")) {
                continue;
            }
            Map<String, String> sectionInfo = gp.getPDFSectionAndText(pdfFile.getAbsolutePath());
            List<CachedTable> tableInfo = gp.getTableInfoFromPDF(pdfFile.getAbsolutePath());
            Set<NLPResult> TDMSAnnotation = annotation.getTdmsAnnotation().get(filename);
            System.out.println(filename + "," + sectionInfo.size() + "," + tableInfo.size());
        }

    }

}
