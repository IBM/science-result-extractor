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

/**
 * Illustration of reading the Section/Table information from the pre-parsed
 * xml/json files
 *
 * @author yhou
 */
public class ArcPDNReader {

    public static void main(String[] args) throws IOException, Exception {
        GrobidPDFProcessor gp = GrobidPDFProcessor.getInstance();
        Properties prop = new Properties();
        prop.load(new FileReader("config.properties"));

        String[] types = {"P", "D", "N"};
        for (String type : types) {
            String pdfDir = prop.getProperty("projectPath") + "/data/ARC-PDN/" + type;
            File aclAntho = new File(pdfDir);

            for (File dir : aclAntho.listFiles()) {
                if (dir.isDirectory() && !dir.getName().contains("xml") && !dir.getName().contains("table")) {
                    String dirname = dir.getName();
                    String pdf_xml_Dir = dir.getParent() + "/" + dirname + "_xml";
                    String pdf_table_Dir = dir.getParent() + "/" + dirname + "_table";
                    //start dealing pdfs
                    //change propfile pdf_xml and pdf_table path for grobidPdfProcessor
                    GrobidPDFProcessor.getInstance().getProperties().setProperty("pdf_xml", "data/" + pdf_xml_Dir.split("/data/")[1]);
                    GrobidPDFProcessor.getInstance().getProperties().setProperty("pdf_table", "data/" + pdf_table_Dir.split("/data/")[1]);
                    prop.setProperty("pdf_xml", "data/" + pdf_xml_Dir.split("/data/")[1]);
                    prop.setProperty("pdf_table", "data/" + pdf_table_Dir.split("/data/")[1]);
                    for (File pdfFile : dir.listFiles()) {
                        String filename = pdfFile.getName();
                        if (!filename.contains(".pdf")) {
                            continue;
                        }
                        Map<String, String> sectionInfo = gp.getPDFSectionAndText(pdfFile.getAbsolutePath());
                        List<CachedTable> tableInfo = gp.getTableInfoFromPDF(pdfFile.getAbsolutePath());
                        System.out.println(filename + "," + sectionInfo.size() + "," + tableInfo.size());

                    }
                }
            }
        }
    }
}
