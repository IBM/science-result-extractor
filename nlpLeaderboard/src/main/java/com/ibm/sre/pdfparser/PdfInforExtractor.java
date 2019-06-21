/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.ibm.sre.pdfparser;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 *
 * 
 * This class illustrates how to use pdfparser to extract section information and table information
 * from a given PDF file. 
 * 
 * @author yhou
 */
public class PdfInforExtractor {

    public static Map<String, String> getPdfSections(String pdfFile) throws IOException, Exception {
        GrobidPDFProcessor gp = GrobidPDFProcessor.getInstance();
        Map<String, String> sections = gp.getPDFSectionAndText(pdfFile);
        return sections;
    }

    public static List<CachedTable> getTableInfo(String pdfFile) throws IOException, Exception {
        GrobidPDFProcessor gp = GrobidPDFProcessor.getInstance();
        List<CachedTable> tables = gp.getTableInfoFromPDF(pdfFile);
        return tables;
    }

    
    public static void main(String[] args) throws IOException, Exception{
        String pdfPath = "/Users/yhou/git/kbp-science/data/pdfFile/D18-1440.pdf";
        Map<String, String> sectionInfo = PdfInforExtractor.getPdfSections(pdfPath);
        List<CachedTable> tableInfo = PdfInforExtractor.getTableInfo(pdfPath);
        System.out.print("\n Section Info \n");
        for(String sectionTitle: sectionInfo.keySet()){
            System.out.println(sectionTitle + "--" + sectionInfo.get(sectionTitle));
        }

        System.out.println("\n table info\n");
        for(CachedTable table: tableInfo){
            System.out.println(table.caption);
        }
    }
    
}
