/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.ibm.sre;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.ibm.sre.pdfparser.CachedTable;
import com.ibm.sre.pdfparser.GrobidPDFProcessor;

/**
 *
 * @author yhou
 */
public class DocTAET {

    public static String getPdfTitle(String pdfFile) throws IOException, Exception {
        GrobidPDFProcessor gp = GrobidPDFProcessor.getInstance();
        Map<String, String> sections = gp.getPDFSectionAndText(pdfFile);
        return sections.get("title");
    }

    public static String getPdfAbstract(String pdfFile) throws IOException, Exception {
        GrobidPDFProcessor gp = GrobidPDFProcessor.getInstance();
        Map<String, String> sections = gp.getPDFSectionAndText(pdfFile);
        return sections.get("abstract");
    }

    public static String getPdfIntroduction(String pdfFile) throws IOException, Exception {
        GrobidPDFProcessor gp = GrobidPDFProcessor.getInstance();
        Map<String, String> sections = gp.getPDFSectionAndText(pdfFile);
        return sections.get("introduction");
    }



    public static String getPdfDatasetContext(String pdfFile) throws IOException, Exception {
        GrobidPDFProcessor gp = GrobidPDFProcessor.getInstance();
        Map<String, String> sections = gp.getPDFSectionAndText(pdfFile);
        String datasetContext = "";
        for (String section : sections.keySet()) {
            if (section.toLowerCase().matches(".*?(experiment|experiments|evaluation|evaluations|dataset|datasets).*?")) {
                datasetContext = datasetContext + " " + sections.get(section).replaceAll("\n", " ");
            }
        }
        return datasetContext;

    }

    public static String getDocTAETRepresentation(String pdfFile) throws IOException, Exception {
        String DocTAETStr = "";
        GrobidPDFProcessor gp = GrobidPDFProcessor.getInstance();
        Map<String, String> sections = gp.getPDFSectionAndText(pdfFile);
        DocTAETStr = DocTAETStr + " " + sections.get("title") + " " + sections.get("abstract") + " " + getPdfDatasetContext_Clean_150tokens(sections) 
                + " " + getTableInfo_150tokens(pdfFile);
        return DocTAETStr.replace("\n", "").trim();
    }

    //first 150 tokens of selected dataset context
    public static String getPdfDatasetContext_Clean_150tokens(String pdfFile) throws IOException, Exception {
        GrobidPDFProcessor gp = GrobidPDFProcessor.getInstance();
        Map<String, String> sections = gp.getPDFSectionAndText(pdfFile);
        return getPdfDatasetContext_Clean_150tokens(sections);
    }

    //first 150 tokens of selected dataset context
    public static String getPdfDatasetContext_Clean_150tokens(Map<String, String> sections) throws IOException, Exception {
        String datasetContext = "";
        for (String section : sections.keySet()) {
            if (section.toLowerCase().matches(".*?(experiment|experiments|evaluation|evaluations|dataset|datasets).*?")) {
                datasetContext = datasetContext + " " + sections.get(section).replaceAll("\n", " ");
            }
        }
        String datasetContext_clean = "";
        for (String sent : datasetContext.split("\\. ")) {
            if (sent.toLowerCase().matches(".*?(experiment on|experiment in|evaluation|evalualtions|evaluate|evaluated|dataset|datasets|corpus|corpora).*?")) {
                datasetContext_clean = datasetContext_clean + " " + sent;
            }
        }
        String datasetContext_clean_150 = "";
        int dsCount = 0;
        for (String s : datasetContext_clean.split(" ")) {
            if (s.isEmpty()) {
                continue;
            }
            datasetContext_clean_150 = datasetContext_clean_150 + " " + s;
            dsCount++;
            if (dsCount > 150) {
                break;
            }
        }
        datasetContext_clean = datasetContext_clean_150.trim();
        return datasetContext_clean;
    }




    public static List<String> getTableBoldNumberContext(String pdfFile) throws IOException, Exception {
        GrobidPDFProcessor gp = GrobidPDFProcessor.getInstance();
        List<CachedTable> tables = gp.getTableInfoFromPDF(pdfFile);
        List<String> numbersAndContext = new ArrayList();
        for (CachedTable table : tables) {
            String caption = "";
            for (String s : table.caption.split("\n")) {
                if (!caption.isEmpty() && s.isEmpty()) {
                    break;
                }
                if (s.startsWith("Table") || !caption.isEmpty()) {
                    caption = caption + " " + s;
                }
            }
            for (CachedTable.NumberCell boldNumber : table.getBoldedNumberCells()) {
                String numberContext = "";
                for (String s : boldNumber.associatedMergedColumns) {
                    numberContext = numberContext + " " + s;
                }
                for (String s : boldNumber.associatedColumns) {
                    numberContext = numberContext + " " + s;
                }
                numberContext = numberContext + " " + caption;
                numbersAndContext.add(boldNumber.number + "#" + numberContext.trim());
            }
        }
        return numbersAndContext;
    }

    public static List<String> getTableNumberContext(String pdfFile) throws IOException, Exception {
        GrobidPDFProcessor gp = GrobidPDFProcessor.getInstance();
        List<CachedTable> tables = gp.getTableInfoFromPDF(pdfFile);
        List<String> numbersAndContext = new ArrayList();
        for (CachedTable table : tables) {
            String caption = "";
            for (String s : table.caption.split("\n")) {
                if (!caption.isEmpty() && s.isEmpty()) {
                    break;
                }
                if (s.startsWith("Table") || !caption.isEmpty()) {
                    caption = caption + " " + s;
                }
            }
            if (table.columns.size() == 0) {
                System.err.println(pdfFile + ":" + "0 columns");
            }
            for (CachedTable.NumberCell boldNumber : table.numberCells) {
                String numberContext = "";
                for (String s : boldNumber.associatedMergedColumns) {
                    numberContext = numberContext + " " + s;
                }
                for (String s : boldNumber.associatedColumns) {
                    numberContext = numberContext + " " + s;
                }
                numberContext = numberContext + " " + caption;
                numbersAndContext.add(boldNumber.number + "#" + numberContext.trim());
            }
        }
        return numbersAndContext;
    }

    public static String getTableInfo_150tokens(String pdfFile) throws IOException, Exception {
        GrobidPDFProcessor gp = GrobidPDFProcessor.getInstance();
        List<CachedTable> tables = gp.getTableInfoFromPDF(pdfFile);
        String tableContext = "";
        for (CachedTable table : tables) {
            String caption = "";
            for (String s : table.caption.split("\n")) {
                if (!caption.isEmpty() && s.isEmpty()) {
                    break;
                }
                if (s.startsWith("Table") || !caption.isEmpty()) {
                    caption = caption + " " + s;
                }
            }

            tableContext = tableContext + " " + caption;
            for (String mergeCol : table.mergedAllColumns) {
                tableContext = tableContext + " " + mergeCol;
            }
            for (String column : table.columns) {
                tableContext = tableContext + " " + column;
            }
        }

        String tableContext_150 = "";
        int tableCount = 0;
        for (String s : tableContext.split(" ")) {
            if (s.isEmpty()) {
                continue;
            }
            tableContext_150 = tableContext_150 + " " + s;
            tableCount++;
            if (tableCount > 150) {
                break;
            }
        }
        tableContext = tableContext_150.trim();
        return tableContext;
    }

    
    
    public static String getTableInfo(String pdfFile) throws IOException, Exception {
        GrobidPDFProcessor gp = GrobidPDFProcessor.getInstance();
        List<CachedTable> tables = gp.getTableInfoFromPDF(pdfFile);
        String tableContext = "";
        for (CachedTable table : tables) {
            String caption = "";
            for (String s : table.caption.split("\n")) {
                if (!caption.isEmpty() && s.isEmpty()) {
                    break;
                }
                if (s.startsWith("Table") || !caption.isEmpty()) {
                    caption = caption + " " + s;
                }
            }

            tableContext = tableContext + " " + caption;
            for (String mergeCol : table.mergedAllColumns) {
                tableContext = tableContext + " " + mergeCol;
            }
            for (String column : table.columns) {
                tableContext = tableContext + " " + column;
            }
        }
        return tableContext;
    }
    
    
    public static List<String> getTableInfoAsList(String pdfFile) throws IOException, Exception {
        GrobidPDFProcessor gp = GrobidPDFProcessor.getInstance();
        List<CachedTable> tables = gp.getTableInfoFromPDF(pdfFile);
        List<String > tableContext = new ArrayList();
        for (CachedTable table : tables) {
            String caption = "";
            for (String s : table.caption.split("\n")) {
                if (!caption.isEmpty() && s.isEmpty()) {
                    break;
                }
                if (s.startsWith("Table") || !caption.isEmpty()) {
                    caption = caption + " " + s;
                }
            }
            tableContext.add(caption);
            for (String mergeCol : table.mergedAllColumns) {
                tableContext.add("table merged column " + mergeCol);
            }
            for (String column : table.columns) {
                tableContext.add("table column " + column);
            }
            for (String row : table.rows) {
                tableContext.add("table row " + row);
            }
        }
        return tableContext;
    }
    

}
