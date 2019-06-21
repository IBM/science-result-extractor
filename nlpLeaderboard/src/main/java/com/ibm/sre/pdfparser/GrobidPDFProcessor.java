/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.ibm.sre.pdfparser;

import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.PeekingIterator;
import com.google.common.collect.Sets;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.ibm.sre.BaseDirInfo;
import com.ibm.sre.pdfparser.CachedTable.NumberCell;
import com.ibm.sre.pdfparser.NLPLeaderboardTable.TableCell;
import com.rits.cloning.Cloner;
import com.sun.javafx.font.PGFont;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.io.Writer;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.xpath.XPath;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathFactory;
import org.apache.commons.lang3.StringUtils;
import org.grobid.core.GrobidModels;
import org.grobid.core.data.BiblioItem;
import org.grobid.core.data.Equation;
import org.grobid.core.data.Figure;
import org.grobid.core.data.Table;
import org.grobid.core.document.DocumentPiece;
import org.grobid.core.engines.Engine;
import org.grobid.core.engines.EngineParsers;
import static org.grobid.core.engines.FullTextParser.getBodyTextFeatured;
import org.grobid.core.engines.config.GrobidAnalysisConfig;
import org.grobid.core.engines.counters.TableRejectionCounters;
import org.grobid.core.engines.label.SegmentationLabels;
import org.grobid.core.engines.label.TaggingLabels;
import org.grobid.core.factory.GrobidFactory;
import org.grobid.core.layout.Block;
import org.grobid.core.layout.BoundingBox;
import org.grobid.core.layout.LayoutToken;
import org.grobid.core.layout.LayoutTokenization;
import org.grobid.core.main.GrobidHomeFinder;
import org.grobid.core.tokenization.LabeledTokensContainer;
import org.grobid.core.tokenization.TaggingTokenSynchronizer;
import org.grobid.core.utilities.BoundingBoxCalculator;
import org.grobid.core.utilities.GrobidProperties;
import org.grobid.core.utilities.LayoutTokensUtil;
import static org.grobid.core.utilities.LayoutTokensUtil.toText;
import org.grobid.core.utilities.Pair;
import org.grobid.core.utilities.crossref.CrossrefClient;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

/**
 *
 * @author yhou
 */
public class GrobidPDFProcessor {

    private Properties prop;
    private String grobidHome;
    private String grobidProperties;
    private static GrobidPDFProcessor instance = null;
    Engine engine;
    EngineParsers parsers;
    Gson gson;
    Cloner cloner;


    public static GrobidPDFProcessor getInstance() throws IOException, Exception {
        if (instance == null) {
            instance = new GrobidPDFProcessor();
        }
        return instance;
    }

    public Properties getProperties() {
        return prop;
    }

    private GrobidPDFProcessor() throws IOException, Exception {
        prop = new Properties();
        prop.load(new FileReader("config.properties"));
        grobidHome = prop.getProperty("pGrobidHome");
        grobidProperties = prop.getProperty("pGrobidProperties");
        GrobidHomeFinder grobidHomeFinder = new GrobidHomeFinder(Arrays.asList(grobidHome));
        GrobidProperties.getInstance(grobidHomeFinder);
//        System.out.println(">>>>>>>> GROBID_HOME="+GrobidProperties.get_GROBID_HOME_PATH());
        engine = GrobidFactory.getInstance().createEngine();
        parsers = new EngineParsers();
        gson = new Gson();
        cloner = new Cloner();
    }

    private String correctMisSpelling(String input, Set dic) {
        String val = "";
        boolean combinelastToken = false;
        for (int i = 0; i < input.split("\\s").length - 1; i++) {
            String cur = input.split("\\s")[i];
            String next = input.split("\\s")[i + 1];
            String combine = cur + next;
            if (cur.matches("^[a-zA-Z]+") && next.matches("^[a-zA-Z]+") && dic.contains(combine)) {
                val = val + " " + combine;
                i = i + 1;
                if (i == input.split("\\s").length - 1) {
                    combinelastToken = true;
                }
            } else if (cur.matches("^[a-zA-Z]+") && next.matches("^[a-zA-Z]+[\\.\\,\\;]")) {
                String combine1 = cur + next.substring(0, next.length() - 1);
                if (dic.contains(combine1)) {
                    val = val + " " + combine1 + next.substring(next.length() - 1);
                    i = i + 1;
                    if (i == input.split("\\s").length - 1) {
                        combinelastToken = true;
                    }
                } else {
                    val = val + " " + cur;
                }
            } else {
                val = val + " " + cur;
            }
        }
        if (!combinelastToken) {
            val = val + " " + input.split("\\s")[input.split("\\s").length - 1];
        }
        return val.trim();
    }

    //filePath store xml files extracted from pdf using grobid
    //java -Xmx4G -jar grobid-core/build/libs/grobid-core-0.5.4-SNAPSHOT-onejar.jar -gH grobid-home -dIn /Users/yhou/git/kbp-science/data/pdfFile -dOut /Users/yhou/git/kbp-science/data/pdfFile_txt -exe processFullText
    //this method extract cleantext from xml files
    private void extractTxtFromXml(String filePath, String outputPath) throws IOException, Exception {
        //load english dictionary
        String dicPath = BaseDirInfo.getBaseDir() + "resources/en_US.dic";
        File file = new File(dicPath);
        BufferedReader br = new BufferedReader(new FileReader(file));
        String st;
        Set<String> dic = new HashSet();
        while ((st = br.readLine()) != null) {
            dic.add(st.trim().split("/")[0]);
        }

        //
        File output = new File(outputPath);
        if (!output.exists()) {
            output.mkdir();
        }

        DocumentBuilderFactory docBuilderFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder docBuilder = docBuilderFactory.newDocumentBuilder();
        XPathFactory xpathFactory = XPathFactory.newInstance();
        XPath xpath = xpathFactory.newXPath();
        Set<String> xmlFiles = new HashSet();

        File folder = new File(filePath);
        for (File s : folder.listFiles()) {
            if (s.isDirectory()) {
                continue;
            }
            StringBuffer sb = new StringBuffer();
            String fileName = s.getName().replace(".tei.xml", "");
            xmlFiles.add(fileName);
            Document doc = docBuilder.parse(s);
            doc.getDocumentElement().normalize();
            sb.append("section: title").append("\n");
            NodeList nList0 = doc.getElementsByTagName("titleStmt");
            for (int i = 0; i < nList0.getLength(); i++) {
                Node nNode = nList0.item(i);
                if (nNode.getNodeType() == Node.ELEMENT_NODE) {
                    Element eElement = (Element) nNode;
                    NodeList nList1 = ((Element) eElement).getElementsByTagName("title");
                    for (int j = 0; j < nList1.getLength(); j++) {
                        Node paragraph = nList1.item(j);
                        sb.append(correctMisSpelling(paragraph.getTextContent(), dic)).append("\n");
                    }
                }
            }
            NodeList nList = doc.getElementsByTagName("abstract");
            sb.append("section: abstract").append("\n");
            for (int i = 0; i < nList.getLength(); i++) {
                Node nNode = nList.item(i);
                if (nNode.getNodeType() == Node.ELEMENT_NODE) {
                    Element eElement = (Element) nNode;
                    NodeList nList1 = ((Element) eElement).getElementsByTagName("p");
                    for (int j = 0; j < nList1.getLength(); j++) {
                        Node paragraph = nList1.item(j);
                        sb.append(correctMisSpelling(paragraph.getTextContent(), dic)).append("\n");
                    }
                }
            }
            Element bodyelement = (Element) xpath.evaluate("//TEI/text/body", doc, XPathConstants.NODE);
            NodeList nList1 = bodyelement.getElementsByTagName("div");
            for (int i = 0; i < nList1.getLength(); i++) {
                Node nNode = nList1.item(i);
                if (nNode.getNodeType() == Node.ELEMENT_NODE) {
                    Element eElement = (Element) nNode;
                    Node head = (Element) xpath.evaluate("head", eElement, XPathConstants.NODE);
                    sb.append("section: " + head.getTextContent()).append("\n");
                    NodeList nList2 = ((Element) eElement).getElementsByTagName("p");
                    for (int j = 0; j < nList2.getLength(); j++) {
                        Node paragraph = nList2.item(j);
                        String para = "";
                        for (int k = 0; k < paragraph.getChildNodes().getLength(); k++) {
                            if (paragraph.getChildNodes().item(k).getNodeType() == Node.TEXT_NODE) {
                                para = para.trim() + paragraph.getChildNodes().item(k).getTextContent();
                            }
                        }
                        sb.append(correctMisSpelling(para.trim(), dic)).append("\n");
                    }
                }
            }
            FileWriter out = new FileWriter(outputPath + "/" + fileName + ".txt");
            out.write(sb.toString());
            out.close();
        }
    }

    public String getPDFTitle(String pdfPath) throws IOException, Exception {
        BiblioItem resHeader = new BiblioItem();
        String tei = engine.processHeader(pdfPath, 1, resHeader);
        return resHeader.getTitle();
    }

    public String getPDFAbstract(String pdfPath) throws IOException, Exception {
        BiblioItem resHeader = new BiblioItem();
        String tei = engine.processHeader(pdfPath, 1, resHeader);
        return resHeader.getAbstract();
    }

    //title, abstract, sections
    public Map<String, String> getPDFSectionAndText(String pdfPath) throws IOException, Exception {
        String dicPath = BaseDirInfo.getBaseDir() + "resources/en_US.dic";
        File file = new File(dicPath);
        BufferedReader br = new BufferedReader(new FileReader(file));
        String st;
        Set<String> dic = new HashSet();
        while ((st = br.readLine()) != null) {
            dic.add(st.trim().split("/")[0]);
        }

        String xml = "";
        String pdf_xml_dir = prop.getProperty("projectPath") + "/" + prop.getProperty("pdf_xml");
        String pdfxmlName = pdfPath.split("/")[pdfPath.split("/").length - 1].replace(".pdf", ".tei.xml");
        String pdfxmlName1 = pdfPath.split("/")[pdfPath.split("/").length - 1].replace(".pdf", ".xml");
        File xmlfile = new File(pdf_xml_dir + "/" + pdfxmlName);
        if(!xmlfile.exists())
            xmlfile = new File(pdf_xml_dir + "/" + pdfxmlName1);
        boolean preParsedXMLFileExist = false;
        if (xmlfile.exists()) {
            preParsedXMLFileExist = true;
        } else {
            GrobidAnalysisConfig config = GrobidAnalysisConfig.defaultInstance();
            xml = engine.fullTextToTEI(new File(pdfPath), config);
        }

        Map<String, String> textwithsection = new LinkedHashMap();
        DocumentBuilderFactory docBuilderFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder docBuilder = docBuilderFactory.newDocumentBuilder();
        XPathFactory xpathFactory = XPathFactory.newInstance();
        XPath xpath = xpathFactory.newXPath();
        Document doc = null;
        if (preParsedXMLFileExist) {
            doc = docBuilder.parse(xmlfile);
        } else {
            InputStream in = org.apache.commons.io.IOUtils.toInputStream(xml, "UTF-8");
            doc = docBuilder.parse(in);
        }
        doc.getDocumentElement().normalize();
        NodeList nList0 = doc.getElementsByTagName("titleStmt");
        String title = "";
        for (int i = 0; i < nList0.getLength(); i++) {
            Node nNode = nList0.item(i);
            if (nNode.getNodeType() == Node.ELEMENT_NODE) {
                Element eElement = (Element) nNode;
                NodeList nList1 = ((Element) eElement).getElementsByTagName("title");
                for (int j = 0; j < nList1.getLength(); j++) {
                    Node paragraph = nList1.item(j);
                    title = title + " " + correctMisSpelling(paragraph.getTextContent(), dic);
                }
            }
        }
        textwithsection.put("title", title.trim());
        String paperabstract = "";
        NodeList nList = doc.getElementsByTagName("abstract");
        for (int i = 0; i < nList.getLength(); i++) {
            Node nNode = nList.item(i);
            if (nNode.getNodeType() == Node.ELEMENT_NODE) {
                Element eElement = (Element) nNode;
                NodeList nList1 = ((Element) eElement).getElementsByTagName("p");
                for (int j = 0; j < nList1.getLength(); j++) {
                    Node paragraph = nList1.item(j);
                    paperabstract = paperabstract + correctMisSpelling(paragraph.getTextContent(), dic) + "\n";
                }
            }
        }
        textwithsection.put("abstract", paperabstract.trim());

        Element bodyelement = (Element) xpath.evaluate("//TEI/text/body", doc, XPathConstants.NODE);
        NodeList nList1 = bodyelement.getElementsByTagName("div");
        for (int i = 0; i < nList1.getLength(); i++) {
            Node nNode = nList1.item(i);
            if (nNode.getNodeType() == Node.ELEMENT_NODE) {
                Element eElement = (Element) nNode;
                Node head = (Element) xpath.evaluate("head", eElement, XPathConstants.NODE);
                String sectiontitle = head.getTextContent();
                String sectionParagraphs = "";
                NodeList nList2 = ((Element) eElement).getElementsByTagName("p");
                for (int j = 0; j < nList2.getLength(); j++) {
                    Node paragraph = nList2.item(j);
                    String para = "";
                    for (int k = 0; k < paragraph.getChildNodes().getLength(); k++) {
                        if (paragraph.getChildNodes().item(k).getNodeType() == Node.TEXT_NODE) {
                            para = para.trim() + paragraph.getChildNodes().item(k).getTextContent();
                        }
                    }
                    sectionParagraphs = sectionParagraphs + correctMisSpelling(para.trim(), dic) + "\n";
                }
                textwithsection.put(sectiontitle.toLowerCase(), sectionParagraphs);

            }
        }
        return textwithsection;
    }

    public Map<String, String> getPDFTablesFromXML(String pdfPath) throws IOException, Exception {
        String dicPath = BaseDirInfo.getBaseDir() + "resources/en_US.dic";
        File file = new File(dicPath);
        BufferedReader br = new BufferedReader(new FileReader(file));
        String st;
        Set<String> dic = new HashSet();
        while ((st = br.readLine()) != null) {
            dic.add(st.trim().split("/")[0]);
        }

        String xml = "";
        String pdf_xml_dir = prop.getProperty("projectPath") + "/" + prop.getProperty("pdf_xml");
        String pdfxmlName = pdfPath.split("/")[pdfPath.split("/").length - 1].replace(".pdf", ".tei.xml");
        File xmlfile = new File(pdf_xml_dir + "/" + pdfxmlName);
        boolean preParsedXMLFileExist = false;
        if (xmlfile.exists()) {
            preParsedXMLFileExist = true;
        } else {
            GrobidAnalysisConfig config = GrobidAnalysisConfig.defaultInstance();
            xml = engine.fullTextToTEI(new File(pdfPath), config);
        }

        Map<String, String> textwithsection = new LinkedHashMap();
        DocumentBuilderFactory docBuilderFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder docBuilder = docBuilderFactory.newDocumentBuilder();
        XPathFactory xpathFactory = XPathFactory.newInstance();
        XPath xpath = xpathFactory.newXPath();
        Document doc = null;
        if (preParsedXMLFileExist) {
            doc = docBuilder.parse(xmlfile);
        } else {
            InputStream in = org.apache.commons.io.IOUtils.toInputStream(xml, "UTF-8");
            doc = docBuilder.parse(in);
        }
        doc.getDocumentElement().normalize();

        Element bodyelement = (Element) xpath.evaluate("//TEI/text/body", doc, XPathConstants.NODE);
        NodeList nList1 = bodyelement.getElementsByTagName("figure");
        for (int i = 0; i < nList1.getLength(); i++) {
            Node nNode = nList1.item(i);
            if (nNode.getNodeType() == Node.ELEMENT_NODE) {
                Element eElement = (Element) nNode;
                Node head = (Element) xpath.evaluate("head", eElement, XPathConstants.NODE);
                String sectiontitle = head.getTextContent();
                String sectionParagraphs = "";
                NodeList nList2 = ((Element) eElement).getElementsByTagName("table");
                for (int j = 0; j < nList2.getLength(); j++) {
                    Node paragraph = nList2.item(j);
                    String para = "";
                    for (int k = 0; k < paragraph.getChildNodes().getLength(); k++) {
                        if (paragraph.getChildNodes().item(k).getNodeType() == Node.TEXT_NODE) {
                            para = para.trim() + paragraph.getChildNodes().item(k).getTextContent();
                        }
                    }
                    sectionParagraphs = sectionParagraphs + correctMisSpelling(para.trim(), dic) + "\n";
                }
                textwithsection.put(sectiontitle.toLowerCase(), sectionParagraphs);

            }
        }

        return textwithsection;
    }

    public List<Table> getTables(String pdfPath) throws IOException, Exception {
        GrobidAnalysisConfig config = GrobidAnalysisConfig.defaultInstance();
        org.grobid.core.document.Document doc = engine.fullTextToTEIDoc(new File(pdfPath), config);
        return doc.getTables();
    }

    public List<NLPLeaderboardTable> getCleanedTables(String pdfPath) {
        List<NLPLeaderboardTable> cleanedTable = new ArrayList();
        GrobidAnalysisConfig config = GrobidAnalysisConfig.defaultInstance();
        try {
            org.grobid.core.document.Document doc = engine.fullTextToTEIDoc(new File(pdfPath), config);
            SortedSet<DocumentPiece> documentBodyParts = doc.getDocumentPart(SegmentationLabels.BODY);
            // full text processing
            Pair<String, LayoutTokenization> featSeg = getBodyTextFeatured(doc, documentBodyParts);
            List<Table> tables = null;
            List<Figure> figs = null;
            String rese = null;
            LayoutTokenization layoutTokenization = null;
            if (featSeg != null) {
                // if featSeg is null, it usually means that no body segment is found in the
                // document segmentation
                String bodytext = featSeg.getA();
                layoutTokenization = featSeg.getB();
                if ((bodytext != null) && (bodytext.trim().length() > 0)) {
                    rese = parsers.getFullTextParser().label(bodytext);
                }
                // we apply now the figure and table models based on the fulltext labeled output
                tables = processTables(rese, layoutTokenization.getTokenization());
                figs = processFigures(rese, layoutTokenization.getTokenization());
//debug
//            System.err.println("process tables");
//            for (Table t : tables) {
//                System.err.println("tableAll:" + LayoutTokensUtil.normalizeText(t.getLayoutTokens()));
//                System.err.println("------");
//            }
//            System.err.println("process figures");
//            for (Figure fig : figs) {
////                System.err.println("figueAll:" + LayoutTokensUtil.normalizeText(fig.getLayoutTokens()));
//                System.err.println("figueAll:" + normalizeText(fig.getLayoutTokens()));
//                System.err.println("------");
//            }
            }

            //first round, collect table caption
            for (Table t1 : tables) {
                String tableStr = LayoutTokensUtil.normalizeText(t1.getLayoutTokens());
                if (tableStr.matches(".*?(Table|TABLE) \\d+(:|\\.| [A-Z]).*?")) {
                    NLPLeaderboardTable newTable = new NLPLeaderboardTable();
                    newTable.appendCaptionLayoutTokens(t1.getLayoutTokens());
                    cleanedTable.add(newTable);
                }
            }
            //second round, paired each caption with possible content, some table content could come from the figures 
            for (NLPLeaderboardTable t1 : cleanedTable) {
                for (Table t2 : tables) {
                    //if t2 contains multiple numbers, then t2 is likely to be a table content (although sometimes it mixes with the wrong table caption)
                    //if t2 is close to t1, then associate t2 as a content candidate to t1
                    String tableStr = LayoutTokensUtil.normalizeText(t2.getLayoutTokens());
                    if ((countNumbersInAText(tableStr) > 5 && hasConsecutiveNumbers(tableStr)) || twoColumnTableContent(normalizeText(t2.getLayoutTokens()))) {
                        BoundingBox captionBox = BoundingBoxCalculator.calculateOneBox(t1.getCaptionLayoutTokens(), true);
                        BoundingBox contentBox = BoundingBoxCalculator.calculateOneBox(t2.getLayoutTokens(), true);
                        if (captionBox.getPage() == contentBox.getPage() && captionBox.distanceTo(contentBox) < 50 && captionAlignVerticalWithContent(captionBox, contentBox)) {
                            t1.getContentTokens().add(t2.getLayoutTokens());
                        }
                    }
                }
                for (Figure f2 : figs) {
                    String figureStr = LayoutTokensUtil.normalizeText(f2.getLayoutTokens());
                    if ((countNumbersInAText(figureStr) > 5 && hasConsecutiveNumbers(figureStr)) || twoColumnTableContent(normalizeText(f2.getLayoutTokens()))) {
                        BoundingBox captionBox = BoundingBoxCalculator.calculateOneBox(t1.getCaptionLayoutTokens(), true);
                        BoundingBox contentBox = BoundingBoxCalculator.calculateOneBox(f2.getLayoutTokens(), true);
                        if (captionBox.getPage() == contentBox.getPage() && captionBox.distanceTo(contentBox) < 50 && captionAlignVerticalWithContent(captionBox, contentBox)) {
                            t1.getContentTokens().add(f2.getLayoutTokens());
                        }
                    }
                }
            }
            //third round, if a table got two content associating with a caption, while one content is already associated with another table and that table only has one content
            //then remove this content from the first table
            List<List<LayoutToken>> contentPairedWithOneCaption = new ArrayList();
            for (NLPLeaderboardTable t1 : cleanedTable) {
                //first, collect contents which paird with only one table caption
                if (t1.getContentTokens().size() == 1) {
                    contentPairedWithOneCaption.add(t1.getContentTokens().get(0));
                    t1.getContentTokens_Clean().addAll(t1.getContentTokens().get(0));
                }
            }
            for (NLPLeaderboardTable t1 : cleanedTable) {
                if (t1.getContentTokens().size() > 1) {
                    for (List<LayoutToken> content : t1.getContentTokens()) {
                        if (!contentPairedWithOneCaption.contains(content)) {
                            t1.getContentTokens_Clean().addAll(content);
                        }
                    }
                }
            }
            
            //fourth round, if a table got both caption and table content, but the table content doesn't have column info
            //add the last text line above the table content as the table content, this is because column information
            //is important to extract results in leaderboards
            for(NLPLeaderboardTable t1: cleanedTable){
                if(t1.getCaptionLayoutTokens().size()>0 && t1.getContentTokens_Clean().size()>0){
                NLPLeaderboardTable t1Clone = cloner.deepClone(t1);
                CachedTable ctable = new CachedTable(normalizeText(t1Clone.getCaptionLayoutTokens()));
                List<TableCell> numberCells = t1Clone.getNumberCells();
                for (TableCell c1 : numberCells) {
                    NumberCell numberCell = ctable.new NumberCell(normalizeText(c1.lt));
                    numberCell.isBolded = c1.isBold;
                    for (String s : c1.getAssociatedTagsStr_row()) {
                        numberCell.associatedRows.add(s);
                        ctable.rows.add(s);
                    }
                    for (String s : c1.getAssociatedTagsStr_column_mergedAll()) {
                        numberCell.associatedMergedColumns.add(s);
                        ctable.mergedAllColumns.add(s);
                    }
                    for (String s : c1.getAssociatedTagsStr_column()) {
                        numberCell.associatedColumns.add(s);
                        ctable.columns.add(s);
                    }
                    ctable.numberCells.add(numberCell);
                }
//                if (LayoutTokensUtil.toText(t1.getContentTokens_Clean()).matches
//                        ("(\\d+\\.\\d+|\\d+\\.\\d+\\%|\\.\\d+|.\\d+\\.\\d+|\\d+\\.\\d+.|.\\.\\d+|\\.\\d+.|.\\d+\\.\\d+\\%|\\d+\\.\\d+\\%)[\\s\\S]*?")) { 
                if(ctable.columns.size()==0){
                        LayoutToken currentFirstToken = t1.getContentTokens_Clean().get(0);
                        BoundingBox currentBox = BoundingBoxCalculator.calculateOneBox(t1.getContentTokens_Clean(), true);
                        Map<Double, List<LayoutToken>> lines = new TreeMap();
                        double previousYPosition = 0;
                        for(LayoutToken lt: layoutTokenization.getTokenization()){
                            if(lt.getPage()<currentFirstToken.getPage()) continue;
                            if(lt.getPage()>currentFirstToken.getPage()) break;
                            
                            if(lines.containsKey(lt.getY())){
                                lines.get(lt.getY()).add(lt);
                                previousYPosition = lt.getY();
                            }else if(lt.getY()<0){
                                lines.get(previousYPosition).add(lt);
                            }else{
                                List<LayoutToken> lt_list = new ArrayList();
                                lt_list.add(lt);
                                lines.put(lt.getY(), lt_list);
                                previousYPosition = lt.getY();
                            }
                        }
                        List<LayoutToken> missingColumn = new ArrayList();
                        for(double yposition: lines.keySet()){
                            if(yposition<0) continue;
                            BoundingBox lineBox = BoundingBoxCalculator.calculateOneBox(lines.get(yposition), true);
                            if(yposition<currentFirstToken.getY()&&tagVerticallyAlignedWithNumber(lineBox, currentBox)){
//                                System.err.println("missing column:" + LayoutTokensUtil.toText(lines.get(yposition)));
                                missingColumn = lines.get(yposition);
                            }
                        }
                        //generate new ContentTokens_Clean() for t1
                        missingColumn.addAll(t1.getContentTokens_Clean());
                        t1.setContentTokens_Clean(missingColumn);
                        
                    }
                }
            }

            
//debug info
//        System.err.println("process tables");
//        for (NLPLeaderboardTable t : cleanedTable) {
//                System.err.println("tableTitle:" + LayoutTokensUtil.normalizeText(t.getLayoutTokens()));
//                System.err.println("tableTitle:" + normalizeText(t.getCaptionLayoutTokens()));
//                System.err.println("tablecontent:" + normalizeText(t.getContentTokens_Clean()));
//                System.err.println("bolded number cells info");
//                getBoldedNumberCellsInfo(t);
//                System.err.println("------");
//            }
        } catch (Exception e) {
            System.err.println("can't parse pdf");
        }

        List<NLPLeaderboardTable> cleanedVerifiedTables = new ArrayList();
        for (NLPLeaderboardTable table : cleanedTable) {
            if (table.getCaptionLayoutTokens().size() > 0 && table.getContentTokens_Clean().size() > 0) {
                cleanedVerifiedTables.add(table);
            }
        }
        return cleanedVerifiedTables;
    }

    //find the missing columns for the table content, tag is a line of layout token and number is the whold table content (without column)
    private boolean tagVerticallyAlignedWithNumber(BoundingBox tag, BoundingBox number) {
        boolean align = false;
        //sometimes, there's a litter overlap between the tag box and the numberbox in y-axis
        if (tag.getY2() >= number.getY()&&(tag.getY2()-number.getY())>1) {
            return false;
        }
        if (tag.getX() >= number.getX() && tag.getX2() <= number.getX2()) {
            align = true;
        }
        else if (tag.getX() <= number.getX() && tag.getX2() >= number.getX2()) {
            align = true;
        }
        else if (tag.getX() >= number.getX() && tag.getX2() >= number.getX2() && number.getX2() > tag.getX()) {
            double overlap = number.getX2() - tag.getX();
            if (overlap / number.getWidth() > 0.5) {
                align = true;
            } else {
                align = false;
            }
        }
        else if (tag.getX() <= number.getX() && tag.getX2() <= number.getX2() && tag.getX2() > number.getX()) {
            double overlap = tag.getX2() - number.getX();
            if (overlap / number.getWidth() > 0.5) {
                align = true;
            } else {
                align = false;
            }
        }else {
           align = false;
        }
        return align;
    }    
    
    
    
    public void getBoldedNumberCellsInfo(NLPLeaderboardTable table) {
        List<TableCell> cells = table.getBoldedNumberCells();
        for (TableCell c1 : cells) {
            System.err.println("text:" + normalizeText(c1.lt));
            System.err.println("associated row:");
            for (TableCell row : c1.getAssociatedTags_row()) {
                System.err.println(normalizeText(row.lt));
            }
            System.err.println("associated merged column:");
            for (TableCell col : c1.getAssociatedTags_column_mergedAll()) {
                System.err.println(normalizeText(col.lt));
            }
            System.err.println("associated column:");
            for (TableCell col : c1.getAssociatedTags_column()) {
                System.err.println(normalizeText(col.lt));
            }
        }
    }

    public List<TableCell> getBoldedNumberCells(NLPLeaderboardTable table) {
        return table.getBoldedNumberCells();
    }

    
    public void writeFullTextTEIXml(String pdfStr, String xmlFileStr) throws IOException, Exception {
        String xml = getFullTextToTEI(pdfStr);
        FileWriter writer = new FileWriter(new File(xmlFileStr));
        writer.write(xml);
        writer.close();
    }
    
    
    public void writeTableInfoInJson(String pdfStr, String jsonFileStr) throws IOException, Exception {
        List<NLPLeaderboardTable> tables = this.getCleanedTables(pdfStr);
        List<CachedTable> cachedTables = new ArrayList();
        Type type = new TypeToken<List<CachedTable>>() {
        }.getType();
        for (NLPLeaderboardTable table : tables) {
            CachedTable ctable = new CachedTable(normalizeText(table.getCaptionLayoutTokens()));
            List<TableCell> numberCells = table.getNumberCells();
            for (TableCell c1 : numberCells) {
                NumberCell numberCell = ctable.new NumberCell(normalizeText(c1.lt));
                numberCell.isBolded = c1.isBold;
                for (String s : c1.getAssociatedTagsStr_row()) {
                    numberCell.associatedRows.add(s);
                    ctable.rows.add(s);
                }
                for (String s : c1.getAssociatedTagsStr_column_mergedAll()) {
                    numberCell.associatedMergedColumns.add(s);
                    ctable.mergedAllColumns.add(s);
                }
                for (String s : c1.getAssociatedTagsStr_column()) {
                    numberCell.associatedColumns.add(s);
                    ctable.columns.add(s);
                }
                ctable.numberCells.add(numberCell);
            }
            cachedTables.add(ctable);
        }
//        String json = gson.toJson(cachedTables, type);
        OutputStream outputStream = new FileOutputStream(new File(jsonFileStr));
        Writer writer = new BufferedWriter(new OutputStreamWriter(outputStream));
        gson.toJson(cachedTables, type, writer);
        writer.close();
    }

    public List<CachedTable> readTablesFromCashedJson(String jsonfile) throws IOException, Exception {
        Type type = new TypeToken<List<CachedTable>>() {
        }.getType();
        InputStream inputStream = new FileInputStream(new File(jsonfile));
        Reader reader = new BufferedReader(new InputStreamReader(inputStream));
        List<CachedTable> tables = gson.fromJson(reader, type);
        return tables;

    }

    public List<CachedTable> getTableInfoFromPDF(String pdfPath) throws IOException, Exception {
        List<CachedTable> tables = new ArrayList();
        String pdf_table_dir = prop.getProperty("projectPath") + "/" + prop.getProperty("pdf_table");
        String pdftableName = pdfPath.split("/")[pdfPath.split("/").length - 1].replace(".pdf", ".json");
        File tableJsonfile = new File(pdf_table_dir + "/" + pdftableName);
        boolean preParsedTableFileExist = false;
        if (tableJsonfile.exists()) {
            preParsedTableFileExist = true;
            tables = readTablesFromCashedJson(pdf_table_dir + "/" + pdftableName);
        } else {
            List<NLPLeaderboardTable> complextables = this.getCleanedTables(pdfPath);
            Type type = new TypeToken<List<CachedTable>>() {
            }.getType();
            for (NLPLeaderboardTable table : complextables) {
                CachedTable ctable = new CachedTable(normalizeText(table.getCaptionLayoutTokens()));
                List<TableCell> numberCells = table.getNumberCells();
                for (TableCell c1 : numberCells) {
                    NumberCell numberCell = ctable.new NumberCell(normalizeText(c1.lt));
                    numberCell.isBolded = c1.isBold;
                    for (String s : c1.getAssociatedTagsStr_row()) {
                        numberCell.associatedRows.add(s);
                        ctable.rows.add(s);
                    }
                    for (String s : c1.getAssociatedTagsStr_column_mergedAll()) {
                        numberCell.associatedMergedColumns.add(s);
                        ctable.mergedAllColumns.add(s);
                    }
                    for (String s : c1.getAssociatedTagsStr_column()) {
                        numberCell.associatedColumns.add(s);
                        ctable.columns.add(s);
                    }
                    ctable.numberCells.add(numberCell);
                }
                tables.add(ctable);
            }

        }

        return tables;
    }
    
    
    public void getBoldedNumberCellsInfo(String pdfStr) throws IOException, Exception {
        List<NLPLeaderboardTable> tables = this.getCleanedTables(pdfStr);

        for (NLPLeaderboardTable table : tables) {
            System.out.println("-------------------");
            System.out.println("table caption:" + normalizeText(table.getCaptionLayoutTokens()));
            System.out.println("bolded number info:");
            List<TableCell> numberCells = table.getBoldedNumberCells();
            for (TableCell c1 : numberCells) {
                System.out.println("number:" + normalizeText(c1.lt));
                for (String s : c1.getAssociatedTagsStr_row()) {
                    System.out.println("associated row:" + s);
                }
                for (String s : c1.getAssociatedTagsStr_column_mergedAll()) {
                    System.out.println("associated merged column:" + s);
                }
                for (String s : c1.getAssociatedTagsStr_column()) {
                    System.out.println("associated column:" + s);
                }
            }
        }
    }
    
    public String getCleanTableCaption(String originalCap) {
        String caption = "";
            for (String s : originalCap.split("\n")) {
                if (!caption.isEmpty() && s.isEmpty()) {
                    break;
                }
                if (s.startsWith("Table") || !caption.isEmpty()) {
                    caption = caption + " " + s;
                }
            }
        return caption;
    }
    

    public String getNumberCellsInfo(String pdfStr) throws IOException, Exception {
        String val = "";
        List<NLPLeaderboardTable> tables = this.getCleanedTables(pdfStr);
        for (NLPLeaderboardTable table : tables) {
            val = val + "Table caption#"+  getCleanTableCaption(normalizeText(table.getCaptionLayoutTokens()))
                    + "\n\n";
            List<TableCell> numberCells = table.getNumberCells();
            for (TableCell c1 : numberCells) {
                val = val  + "number#" + normalizeText(c1.lt) + "\n" + "IsBolded#" + c1.isBold + "\n";
                for (String s : c1.getAssociatedTagsStr_row()) {
                    val = val+ "associated row#" + s + "\n";
                }
                for (String s : c1.getAssociatedTagsStr_column_mergedAll()) {
                    val = val+ "associated column#" + s + "\n";
                }
                for (String s : c1.getAssociatedTagsStr_column()) {
                    val = val  + "associated column#" + s + "\n";
                }
                val = val + "\n";
            }
        }
        return val;
    }
    
        public String getNumberCellsInfo_fromCache(String pdfStr) throws IOException, Exception {
        String val = "";
        List<CachedTable> tables = this.getTableInfoFromPDF(pdfStr);
        for (CachedTable table : tables) {
            val = val + "Table caption#"+  getCleanTableCaption(table.caption)
                    + "\n\n";
            Set<CachedTable.NumberCell> numberCells = table.numberCells;
            for (CachedTable.NumberCell c1 : numberCells) {
                val = val  + "number#" + c1.number + "\n" + "IsBolded#" + c1.isBolded + "\n";
                for (String s : c1.associatedRows) {
                    val = val+ "associated row#" + s + "\n";
                }
                for (String s : c1.associatedMergedColumns) {
                    val = val+ "associated column#" + s + "\n";
                }
                for (String s : c1.associatedColumns) {
                    val = val  + "associated column#" + s + "\n";
                }
                val = val + "\n";
            }
        }
        return val;
    }


    public String normalizeText(List<LayoutToken> tokens) {
//        return StringUtils.normalizeSpace(toText(tokens).replace("\n", " "));
//        return StringUtils.normalizeSpace(toText(tokens));
        return toText(tokens);
    }

    private boolean captionAlignVerticalWithContent(BoundingBox caption, BoundingBox content) {
        boolean align = false;
        if (caption.getX() >= content.getX() && caption.getX2() <= content.getX2()) {
            align = true;
        }
        if (caption.getX() <= content.getX() && caption.getX2() >= content.getX2()) {
            align = true;
        }
        if (caption.getX() >= content.getX() && caption.getX2() >= content.getX2() && content.getX2() > caption.getX()) {
            double overlap = content.getX2() - caption.getX();
            double dist = caption.getX() - content.getX();
            if (dist < 10 || overlap / content.getWidth() > 0.8) {
                align = true;
            } else {
                align = false;
            }
        }
        if (caption.getX() <= content.getX() && caption.getX2() <= content.getX2() && caption.getX2() > content.getX()) {
            double overlap = caption.getX2() - content.getX();
            double dist = content.getX() - caption.getX();
            if (dist < 10 || overlap / content.getWidth() > 0.8) {
                align = true;
            } else {
                align = false;
            }
        }

        return align;
    }

    private int countNumbersInAText(String text) {
        int numberCount = 0;
        Pattern number = Pattern.compile("\\d+|\\d+\\.\\d+|\\d+\\.\\d+\\%|\\d+\\%|\\.\\d+");
        Pattern word = Pattern.compile("(?<!\\S)\\p{Alpha}+(?!\\S)");
        Matcher matcherNumber = number.matcher(text);
        Matcher matcherWord = word.matcher(text);
        while (matcherNumber.find()) {
            numberCount++;
        }
        return numberCount;
    }

//to detect small tables which only have two columns, and numbers only shown in one column    
// "Model \n" +
//"Score \n" +
//"2-Decoder Wins \n" +
//"49 \n" +
//"Pointer-Generator Wins \n" +
//"31 \n" +
//"Non-distinguishable \n" +
//"20 "
    private boolean twoColumnTableContent(String text) {
        if (text.matches("[\\s\\S]*? \\n(\\d+|\\d+\\.\\d+|\\d+\\%|\\d+\\.\\d+\\%|\\.\\d+) \\n(.*?) \\n(\\d+|\\d+\\.\\d+|\\d+\\%|\\d+\\.\\d+\\%|\\.\\d+) \\n[\\s\\S]*?")) //   if(text.matches("[\\s\\S]*?\\n(\\d+) \\n[\\s\\S]*?"))
        {
            return true;
        }
        return false;
    }

    //whether the text contains two or three consecutive numbers
    private boolean hasConsecutiveNumbers(String text) {
        if (text.matches(".*?(\\d+|\\d+\\.\\d+|\\d+\\%|\\d+\\.\\d+\\%|\\.\\d+) (\\d+|\\d+\\.\\d+|\\d+\\%|\\d+\\.\\d+\\%|\\.\\d+).*?")
                || text.matches(".*?(\\d+|\\d+\\.\\d+|\\d+\\%|\\d+\\.\\d+\\%|\\.\\d+) (\\d+|\\d+\\.\\d+|\\d+\\%|\\d+\\.\\d+\\%|\\.\\d+) (\\d+|\\d+\\.\\d+|\\d+\\%|\\d+\\.\\d+\\%|\\.\\d+).*?")
                || text.matches("")) {
            return true;
        }
        return false;
    }

    private List<Figure> processFigures(String rese, List<LayoutToken> layoutTokens) {

        List<Figure> results = new ArrayList<>();

        TaggingTokenClusteror clusteror = new TaggingTokenClusteror(GrobidModels.FULLTEXT, rese, layoutTokens, true);

        for (TaggingTokenCluster cluster : clusteror.cluster()) {
            if (cluster.getTaggingLabel() != TaggingLabels.FIGURE) {
                continue;
            }
            List<LayoutToken> tokenizationFigure = cluster.concatTokens();
            if (tokenizationFigure.isEmpty()) {
                continue;
            }
            NLPLeaderboardFigParser figParser = new NLPLeaderboardFigParser();
            Figure result = figParser.processing(tokenizationFigure,
                    cluster.getFeatureBlock());
            SortedSet<Integer> blockPtrs = new TreeSet<>();
            for (LayoutToken lt : tokenizationFigure) {
                if (!LayoutTokensUtil.spaceyToken(lt.t()) && !LayoutTokensUtil.newLineToken(lt.t())) {
                    blockPtrs.add(lt.getBlockPtr());
                }
            }
            result.setBlockPtrs(blockPtrs);

            result.setLayoutTokens(tokenizationFigure);

            // the first token could be a space from previous page
            for (LayoutToken lt : tokenizationFigure) {
                if (!LayoutTokensUtil.spaceyToken(lt.t()) && !LayoutTokensUtil.newLineToken(lt.t())) {
                    result.setPage(lt.getPage());
                    break;
                }
            }

            results.add(result);
            result.setId("" + (results.size() - 1));
        }
        return results;
    }

    private List<Table> processTables(String rese,
            List<LayoutToken> tokenizations) {
        List<Table> results = new ArrayList<>();
        TaggingTokenClusteror clusteror = new TaggingTokenClusteror(GrobidModels.FULLTEXT, rese, tokenizations, true);
//        TaggingTokenSynchronizer taggingTokenSynchronizer = new TaggingTokenSynchronizer(GrobidModels.FULLTEXT, rese, tokenizations);
//        PeekingIterator<LabeledTokensContainer> it = Iterators.peekingIterator(taggingTokenSynchronizer);
//        while (it.hasNext()) {
//            LabeledTokensContainer cont = it.next();
//            System.err.println("container:" + cont.getPlainLabel() + ":" + cont.getToken() + ":" + cont.isBeginning());
//        }        

//        List<TaggingTokenClusteror> correctionCluster = new ArrayList();
//        for(TaggingTokenCluster cluster: clusteror.cluster()){
//            List<LayoutToken> tokenizationTable = cluster.concatTokens();
//            String s = "";
//            for(LayoutToken t: tokenizationTable){
//                s = s + t.getText();
//            }
//            System.err.println("correction:" + cluster.getTaggingLabel().getLabel() +":" + s);
////            if(s.matches("Table " + "\\d" + ":" + ".*?")){
////                correctionCluster.add(clusteror);
////                System.err.println("added");
////            }
//        }
//        for (TaggingTokenCluster cluster : Iterables.filter(clusteror.cluster(),
//                new TaggingTokenClusteror.LabelTypePredicate(TaggingLabels.TABLE))) {
        for (TaggingTokenCluster cluster : clusteror.postClusterChangeLabel(clusteror.cluster())) {
//        for (TaggingTokenCluster cluster : clusteror.cluster()) {
            if (cluster.getTaggingLabel() != TaggingLabels.TABLE) {
                continue;
            }
            List<LayoutToken> tokenizationTable = cluster.concatTokens();
//            String s = "";
//            for(LayoutToken t: tokenizationTable){
//                s = s + t.getText();
//            }
//            System.err.println("cluster:" + s);
            if (tokenizationTable.isEmpty()) {
                continue;
            }
            Table result = parsers.getTableParser().processing(
                    tokenizationTable,
                    cluster.getFeatureBlock()
            );

            SortedSet<Integer> blockPtrs = new TreeSet<>();
            for (LayoutToken lt : tokenizationTable) {
                if (!LayoutTokensUtil.spaceyToken(lt.t()) && !LayoutTokensUtil.newLineToken(lt.t())) {
                    blockPtrs.add(lt.getBlockPtr());
                }
            }
            result.setBlockPtrs(blockPtrs);
            result.setLayoutTokens(tokenizationTable);

            // the first token could be a space from previous page
            for (LayoutToken lt : tokenizationTable) {
                if (!LayoutTokensUtil.spaceyToken(lt.t()) && !LayoutTokensUtil.newLineToken(lt.t())) {
                    result.setPage(lt.getPage());
                    break;
                }
            }
            results.add(result);
            result.setId("" + (results.size() - 1));
        }
//        postProcessTables(results);
        return results;
    }

    private void postProcessTables(List<Table> tables) {
        for (Table table : tables) {
            if (!table.firstCheck()) {
                continue;
            }

            // cleaning up tokens
            List<LayoutToken> fullDescResult = new ArrayList<>();
            BoundingBox curBox = BoundingBox.fromLayoutToken(table.getFullDescriptionTokens().get(0));
            int distanceThreshold = 200;
            for (LayoutToken fdt : table.getFullDescriptionTokens()) {
                BoundingBox b = BoundingBox.fromLayoutToken(fdt);
                if (b.getX() < 0) {
                    fullDescResult.add(fdt);
                    continue;
                }
                if (b.distanceTo(curBox) > distanceThreshold) {
                    Engine.getCntManager().i(TableRejectionCounters.HEADER_NOT_CONSECUTIVE);
                    table.setGoodTable(false);
                    break;
                } else {
                    curBox = curBox.boundBox(b);
                    fullDescResult.add(fdt);
                }
            }
            table.getFullDescriptionTokens().clear();
            table.getFullDescriptionTokens().addAll(fullDescResult);

            List<LayoutToken> contentResult = new ArrayList<>();

            curBox = BoundingBox.fromLayoutToken(table.getContentTokens().get(0));
            for (LayoutToken fdt : table.getContentTokens()) {
                BoundingBox b = BoundingBox.fromLayoutToken(fdt);
                if (b.getX() < 0) {
                    contentResult.add(fdt);
                    continue;
                }

                if (b.distanceTo(curBox) > distanceThreshold) {
                    break;
                } else {
                    curBox = curBox.boundBox(b);
                    contentResult.add(fdt);
                }
            }
            table.getContentTokens().clear();
            table.getContentTokens().addAll(contentResult);

            table.secondCheck();
        }
    }

    public List<LayoutToken> getLayoutTokens(String pdfPath) throws IOException, Exception {
        GrobidAnalysisConfig config = GrobidAnalysisConfig.defaultInstance();
        org.grobid.core.document.Document doc = engine.fullTextToTEIDoc(new File(pdfPath), config);
        return doc.getTokenizations();
    }

    public String getFullTextToTEI(String pdfPath) throws IOException, Exception {
        GrobidAnalysisConfig config = GrobidAnalysisConfig.defaultInstance();
        String xml = engine.fullTextToTEI(new File(pdfPath), config);
        org.grobid.core.document.Document doc = engine.fullTextToTEIDoc(new File(pdfPath), config);
        return xml;
    }


}
