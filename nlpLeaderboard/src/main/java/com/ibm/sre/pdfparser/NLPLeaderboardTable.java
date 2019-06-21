/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.ibm.sre.pdfparser;

import com.google.common.collect.Iterators;
import com.google.common.collect.PeekingIterator;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import nu.xom.Attribute;
import nu.xom.Element;
import org.apache.commons.lang3.StringUtils;
import org.grobid.core.data.Figure;
import org.grobid.core.document.xml.XmlBuilderUtils;
import org.grobid.core.engines.Engine;
import org.grobid.core.engines.config.GrobidAnalysisConfig;
import org.grobid.core.engines.counters.TableRejectionCounters;
import org.grobid.core.layout.BoundingBox;
import org.grobid.core.layout.LayoutToken;
import org.grobid.core.tokenization.LabeledTokensContainer;
import org.grobid.core.utilities.BoundingBoxCalculator;
import org.grobid.core.utilities.LayoutTokensUtil;
import org.grobid.core.utilities.TextUtilities;
import org.grobid.core.utilities.counters.CntManager;

/**
 *
 * @author yhou
 */
public class NLPLeaderboardTable extends Figure {

    private List<List<LayoutToken>> contentTokens = new ArrayList<>();
    private List<LayoutToken> contentTokens_clean = new ArrayList();
    private List<LayoutToken> fullDescriptionTokens = new ArrayList<>();
    private List<TableCell> tableCells = new ArrayList();
    private List<TableCell> tableCells_clean = new ArrayList();
    private boolean goodTable = true;

    public void setGoodTable(boolean goodTable) {
        this.goodTable = goodTable;
    }

    public NLPLeaderboardTable() {
        caption = new StringBuilder();
        header = new StringBuilder();
        content = new StringBuilder();
        label = new StringBuilder();
    }

    public class TableCell {

        public Map<Double, TableCell> associatedTags_row;
        public Map<Double, TableCell> associatedTags_column;
        public List<TableCell> associatedTags_column_mergedAll;
        public List<LayoutToken> lt;
        public String font;
        public boolean isNumberCell;
        public boolean isRowHead;
        public boolean isColumnHead;
        public boolean isBold;
        public BoundingBox box;

        public TableCell() {
            associatedTags_row = new TreeMap();
            associatedTags_column = new TreeMap();
            associatedTags_column_mergedAll = new ArrayList();
            lt = new ArrayList();
            font = "";
            isNumberCell = false;
            isRowHead = false;
            isColumnHead = false;
            isBold = false;
            box = null;
        }

        public List<TableCell> getAssociatedTags_row() {
            List<TableCell> rows = new ArrayList();
            for (TableCell cell : associatedTags_row.values()) {
                rows.add(cell);
            }
            return rows;
        }

        public List<String> getAssociatedTagsStr_row() {
            List<String> rows = new ArrayList();
            for (TableCell cell : associatedTags_row.values()) {
                String s = "";
                for(LayoutToken lt: cell.lt){
                    s = s + " " + lt.t();
                }
                rows.add(s.trim());
            }
            return rows;
        }



        public List<String> getAssociatedTagsStr_column() {
            List<String> columns = new ArrayList();
            for (TableCell cell : associatedTags_column.values()) {
                String s = "";
                for(LayoutToken lt: cell.lt){
                    s = s + " " + lt.t();
                }
                columns.add(s.trim());
            }
            return columns;
        }


        public List<TableCell> getAssociatedTags_column() {
            List<TableCell> columns = new ArrayList();
            for (TableCell cell : associatedTags_column.values()) {
                columns.add(cell);
            }
            return columns;
        }



        public List<TableCell> getAssociatedTags_column_mergedAll() {
            return associatedTags_column_mergedAll;
        }

        public List<String> getAssociatedTagsStr_column_mergedAll() {
            List<String> columns_mergerAll = new ArrayList();
            for (TableCell cell : associatedTags_column_mergedAll) {
                String s = "";
                for(LayoutToken lt: cell.lt){
                    s = s + " " + lt.t();
                }
                columns_mergerAll.add(s.trim());
            }
            
            return columns_mergerAll;
        }



    }

    private String cleanString(String input) {
        return input.replace("\n", " ").replace("  ", " ").trim();
    }

    public List<List<LayoutToken>> getContentTokens() {
        return contentTokens;
    }

    public List<LayoutToken> getContentTokens_Clean() {
        return contentTokens_clean;
    }

    public void setContentTokens_Clean(List<LayoutToken> contentTokens) {
        contentTokens_clean = contentTokens;
    }



    public List<LayoutToken> getFullDescriptionTokens() {
        return fullDescriptionTokens;
    }

    public boolean isGoodTable() {
        return goodTable;
    }

    //12--**
    //12.34--**.**
    private String getNumberFormat(String number) {
        String format = "";
        if (number.matches("\\d+|\\d+\\.\\d+|\\d+\\%|\\d+\\.\\d+\\%")) {
            for (char ch : number.toCharArray()) {
                if (ch >= '0' && ch <= '9') {
                    format = format + " " + "*";
                } else {
                    format = format + " " + ch;
                }
            }
        }
        return format.trim();
    }

    private String getMostFrequentKey(Map<String, Integer> map) {
        if (map.isEmpty()) {
            return null;
        }
        String val = map.keySet().iterator().next();
        for (String s : map.keySet()) {
            if (s == val) {
                continue;
            }
            if (map.get(s) > map.get(val)) {
                val = s;
            } else if (map.get(s) == map.get(val)) {
                if (!s.contains("medi")) {
                    val = s;
                }
            }
        }
        return val;
    }

    public List<TableCell> getBoldedNumberCells() {
        List<TableCell> val = new ArrayList();
        processTableCells();
        for (TableCell c : tableCells_clean) {
            if (c.isNumberCell && c.isBold) {
                val.add(c);
            }
        }
        return val;
    }
    
    
        public List<TableCell> getNumberCells() {
        List<TableCell> val = new ArrayList();
        processTableCells();
        for (TableCell c : tableCells_clean) {
            if (c.isNumberCell) {
                val.add(c);
            }
        }
        return val;
    }

    private void processTableCells() {
        if (this.contentTokens_clean.isEmpty()) {
            return;
        }
//        String tableContent = LayoutTokensUtil.toText(contentTokens_clean);
//        String[] rows = tableContent.split("\n");

        //collect all cells and their layoutToken
        List<TableCell> firstCollection = new ArrayList();
        TableCell cell = new TableCell();
        firstCollection.add(cell);

        for (LayoutToken lt : contentTokens_clean) {
            if (lt.t().matches("\\s")) {
                cell = new TableCell();
                firstCollection.add(cell);
            } else {
                cell.lt.add(lt);
                cell.font = lt.getFont();
            }
        }

        for (TableCell c1 : firstCollection) {
            if (c1.lt.size() > 0) {
                tableCells.add(c1);
//                System.err.println(LayoutTokensUtil.toText(c1.lt));
            }
        }

        //tag number cells
        //sometimes, a number cell like 1 or 2 is part of a tag name, e.g., ROUCE 1, ROUCE 2
        //have to exclude tagging these cells as number cells by comparing with other number cells
        //numberFormat: xx, xx.xx, xx.xx%, xx%
        Map<String, Integer> numberFormat = new TreeMap();
        Map<String, Integer> fontMap = new TreeMap();

        for (TableCell c1 : tableCells) {
            if (LayoutTokensUtil.toText(c1.lt).matches("\\d+|\\d+\\.\\d+|\\d+\\%|\\d+\\.\\d+\\%|\\.\\d+|.\\d+\\.\\d+|\\d+\\.\\d+.|.\\.\\d+|\\.\\d+.|.\\d+\\.\\d+\\%|\\d+\\.\\d+\\%")) {
                String format = getNumberFormat(LayoutTokensUtil.toText(c1.lt));
                if (numberFormat.containsKey(format)) {
                    int oldCount = numberFormat.get(format);
                    numberFormat.put(format, oldCount + 1);
                } else {
                    numberFormat.put(format, 1);
                }
                //fontMap
                if (fontMap.containsKey(c1.font)) {
                    int oldCount = fontMap.get(c1.font);
                    fontMap.put(c1.font, oldCount + 1);
                } else {
                    fontMap.put(c1.font, 1);
                }
            }
        }

        //detect number cells
        for (TableCell c1 : tableCells) {
            //first, detectly obvious number cells, this include numbers with one special label, like 35.7*
            if (LayoutTokensUtil.toText(c1.lt).matches("\\d+\\.\\d+|\\d+\\.\\d+\\%|\\.\\d+|.\\d+\\.\\d+|\\d+\\.\\d+.|.\\.\\d+|\\.\\d+.|.\\d+\\.\\d+\\%|\\d+\\.\\d+\\%")) {
                c1.isNumberCell = true;
                if (fontMap.containsKey(c1.font)) {
                    if (fontMap.size() == 2 && !c1.font.equalsIgnoreCase(getMostFrequentKey(fontMap))) {
                        c1.isBold = true;
                    } else if (!c1.font.equalsIgnoreCase(getMostFrequentKey(fontMap)) && fontMap.get(c1.font) < 10) {
                        c1.isBold = true;
                    }
                }

            }
            //for the mixture of x, xx, xx.xx, and x is part of the head instead of number cell
            //and it needs to be merged into some column head
            if (LayoutTokensUtil.toText(c1.lt).matches("\\d+|\\d+\\.\\d+|\\d+\\%|\\d+\\.\\d+\\%")) {
                String format = getNumberFormat(LayoutTokensUtil.toText(c1.lt));
                if (numberFormat.containsKey(format)) {
                    if (format.equalsIgnoreCase(getMostFrequentKey(numberFormat))) {
                        c1.isNumberCell = true;
                        //detect whether the number is bolded
                        if (fontMap.containsKey(c1.font)) {
                            if (fontMap.size() == 2 && !c1.font.equalsIgnoreCase(getMostFrequentKey(fontMap))) {
                                c1.isBold = true;
                            } else if (!c1.font.equalsIgnoreCase(getMostFrequentKey(fontMap)) && fontMap.get(c1.font) < 10) {
                                c1.isBold = true;
                            }
                        }
                    }
                }
            }
        }

        //get the clean table cells, merge some table cells, e.g., row head: RL, +, pg, +, cbdec; or column head: f 1
        TableCell cell_clean = tableCells.get(0);
        tableCells_clean.add(cell_clean);

        for (int i = 1; i < tableCells.size(); i++) {
            TableCell cellOld = tableCells.get(i);
            BoundingBox curcellBox = BoundingBoxCalculator.calculateOneBox(cell_clean.lt, true);
            BoundingBox cellOldBox = BoundingBoxCalculator.calculateOneBox(cellOld.lt, true);
            if(cellOldBox==null) continue;
            if(curcellBox==null) continue;
            if (!cellOld.isNumberCell && (cellOldBox.getX() > curcellBox.getX2() && curcellBox.distanceTo(cellOldBox) < 5 && tagHorizontallyAlignedWithNumber(curcellBox, cellOldBox))) {
                cell_clean.lt.addAll(cellOld.lt);
            } else {
                cell_clean = cellOld;
                tableCells_clean.add(cell_clean);
            }
        }

        //calculate the bounding box for tableCells_clean
        for (TableCell c1 : tableCells_clean) {
            c1.box = BoundingBoxCalculator.calculateOneBox(c1.lt, true);
//            System.err.println(LayoutTokensUtil.toText(c1.lt));
        }

        //associated tags to numberCells, and cellect all columnHeads
        for (TableCell c1 : tableCells_clean) {
            if (c1.isNumberCell) {
                BoundingBox numberBox = c1.box;
                for (TableCell c2 : tableCells_clean) {
                    if (!c2.isNumberCell) {
                        BoundingBox tagBox = c2.box;
                        if (tagHorizontallyAlignedWithNumber(tagBox, numberBox)) {
                            c1.associatedTags_row.put(c2.box.getX(), c2);
                            c2.isRowHead = true;
//                            System.err.println(LayoutTokensUtil.toText(c1.lt) + "---" + LayoutTokensUtil.toText(c2.lt));
                        }
                    }
                }
            }
        }

        //find the box for all number cells
        TableCell allFirstTagRowHead = new TableCell();
        TableCell allNumbers = new TableCell();
        for (TableCell c1 : tableCells_clean) {
            if (c1.isNumberCell) {
                allNumbers.lt.addAll(c1.lt);
            }
            if(c1.isRowHead)
               allFirstTagRowHead.lt.addAll(c1.lt);
        }
        if (allNumbers.lt.size() > 0) {
            allNumbers.box = BoundingBoxCalculator.calculateOneBox(allNumbers.lt, true);
        }
        if(allFirstTagRowHead.lt.size()>0){
            allFirstTagRowHead.box = BoundingBoxCalculator.calculateOneBox(allFirstTagRowHead.lt, true);
        }
        //
        List<TableCell> rowHeads = new ArrayList();
        List<TableCell> columnHeads = new ArrayList();
        for (TableCell c1 : tableCells_clean) {
            if (c1.isNumberCell) {
                continue;
            } else if (c1.isRowHead) {
                rowHeads.add(c1);
            } else {
                //the current box is completely "lefter"
                boolean isMergedRowHead = false;
                if (allNumbers.box != null && allFirstTagRowHead.box!=null) {
                    boolean left = c1.box.getX2() < allNumbers.box.getX();
                    boolean right = allNumbers.box.getX2() < c1.box.getX();
                    boolean bottom = allNumbers.box.getY2() < c1.box.getY();
                    boolean top = c1.box.getY2() < allNumbers.box.getY();

                    boolean left1 = c1.box.getX2() < allFirstTagRowHead.box.getX();
                    boolean right1 = allFirstTagRowHead.box.getX2() < c1.box.getX();
                    boolean bottom1 = allFirstTagRowHead.box.getY2() < c1.box.getY();
                    boolean top1 = c1.box.getY2() < allFirstTagRowHead.box.getY();

                    if (left && !top && !bottom&&left1&&!top1&&!bottom1) {
                        c1.isRowHead = true;
                        rowHeads.add(c1);
                        isMergedRowHead = true;
                    }
                }
                if (!isMergedRowHead) {
                    columnHeads.add(c1);
                }
            }
        }

        //row head association
        //horizontally, for each number cell, associate it to the closest merged row
        //first collect all row heads and cluster them according to x position
        TreeMap<Double, List<TableCell>> columnOfRows = new TreeMap();
        for (TableCell c1 : rowHeads) {
            boolean columnOfRowsStart = true;
            for (List<TableCell> cells : columnOfRows.values()) {
                if (cells.contains(c1)) {
                    columnOfRowsStart = false;
                    break;
                }
            }
            if (!columnOfRowsStart) {
                continue;
            }
            List<TableCell> cellInOneRow = new ArrayList();
            cellInOneRow.add(c1);
            BoundingBox c1Box = c1.box;
            for (TableCell c2 : rowHeads) {
                BoundingBox c2Box = c2.box;
                if (tagVerticallyAlignedWithNumber(c1Box, c2Box)) {
                    cellInOneRow.add(c2);
                }
            }
            columnOfRows.put(c1Box.getX(), cellInOneRow);
        }
        //starting working from the most left row, then moving towards the right
        for (Entry<Double, List<TableCell>> item : columnOfRows.entrySet()) {
            if (item.getValue().size() <= 1) {
                continue;
            }
            //tagNum>=2
            Set<TableCell> nonAlignedNumbers = new HashSet();
            for (TableCell c1 : tableCells_clean) {
                if (!c1.isNumberCell) {
                    continue;
                }
                BoundingBox c1Box = c1.box;
                boolean tagged = false;
                for (TableCell tag : item.getValue()) {
                    if (tagHorizontallyAlignedWithNumber(tag.box, c1Box)) {
                        tagged = true;
                        c1.associatedTags_row.put(tag.box.getX(), tag);
                    }
                }
                if (!tagged) {
                    nonAlignedNumbers.add(c1);
                }

            }
            //assign nonAlinedNumbers to column names, tag each number cell with the closest element among all members in the same row
            for (TableCell c1 : nonAlignedNumbers) {
                double curdistance = 1000;
                TableCell predictedTag = null;
                for (TableCell tag : item.getValue()) {
                    if (tag.box.getX2() < c1.box.getX()) {//make sure the tag is on the left 
                        if (calculateDistanceOfTwoBox(tag.box, c1.box) < curdistance) {
                            curdistance = calculateDistanceOfTwoBox(tag.box, c1.box);
                            predictedTag = tag;
                        }
                    }
                }
                if (predictedTag != null) {
                    c1.associatedTags_row.put(predictedTag.box.getX(), predictedTag);
                }
            }
        }

        //column head association
        //vertically, for each number cell, associate it to the closest one
        //collect all column heads and cluster them according to y position
        TreeMap<Double, List<TableCell>> rowsOfColumn = new TreeMap(Collections.reverseOrder());
        for (TableCell c1 : columnHeads) {
            boolean rowOfColumnStart = true;
            for (List<TableCell> cells : rowsOfColumn.values()) {
                if (cells.contains(c1)) {
                    rowOfColumnStart = false;
                    break;
                }
            }
            if (!rowOfColumnStart) {
                continue;
            }
            List<TableCell> cellInOneRow = new ArrayList();
            cellInOneRow.add(c1);
            BoundingBox c1Box = c1.box;
            for (TableCell c2 : columnHeads) {
                BoundingBox c2Box = c2.box;
                if (tagHorizontallyAlignedWithNumber(c1Box, c2Box)) {
                    cellInOneRow.add(c2);
                }
            }
            rowsOfColumn.put(c1Box.getY(), cellInOneRow);
        }

        //starting work from the most bottom column, then go towards up
        for (Entry<Double, List<TableCell>> item : rowsOfColumn.entrySet()) {
            if (item.getValue().size() == 1) {
                //if the mergedAll column is in the first row, it applied to all number cells and add it to
                //associatedTags_column
                if (item.getKey() == rowsOfColumn.lastKey()) {
                    BoundingBox columnBox = item.getValue().iterator().next().box;
                    for (TableCell c1 : tableCells_clean) {
                        if (c1.isNumberCell) {
                            BoundingBox c1Box = c1.box;
                            if (c1Box.getY() > columnBox.getY2()) {
                                c1.associatedTags_column.put(item.getValue().iterator().next().box.getY(), item.getValue().iterator().next());
                            }
                        }
                    }

                } else {
                    //this is a mergedAll column, all number cells below it should be tagged with it as a column head
                    //a number cell can only belong to a mergedAll column, choose the closest one among all mergedAll column
                    BoundingBox columnBox = item.getValue().iterator().next().box;
                    for (TableCell c1 : tableCells_clean) {
                        if (c1.isNumberCell) {
                            BoundingBox c1Box = c1.box;
                            if (c1Box.getY() > columnBox.getY2() && c1.associatedTags_column_mergedAll.isEmpty()) {
                                c1.associatedTags_column_mergedAll.add(item.getValue().iterator().next());
                            }
                        }
                    }
                }
            } else {
                //tagNum>=2
                Set<TableCell> nonAlignedNumbers = new HashSet();
                for (TableCell c1 : tableCells_clean) {
                    if (!c1.isNumberCell) {
                        continue;
                    }
                    BoundingBox c1Box = c1.box;
                    boolean tagged = false;
                    for (TableCell tag : item.getValue()) {
                        if (tagVerticallyAlignedWithNumber(tag.box, c1Box)) {
                            tagged = true;
                            c1.associatedTags_column.put(tag.box.getY(), tag);
                        }
                    }
                    if (!tagged) {
                        nonAlignedNumbers.add(c1);
                    }

                }
                //assign nonAlinedNumbers to column names, tag each number cell with the closest element among all members in the same row
                for (TableCell c1 : nonAlignedNumbers) {
                    double curdistance = 1000;
                    TableCell predictedTag = null;
                    for (TableCell tag : item.getValue()) {
                        if (tag.box.getY2() < c1.box.getY()) {//make sure tag in on the top 
                            if (calculateDistanceOfTwoBox(tag.box, c1.box) < curdistance) {
                                curdistance = calculateDistanceOfTwoBox(tag.box, c1.box);
                                predictedTag = tag;
                            }
                        }
                    }
                    if (predictedTag != null) {
                        c1.associatedTags_column.put(predictedTag.box.getY(), predictedTag);
                    }
                }
            }
        }
    }

    //the original BoundingBox.distanceTo() has a bug when the second box 
    //is in top-right of the second box, the distance is NaN
    private double calculateDistanceOfTwoBox(BoundingBox box1, BoundingBox to) {
        if (box1.getPage() != to.getPage()) {
            return 1000 * Math.abs(box1.getPage() - to.getPage());
        }

        //the current box is completely "lefter"
        boolean left = box1.getX2() < to.getX();
        boolean right = to.getX2() < box1.getX();
        boolean bottom = to.getY2() < box1.getY();
        boolean top = box1.getY2() < to.getY();
        if (top && left) {
            return dist(box1.getX2(), box1.getY2(), to.getX(), to.getY());
        } else if (left && bottom) {
            return dist(box1.getX2(), box1.getY(), to.getX(), to.getY2());
        } else if (bottom && right) {
            return dist(box1.getX(), box1.getY(), to.getX2(), to.getY2());
        } else if (right && top) {
            return dist(box1.getX(), box1.getY2(), to.getX2(), to.getY());
        } else if (left) {
            return to.getX() - box1.getX2();
        } else if (right) {
            return box1.getX() - to.getX2();
        } else if (bottom) {
            return box1.getY() - to.getY2();
        } else if (top) {
            return to.getY() - box1.getY2();
        } else {
            return 0;
        }

    }

    private double dist(double x1, double y1, double x2, double y2) {
        return Math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
    }

    //tag is on the left part, number is on the right part
    //tag is the row head for the number
    //Horizontally, tag(number) is within the number(tag)
    //if the tag and the number horizontally overlap, then the overlap part
    // should be more than half of the number box's height
    private boolean tagHorizontallyAlignedWithNumber(BoundingBox tag, BoundingBox number) {
        boolean align = false;
        if (tag.getX2() > number.getX()) {
            return false;
        }
        if (tag.getY() == number.getY() && tag.getY2() == number.getY2()) {
            align = true;
        }
        else if (tag.getY() <= number.getY() && tag.getY2() >= number.getY2()) {
            align = true;
        }
        else if (tag.getY() >= number.getY() && tag.getY2() <= number.getY2()) {
            align = true;
        }
        else if (tag.getY() >= number.getY() && tag.getY2() >= number.getY2() && number.getY2() > tag.getY()) {
            double overlap = number.getY2() - tag.getY();
            if (overlap / number.getHeight() > 0.5) {
                align = true;
            } else {
                align = false;
            }
        }
        else if (tag.getY() <= number.getY() && tag.getY2() <= number.getY2() && tag.getY2() > number.getY()) {
            double overlap = tag.getY2() - number.getY();
            if (overlap / number.getHeight() > 0.5) {
                align = true;
            } else {
                align = false;
            }
        }else {
            align = false;
        }
        return align;
    }

    //more strict than horisontally alignment
    //tag is above the number
    private boolean tagVerticallyAlignedWithNumber(BoundingBox tag, BoundingBox number) {
        boolean align = false;
        if (tag.getY2() >= number.getY()) {
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
            if (overlap / number.getWidth() > 0.7) {
                align = true;
            } else {
                align = false;
            }
        }
        else if (tag.getX() <= number.getX() && tag.getX2() <= number.getX2() && tag.getX2() > number.getX()) {
            double overlap = tag.getX2() - number.getX();
            if (overlap / number.getWidth() > 0.7) {
                align = true;
            } else {
                align = false;
            }
        }else {
           align = false;
        }
        return align;
    }

}
