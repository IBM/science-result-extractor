package com.ibm.sre.pdfparser;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.Lists;
import com.google.common.collect.PeekingIterator;
import com.ibm.sre.pdfparser.TaggingTokenCluster;

import org.grobid.core.engines.label.TaggingLabel;
import org.grobid.core.layout.LayoutToken;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.grobid.core.GrobidModel;
import org.grobid.core.engines.label.TaggingLabels;
import org.grobid.core.layout.BoundingBox;
import org.grobid.core.tokenization.LabeledTokensContainer;
import org.grobid.core.tokenization.TaggingTokenSynchronizer;
import org.grobid.core.utilities.LayoutTokensUtil;
import static org.grobid.core.utilities.LayoutTokensUtil.toText;

/**
 * Created by zholudev on 12/01/16.
 * Cluster of related tokens
 */
public class TaggingTokenClusteror {
    private final TaggingTokenSynchronizer taggingTokenSynchronizer;

    public static class LabelTypePredicate implements Predicate<org.grobid.core.tokenization.TaggingTokenCluster> {
        private TaggingLabel label;

        public LabelTypePredicate(TaggingLabel label) {
            this.label = label;
        }

        @Override
        public boolean apply(org.grobid.core.tokenization.TaggingTokenCluster taggingTokenCluster) {
            return taggingTokenCluster.getTaggingLabel() == label;
        }
    }

    public static class LabelTypeExcludePredicate implements Predicate<org.grobid.core.tokenization.TaggingTokenCluster> {
        private TaggingLabel[] labels;

        public LabelTypeExcludePredicate(TaggingLabel... labels) {
            this.labels = labels;
        }

        @Override
        public boolean apply(org.grobid.core.tokenization.TaggingTokenCluster taggingTokenCluster) {
            for (TaggingLabel label : labels) {
                if (taggingTokenCluster.getTaggingLabel() == label) {
                    return false;
                }
            }
            return true;
        }
    }

    public TaggingTokenClusteror(GrobidModel grobidModel, String result, List<LayoutToken> tokenizations) {
        taggingTokenSynchronizer = new TaggingTokenSynchronizer(grobidModel, result, tokenizations);
    }

    public TaggingTokenClusteror(GrobidModel grobidModel, String result, List<LayoutToken> tokenizations,
                                 boolean computerFeatureBlock) {
        taggingTokenSynchronizer = new TaggingTokenSynchronizer(grobidModel, result, tokenizations, computerFeatureBlock);
    }
    
    //improve recall of table caption recognition
    public List<TaggingTokenCluster> postClusterChangeLabel(List<TaggingTokenCluster> result){
        //change some cluster labels, improve recall for Table caption detection
        List<TaggingTokenCluster> result1 = new ArrayList();
        for(int i=0; i<result.size()-1; i++){
            TaggingTokenCluster cur = result.get(i);
            TaggingTokenCluster next = result.get(i+1);
            if(cur.getTaggingLabel()==TaggingLabels.TABLE_MARKER&&next.getTaggingLabel()==TaggingLabels.PARAGRAPH){
                if(next.concatTokens().size()==0)continue;
                if(next.concatTokens().get(0).getText().equalsIgnoreCase(":")||next.concatTokens().get(0).getText().equalsIgnoreCase(".")||next.concatTokens().get(0).getText().matches("[A-Z].*?")){
                    cur.setTaggingLabel(TaggingLabels.TABLE);
                    for(LabeledTokensContainer ltc: next.getLabeledTokensContainers())
                        cur.addLabeledTokensContainer(ltc);
                    result1.add(cur);
                }
            }else{
              result1.add(cur);
            }
        }
     
        return result1;
    
    }

    public List<TaggingTokenCluster> cluster() {
        
        List<TaggingTokenCluster> result = new ArrayList<>();

        PeekingIterator<LabeledTokensContainer> it = Iterators.peekingIterator(taggingTokenSynchronizer);
        if (!it.hasNext() || (it.peek() == null)) {
            return Collections.emptyList();
        }

        // a boolean is introduced to indicate the start of the sequence in the case the label
        // has no beginning indicator (e.g. I-)
        boolean begin = true;
        TaggingTokenCluster curCluster = new TaggingTokenCluster(it.peek().getTaggingLabel());
        BoundingBox curBox=null;
 
        
        
        while (it.hasNext()) {
            LabeledTokensContainer cont = it.next();
            BoundingBox b = BoundingBox.fromLayoutToken(cont.getLayoutTokens().get(0));
            if(!curCluster.concatTokens().isEmpty()){
                curBox = BoundingBox.fromLayoutToken(curCluster.concatTokens().get(0));
                if(b.distanceTo(curBox)>600){
                    curCluster = new TaggingTokenCluster(cont.getTaggingLabel());
                    result.add(curCluster);
                }
            }
            
            if (begin || cont.isBeginning() || cont.getTaggingLabel() != curCluster.getTaggingLabel()) {
                curCluster = new TaggingTokenCluster(cont.getTaggingLabel());
                result.add(curCluster);
            }
            
            //for table, seperate caption and content
            if(curCluster!=null){
                String tableStr = LayoutTokensUtil.normalizeText(curCluster.concatTokens());
                if(tableStr.matches(".*?(Table|TABLE) \\d+(:|\\.| [A-Z]).*?")){
//                if(tableStr.matches(".*?(Table|TABLE|Figure|FIGURE) \\d+(:|\\.).*?")){
                    if(toText(curCluster.getLastContainer().getLayoutTokens()).equalsIgnoreCase(". \n\n")){ 
                        curCluster = new TaggingTokenCluster(cont.getTaggingLabel());
                        result.add(curCluster);
                    }
                }
            }
                    
                    
            curCluster.addLabeledTokensContainer(cont);
            if (begin)
                begin = false;
        }

        return result;
    }

}
