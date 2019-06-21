/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.ibm.sre.pdfparser;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import java.util.ArrayList;
import java.util.List;
import org.grobid.core.engines.label.TaggingLabel;
import org.grobid.core.layout.LayoutToken;
import org.grobid.core.tokenization.LabeledTokensContainer;

/**
 *
 * @author yhou
 * add set labeling method 
 */
public class TaggingTokenCluster {
    public static final Function<LabeledTokensContainer, String> CONTAINERS_TO_FEATURE_BLOCK = new Function<LabeledTokensContainer, String>() {
        @Override
        public String apply(LabeledTokensContainer labeledTokensContainer) {
            if (labeledTokensContainer == null) {
                return "\n";
            }

            if (labeledTokensContainer.getFeatureString() == null) {
                throw new IllegalStateException("This method must be called when feature string is not empty for " +
                        "LabeledTokenContainers");
            }
            return labeledTokensContainer.getFeatureString();
        }
    };
    private List<LabeledTokensContainer> labeledTokensContainers = new ArrayList<>();
    private TaggingLabel taggingLabel;

    public TaggingTokenCluster(TaggingLabel taggingLabel) {
        this.taggingLabel = taggingLabel;
    }

    public void addLabeledTokensContainer(LabeledTokensContainer cont) {
        labeledTokensContainers.add(cont);
    }

    public List<LabeledTokensContainer> getLabeledTokensContainers() {
        return labeledTokensContainers;
    }

    public void setTaggingLabel(TaggingLabel label) {
        this.taggingLabel = label;
    }
    
    public TaggingLabel getTaggingLabel() {
        return taggingLabel;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (LabeledTokensContainer c : labeledTokensContainers) {
            sb.append(c).append("\n");
        }
        sb.append("\n");
        return sb.toString();
    }

    public LabeledTokensContainer getLastContainer() {
        if (labeledTokensContainers.isEmpty()) {
            return null;
        }

        return labeledTokensContainers.get(labeledTokensContainers.size() - 1);
    }

    public List<LayoutToken> concatTokens() {

        Iterable<LayoutToken> it = Iterables.concat(Iterables.transform(labeledTokensContainers, new Function<LabeledTokensContainer, List<LayoutToken>>() {
            @Override
            public List<LayoutToken> apply(LabeledTokensContainer labeledTokensContainer) {
                return labeledTokensContainer.getLayoutTokens();
            }
        }));
        return Lists.newArrayList(it);
    }

    public String getFeatureBlock() {
        return Joiner.on("\n").join(Iterables.transform(labeledTokensContainers, CONTAINERS_TO_FEATURE_BLOCK));
    }
}
