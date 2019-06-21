package com.ibm.sre.pdfparser;


import org.grobid.core.GrobidModels;
import org.grobid.core.data.Figure;
import org.grobid.core.engines.label.TaggingLabel;
import org.grobid.core.engines.tagging.GenericTaggerUtils;
import org.grobid.core.exceptions.GrobidException;
import org.grobid.core.layout.LayoutToken;
import org.grobid.core.tokenization.TaggingTokenCluster;
import org.grobid.core.tokenization.TaggingTokenClusteror;
import org.grobid.core.utilities.LayoutTokensUtil;
import org.grobid.core.utilities.Pair;
import org.grobid.core.utilities.TextUtilities;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import org.grobid.core.engines.Engine;

import static org.grobid.core.engines.label.TaggingLabels.*;

/**
 * @author Patrice
 */
class NLPLeaderboardFigParser extends org.grobid.core.engines.AbstractParser {

    NLPLeaderboardFigParser() {
        super(GrobidModels.FIGURE);
    }

    /**
     * The processing here is called from the full text parser in cascade.
     * Start and end position in the higher level tokenization are indicated in
     * the resulting Figure object.
     */
    public Figure processing(List<LayoutToken> tokenizationFigure, String featureVector) {

        String res;
        try {
            res = label(featureVector);
        } catch (Exception e) {
            throw new GrobidException("CRF labeling in ReferenceSegmenter fails.", e);
        }
        if (res == null) {
            return null;
        }
//        List<Pair<String, String>> labeled = GenericTaggerUtils.getTokensAndLabels(res);

//		System.out.println(Joiner.on("\n").join(labeled));
//		System.out.println("----------------------");
//		System.out.println("----------------------");

//		return getExtractionResult(tokenizationFigure, labeled);
        return getExtractionResult(tokenizationFigure, res);
    }

    private Figure getExtractionResult(List<LayoutToken> tokenizations, String result) {
        TaggingTokenClusteror clusteror = new TaggingTokenClusteror(GrobidModels.FIGURE, result, tokenizations);
        List<TaggingTokenCluster> clusters = clusteror.cluster();
        
        Figure figure = new Figure();
        figure.setLayoutTokens(tokenizations);
        
        for (TaggingTokenCluster cluster : clusters) {
            if (cluster == null) {
                continue;
            }

            TaggingLabel clusterLabel = cluster.getTaggingLabel();
            Engine.getCntManager().i(clusterLabel);

            String clusterContent = LayoutTokensUtil.normalizeText(LayoutTokensUtil.toText(cluster.concatTokens()));
            if (clusterLabel.equals(FIG_DESC)) {
                figure.appendCaption(clusterContent);
                figure.appendCaptionLayoutTokens(cluster.concatTokens());
            } else if (clusterLabel.equals(FIG_HEAD)) {
                figure.appendHeader(clusterContent);
            } else if (clusterLabel.equals(FIG_LABEL)) {
                figure.appendLabel(clusterContent);
                //label should also go to head
                figure.appendHeader(" " + clusterContent + " ");
            } else if (clusterLabel.equals(FIG_OTHER)) {

            } else if (clusterLabel.equals(FIG_CONTENT)) {
                figure.appendContent(clusterContent);
            } else {
               }
        }
        return figure;
    }



    

}
