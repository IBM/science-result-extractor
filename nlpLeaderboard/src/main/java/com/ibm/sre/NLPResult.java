/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.ibm.sre;

import java.util.Objects;

/**
 *
 * @author yhou
 */
public class NLPResult {

    public String paperName;
    public String taskName;
    public String datasetName;
    public String evaluationMetric;
    public String score;

    public NLPResult(String paper, String task, String dataset) {
        this.paperName = paper;
        this.taskName = task;
        this.datasetName = dataset;
        this.evaluationMetric = "";
        this.score = "";
    }

    public void setEvaluationMetric(String evaluationMetric) {
        this.evaluationMetric = evaluationMetric;
    }

    public void setEvaluationScore(String score) {
        this.score = score;
    }

//    @Override
//    public int hashCode() {
//        int hash = 7;
//        hash = 97 * hash + Objects.hashCode(this.taskName);
//        return hash;
//    }
//       @Override
//    public int hashCode() {
//        int hash = 7;
//        hash = 97 * hash + Objects.hashCode(this.taskName);
//        hash = 97 * hash + Objects.hashCode(this.datasetName);
//        hash = 97 * hash + Objects.hashCode(this.evaluationMetric);
//        return hash;
//    }
    @Override
    public String toString() {
        return "NLPResult{" + "paperName=" + paperName + ", taskName=" + taskName + ", datasetName=" + datasetName + ", evaluationMetric=" + evaluationMetric + ", score=" + score + '}';
    }

//    @Override
//    public boolean equals (Object obj){
//        if(this==obj)
//            return true;
//        if(obj==null||obj.getClass()!=this.getClass())
//            return false;
//        NLPResult result = (NLPResult) obj;
//        if(this.evaluationMetric.isEmpty()||result.evaluationMetric.isEmpty())
//            return false;
//        return (this.taskName.equalsIgnoreCase(result.taskName)
//                &&(this.datasetName.toLowerCase().contains(result.datasetName.toLowerCase())||result.datasetName.toLowerCase().contains(this.datasetName.toLowerCase()))
//                &&(this.evaluationMetric.toLowerCase().contains(result.evaluationMetric.toLowerCase())||result.evaluationMetric.toLowerCase().contains(this.evaluationMetric.toLowerCase())));
//    }
//    public boolean equals_relax (Object obj){
//        if(this==obj)
//            return true;
//        if(obj==null||obj.getClass()!=this.getClass())
//            return false;
//        NLPResult result = (NLPResult) obj;
//        if(this.evaluationMetric.isEmpty()||result.evaluationMetric.isEmpty())
//            return false;
//        return (this.taskName.equalsIgnoreCase(result.taskName)
//                &&(this.datasetName.toLowerCase().contains(result.datasetName.toLowerCase())||result.datasetName.toLowerCase().contains(this.datasetName.toLowerCase()))
//                &&(this.evaluationMetric.toLowerCase().contains(result.evaluationMetric.toLowerCase())||result.evaluationMetric.toLowerCase().contains(this.evaluationMetric.toLowerCase())));
//    }
    public boolean equals_relax(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || obj.getClass() != this.getClass()) {
            return false;
        }
        NLPResult result = (NLPResult) obj;
        if (this.evaluationMetric.isEmpty() || result.evaluationMetric.isEmpty()) {
            return false;
        }
        return (this.taskName.equalsIgnoreCase(result.taskName)
                && this.datasetName.equalsIgnoreCase(result.datasetName)
                && this.evaluationMetric.equalsIgnoreCase(result.evaluationMetric));
    }

    public boolean equals_strict(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null || obj.getClass() != this.getClass()) {
            return false;
        }
        NLPResult result = (NLPResult) obj;
        if (this.evaluationMetric.isEmpty() || result.evaluationMetric.isEmpty()) {
            return false;
        }
        if (this.score.isEmpty() || result.score.isEmpty()) {
            return false;
        }

        if (this.taskName.equalsIgnoreCase(result.taskName)
                && this.datasetName.equalsIgnoreCase(result.datasetName)
                && this.evaluationMetric.equalsIgnoreCase(result.evaluationMetric)) {
            if (this.score.matches("\\d+|\\d+\\.\\d+") && result.score.matches("\\d+|\\d+\\.\\d+")) {
                double d1 = Double.valueOf(this.score);
                double d2 = Double.valueOf(result.score);
                if (d1 == d2) {
                    return true;
                }
            } else if (this.score.equalsIgnoreCase(result.score)) {
                return true;
            }

        }
        return false;
    }

}
