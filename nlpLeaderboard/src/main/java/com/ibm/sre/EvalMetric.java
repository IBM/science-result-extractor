/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.ibm.sre;

/**
 *
 * @author yhou
 */
public class EvalMetric {
        public String id;
        public String name;
        public String intro;
        public String indicator;//indicator=1, higher the better; indicator=0, lower the better

        public EvalMetric(String id, String name) {
            this.id = id;
            this.name = name;
        }

        public void setIntro(String intro) {
            this.intro = intro;
        }    
}
