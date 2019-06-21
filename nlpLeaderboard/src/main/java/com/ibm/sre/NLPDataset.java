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
    public class NLPDataset {
        public String id;
        public String name;
        public String intro;
        public String language;

        public NLPDataset(String id, String name) {
            this.id = id;
            this.name = name;
        }

        public void setIntro(String intro) {
            this.intro = intro;
        }
    }