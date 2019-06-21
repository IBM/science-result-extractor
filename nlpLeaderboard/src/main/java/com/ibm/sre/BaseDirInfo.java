/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.ibm.sre;

import java.io.File;

/**
 *
 * @author yhou
 */
public class BaseDirInfo {

   public static String getBaseDir() {
        return new File("").getAbsolutePath() + "/";
    }
    
    public static String getPath(String path) {
        return new File("").getAbsolutePath() + "/" + path;
    }

 
    public static void main(String[] args){ 
        System.err.println(BaseDirInfo.getBaseDir());
    }
        
    
}
