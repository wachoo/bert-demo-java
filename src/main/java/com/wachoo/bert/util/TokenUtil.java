package com.wachoo.bert.util;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

/**
 * @Author : wachoo
 * @Time : 2021/4/1 11:34
 */
public class TokenUtil {

    public static void readAllLinesToMap(String filePath, SortedMap<String, Integer> map) {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line = reader.readLine();
            int idx = 0;
            while (line != null) {
                map.put(org.apache.commons.lang3.StringUtils.strip(line), idx++);
                line = reader.readLine();
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Converts a sequence of [tokens|ids] using the vocab.
     * @return
     */
    public static List<Integer> convertIdsByVocab(SortedMap<String, Integer> vocab, List<String> items){
        List<Integer> rst = new LinkedList<>();
        items.forEach(e ->{
            rst.add(vocab.get(e));
        });
        return rst;
    }
}
