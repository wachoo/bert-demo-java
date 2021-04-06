package com.wachoo.bert;

import com.wachoo.bert.util.TokenUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.NavigableMap;

/**
 * @Author : wachoo
 * @Time : 2021/4/1 15:58
 */
public class InputVector {

    private List<Integer> inputIds;
    private List<Integer> inputMask;
    private List<Integer> segmentIds;

    public InputVector(Builder builder) {
        this.inputIds = builder.inputIds;
        this.inputMask = builder.inputMask;
        this.segmentIds = builder.segmentIds;
    }

    public List<Integer> getInputIds() {
        return inputIds;
    }

    public List<Integer> getInputMask() {
        return inputMask;
    }

    public List<Integer> getSegmentIds() {
        return segmentIds;
    }

    public static class Builder {
        private List<Integer> inputIds;
        private List<Integer> inputMask = new ArrayList<>();
        private List<Integer> segmentIds = new ArrayList<>();

        public Builder(NavigableMap<String, Integer> vocab, List<String> tokenizerTokens, int maxSeqLength) {
            List<String> tokens = new ArrayList<>();
            if (tokenizerTokens.size() > maxSeqLength - 2) {
                tokenizerTokens = tokenizerTokens.subList(0, maxSeqLength - 2);
            }
            tokens.add("[CLS]");
            segmentIds.add(0);
            tokenizerTokens.forEach(e -> {
                tokens.add(e);
                segmentIds.add(0);
            });
            tokens.add("[SEP]");
            segmentIds.add(0);

            // convert_tokens_to_ids
            inputIds = TokenUtil.convertIdsByVocab(vocab, tokens);
            inputIds.forEach(e -> {
                inputMask.add(1);
            });
            while (inputIds.size() < maxSeqLength) {
                inputIds.add(0);
                inputMask.add(0);
                segmentIds.add(0);
            }
        }

        public InputVector build() {
            return new InputVector(this);
        }
    }
}
