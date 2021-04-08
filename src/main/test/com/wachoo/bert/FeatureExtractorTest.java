package com.wachoo.bert;


import org.junit.jupiter.api.Test;

class FeatureExtractorTest {

    @Test
    void execute() {
        String sentence = "床前明月光，疑是地上霜，剧透网民孤，低头思故乡";
        String modelPath = "D:\\workspace\\gitt\\bert-demo-java\\output\\cv_long_text_output_20210330\\savedmodel";
        int hiddenSize = 6;
        FeatureExtractor extractor = FeatureExtractor.newBuilder()
                .modelPath(modelPath)
                .sentence(sentence)
                .hiddenSize(hiddenSize)
                .build();
        float[] floats = extractor.execute();

        StringBuffer stringBuffer = new StringBuffer();
        for (int i = 0; i < floats.length; i++) {
            stringBuffer.append(floats[i]).append("  ");
        }
        System.out.println(stringBuffer.toString());
    }

    @Test
    void inputVectorBuild() {
        String sentence = "床前明月光，疑是地上霜，剧透网民孤，低头思故乡";
        InputVector inputVector = new FeatureExtractor().sentenceToVector(sentence, 128);
        System.out.println("inputVector.getInputIds() = " + inputVector.getInputIds().toString());
        System.out.println("inputVector.getInputMask() = " + inputVector.getInputMask().toString());
        System.out.println("inputVector.segmentIds = " + inputVector.getSegmentIds().toString());
    }


}