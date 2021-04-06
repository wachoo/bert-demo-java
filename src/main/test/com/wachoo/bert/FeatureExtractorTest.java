package com.wachoo.bert;

import org.junit.jupiter.api.Test;

class FeatureExtractorTest {

    @Test
    void execute() {
        String sentence = "床前明月光，疑是地上霜，剧透网民孤，低头思故乡";
        String modelPath = "D:\\workspace\\gitt\\bert-demo-java\\output\\cv_long_text_output_20210330\\savedmodel";
        float[] floats = new FeatureExtractor().execute(modelPath, sentence);

        StringBuffer stringBuffer = new StringBuffer();
        for (int i = 0; i < floats.length; i++) {
            stringBuffer.append(floats[i]).append("  ");
        }
        System.out.println(stringBuffer.toString());
    }

    @Test
    void inputVectorBuild() {
//        String tokens = "[CLS] 目 前 是 销 售 冠 军 ， 销 售 额 为 公 司 总 销 售 额 的 1 / 3 。 （ 2019 年 也 是 销 售 冠 军 ） ， ， 2016 - 至 今 ， 本 人 成 功 开 拓 了 20 多 个 长 期 客 户 ， 其 中 10 个 是 大 客 户 。 ， ， * 轨 道 衡 行 业 龙 头 企 业 ， * 轨 道 衡 行 业 第 二 企 业 ， * 印 染 机 械 行 业 龙 头 企 业 ， * 环 保 行 业 龙 头 企 业 ， * 失 重 秤 行 业 前 三 的 企 业 ， * 医 药 机 [SEP]";
        String sentences = "0\t目前是销售冠军，销售额为公司总销售额的1/3。（2019年也是销售冠军），，2016-至今，本人成功开拓了20多个长期客户，其中10个是大客户。，，*轨道衡行业龙头企业，*轨道衡行业第二企业，*印染机械行业龙头企业，*环保行业龙头企业，*失重秤行业前三的企业，*医药机械行业前三的企业，*某大型医药集团1，*某大型医药集团2";
        InputVector inputVector = new FeatureExtractor().sentenceToVector(sentences, 128);
        System.out.println("inputVector.getInputIds() = " + inputVector.getInputIds().toString());
        System.out.println("inputVector.getInputMask() = " + inputVector.getInputMask().toString());
        System.out.println("inputVector.segmentIds = " + inputVector.getSegmentIds().toString());
    }


}