package com.wachoo.bert;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.base.Stopwatch;
import com.wachoo.bert.util.TokenUtil;
import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.util.List;
import java.util.NavigableMap;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;


/**
 * @Author : wachoo
 * @Time : 2021/4/2 10:01
 */
public class FeatureExtractor {

    /**
     * execution method
     * @param modelPath
     * @param sentence
     * @return
     */
    public static float[] execute(String modelPath, String sentence) {
        int seqLength = 128;
        int hiddenSize = 6;

        FeatureExtractor featureExtractor = new FeatureExtractor();
        InputVector inputVector = featureExtractor.sentenceToVector(sentence, seqLength);
        String strInputIds = StringUtils.join(inputVector.getInputIds(), ",");
        String strInputMask = StringUtils.join(inputVector.getInputMask(), ",");
        String strSegmentIds = StringUtils.join(inputVector.getSegmentIds(), ",");
        System.out.println("inputVector.getInputIds() = " + strInputIds);
        System.out.println("inputVector.getInputMask() = " + strInputMask);
        System.out.println("inputVector.segmentIds = " + strSegmentIds);

        Tensor<Integer> inputIds = fromStringToTensor(strInputIds, seqLength);
        Tensor<Integer> inputMask = fromStringToTensor(strInputMask, seqLength);
        Tensor<Integer> segmentIds = fromStringToTensor(strSegmentIds, seqLength);

        return featureExtractor.predict(modelPath, hiddenSize, inputIds, inputMask, segmentIds);
    }

    private float[] predict(String modelPath, int hiddenSize, Tensor<Integer> inputIds, Tensor<Integer> inputMask, Tensor<Integer> segmentIds) {
        try (Session sess = SavedModelBundle.load(modelPath, "serve").session()) {
            Stopwatch stopwatch = Stopwatch.createStarted();
            List<Tensor<?>> tensors = sess.runner()
                    .feed("input_ids", inputIds)
                    .feed("input_mask", inputMask)
                    .feed("segment_ids", segmentIds)
                    .fetch("loss/Softmax")
                    .run();
            Tensor out = tensors.get(0);
            System.out.println(String.format("predicted, time cost %d ms", stopwatch.elapsed(TimeUnit.MILLISECONDS)));

            float[][] outArr = new float[1][hiddenSize];
            out.copyTo(outArr);
            return outArr[0];
        }
    }

    public InputVector sentenceToVector(String sentence, int maxSeqLength) {
        String vocalPath = this.getClass().getClassLoader().getResource("vocab.txt").getPath();
        NavigableMap<String, Integer> navMap = new TreeMap<>();
        TokenUtil.readAllLinesToMap(vocalPath, navMap);
        NavigableMap<String, Integer> vocab = navMap.descendingMap();

        TokenPreProcess preTokenizePreProcessor = new CVWordPiecePreProcessor(true, true, vocab);
        TokenPreProcess tokenPreProcess = new CVWordPiecePreProcessor(true, true, vocab);
        CVBertWordPieceTokenizer pieceTokenizer = new CVBertWordPieceTokenizer(sentence, vocab, preTokenizePreProcessor, tokenPreProcess);
        List<String> tokenizerTokens = pieceTokenizer.getTokens();

        InputVector inputVector = new InputVector.Builder(vocab, tokenizerTokens, maxSeqLength).build();
        return inputVector;
    }

    private static Tensor<Integer> fromStringToTensor(String input, int length) {
        int[] arr = Splitter.on(',')
                .trimResults().omitEmptyStrings().splitToList(input).stream()
                .mapToInt(x -> Integer.valueOf(x))
                .toArray();
        Preconditions.checkArgument(length == arr.length);
        Tensor<Integer> tensor = Tensors.create(new int[][]{arr});
        return tensor;
    }
}
