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
import org.tensorflow.framework.ConfigProto;
import org.tensorflow.framework.GPUOptions;

import java.util.*;
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
    public float[] execute(String modelPath, String sentence) {
        int seqLength = 128;
        int hiddenSize = 6;

        Stopwatch stopwatch = Stopwatch.createStarted();
        InputVector inputVector = this.sentenceToVector(sentence, seqLength);
        System.out.println(String.format("sentenceToVector, time cost %d ms", stopwatch.elapsed(TimeUnit.MILLISECONDS)));

        Tensor<Integer> inputIds = listToTensor(inputVector.getInputIds(), seqLength);
        Tensor<Integer> inputMask = listToTensor(inputVector.getInputMask(), seqLength);
        Tensor<Integer> segmentIds = listToTensor(inputVector.getSegmentIds(), seqLength);
        System.out.println(String.format("listToTensor, time cost %d ms", stopwatch.elapsed(TimeUnit.MILLISECONDS)));


        return this.predict(modelPath, hiddenSize, inputIds, inputMask, segmentIds);
    }

    private float[] predict(String modelPath, int hiddenSize, Tensor<Integer> inputIds, Tensor<Integer> inputMask, Tensor<Integer> segmentIds) {
        SavedModelBundle bundle = getModelBundle(modelPath);
        try (Session sess = bundle.session()) {
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

    private SavedModelBundle getModelBundle(String modelPath) {
        Map<String, Integer> hashMap = new HashMap<>();
        hashMap.put("CPU", 4);
//        hashMap.put("GPU", 1);
        ConfigProto configProto = ConfigProto.newBuilder()
                .putAllDeviceCount(hashMap)
                .setInterOpParallelismThreads(3)
                .setIntraOpParallelismThreads(3)
                .setAllowSoftPlacement(true)
                .setGpuOptions(GPUOptions.newBuilder()
                        .setAllowGrowth(true)
                        .setPerProcessGpuMemoryFraction(0.9)
                        .build())
                .build();
        SavedModelBundle bundle = SavedModelBundle
                .loader(modelPath)
                .withConfigProto(configProto.toByteArray())
                .withTags("serve").load();
        return bundle;
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

    private static Tensor<Integer> listToTensor(List<Integer> input, int length) {
        int[] arr = input.stream()
                .mapToInt(x -> Integer.valueOf(x))
                .toArray();
        Preconditions.checkArgument(length == arr.length);
        Tensor<Integer> tensor = Tensors.create(new int[][]{arr});
        return tensor;
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
