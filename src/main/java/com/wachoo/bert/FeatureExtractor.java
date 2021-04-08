package com.wachoo.bert;

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.base.Stopwatch;
import com.wachoo.bert.util.TokenUtil;
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
    private String sentence = "ABC";
    private Integer seqLength = 128;
    private Model model;
    private String vocabPath;

    public FeatureExtractor() {
    }

    public FeatureExtractor(Model model, String sentence) {
        this.sentence = sentence;
        this.model = model;
        this.vocabPath = FeatureExtractor.class.getClassLoader().getResource("vocab.txt").getPath();

    }

    public void setSeqLength(Integer seqLength) {
        this.seqLength = seqLength;
    }

    /**
     * execution method
     * @return
     */
    public float[] execute() {
        Stopwatch stopwatch = Stopwatch.createStarted();
        InputVector inputVector = this.sentenceToVector(sentence, seqLength);
        System.out.println(String.format("sentenceToVector, time cost %d ms", stopwatch.elapsed(TimeUnit.MILLISECONDS)));

        Tensor<Integer> inputIds = listToTensor(inputVector.getInputIds(), seqLength);
        Tensor<Integer> inputMask = listToTensor(inputVector.getInputMask(), seqLength);
        Tensor<Integer> segmentIds = listToTensor(inputVector.getSegmentIds(), seqLength);
        System.out.println(String.format("listToTensor, time cost %d ms", stopwatch.elapsed(TimeUnit.MILLISECONDS)));

        return model.predict(inputIds, inputMask, segmentIds);
    }

    public InputVector sentenceToVector(String sentence, int maxSeqLength) {
        NavigableMap<String, Integer> navMap = new TreeMap<>();
        TokenUtil.readAllLinesToMap(this.vocabPath, navMap);
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
