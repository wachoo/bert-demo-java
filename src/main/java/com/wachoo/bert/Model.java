package com.wachoo.bert;

import com.google.common.base.Stopwatch;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.framework.ConfigProto;
import org.tensorflow.framework.GPUOptions;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * @Author : wangchao
 * @Time : 2021/4/8 21:04
 */
public class Model implements Cloneable{

    private String modelPath;
    private int hiddenSize;

    private String inputIds;
    private String inputMask;
    private String segmentIds;
    private String fetch;

    private Session session;


    public Model(Builder builder) {
        modelPath = builder.modelPath;
        hiddenSize = builder.hiddenSize;
        inputIds = builder.inputIds;
        inputMask = builder.inputMask;
        segmentIds = builder.segmentIds;
        fetch = builder.fetch;
        session = getModelBundle(modelPath).session();
    }

    public static Builder newBuilder() {
        return new Builder();
    }
    public static class Builder {
        private String modelPath;
        private int hiddenSize = 6;

        private String inputIds = "input_ids";
        private String inputMask = "input_mask";
        private String segmentIds = "segment_ids";
        private String fetch = "loss/Softmax";

        private Builder() {
        }

        public Model build() {
            return new Model(this);
        }

        public Builder modelPath(String modelPath) {
            this.modelPath = modelPath;
            return this;
        }
        public Builder hiddenSize(int hiddenSize) {
            this.hiddenSize = hiddenSize;
            return this;
        }
        public Builder inputIds(String inputIds) {
            this.inputIds = inputIds;
            return this;
        }
        public Builder inputMask(String inputMask) {
            this.inputMask = inputMask;
            return this;
        }
        public Builder segmentIds(String segmentIds) {
            this.segmentIds = segmentIds;
            return this;
        }
        public Builder fetch(String fetch) {
            this.fetch = fetch;
            return this;
        }
    }

    public float[] predict(Tensor<Integer> inputIds, Tensor<Integer> inputMask, Tensor<Integer> segmentIds) {
        try {
            Stopwatch stopwatch = Stopwatch.createStarted();
            List<Tensor<?>> tensors =  session.runner()
                    .feed(this.inputIds, inputIds)
                    .feed(this.inputMask, inputMask)
                    .feed(this.segmentIds, segmentIds)
                    .fetch(this.fetch)
                    .run();
            Tensor out = tensors.get(0);
            System.out.println(String.format("predicted, time cost %d ms", stopwatch.elapsed(TimeUnit.MILLISECONDS)));

            float[][] outArr = new float[1][hiddenSize];
            out.copyTo(outArr);
            return outArr[0];
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
    public void close() {
        session.close();
    }

    private SavedModelBundle getModelBundle(String modelPath) {
        ConfigProto configProto = getConfigProto();
        SavedModelBundle bundle = SavedModelBundle
                .loader(modelPath)
                .withConfigProto(configProto.toByteArray())
                .withTags("serve").load();
        return bundle;
    }

    protected ConfigProto getConfigProto() {
        Map<String, Integer> hashMap = new HashMap<>();
        hashMap.put("CPU", 4);
        hashMap.put("GPU", 4);
        ConfigProto configProto = ConfigProto.newBuilder()
                .putAllDeviceCount(hashMap)
                .setInterOpParallelismThreads(3)
                .setIntraOpParallelismThreads(3)
                .setAllowSoftPlacement(true)
                .setGpuOptions(GPUOptions.newBuilder()
                        .setAllowGrowth(true)
                        .setPerProcessGpuMemoryFraction(0.2)
                        .build())
                .build();
        return configProto;
    }
}
