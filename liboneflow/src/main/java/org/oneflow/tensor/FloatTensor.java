package org.oneflow.tensor;

import org.oneflow.DType;
import org.oneflow.Tensor;

import java.nio.Buffer;
import java.nio.FloatBuffer;

public class FloatTensor extends Tensor {
    private final FloatBuffer data;

    public FloatTensor(long[] shape, FloatBuffer data) {
        super(shape);
        this.data = data;
    }

    @Override
    public DType getDataType() {
        return DType.FLOAT;
    }

    @Override
    public Buffer getRawDataBuffer() {
        return data;
    }

    @Override
    public float[] getDataAsFloatArray() {
        data.rewind();
        float[] arr = new float[data.remaining()];
        data.get(arr);
        return arr;
    }
}
