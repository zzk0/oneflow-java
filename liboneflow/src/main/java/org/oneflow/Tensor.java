package org.oneflow;


import org.oneflow.tensor.FloatTensor;
import org.oneflow.tensor.IntTensor;

import java.nio.Buffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

public abstract class Tensor {
    private final long[] shape;

    protected Tensor(long[] shape) {
        this.shape = shape;
    }

    // --------------------------- Constructor Methods ---------------------------

    public static Tensor fromBlob(int[] data, long[] shape) {
        // Todo: replace Integer.BYTES using enum DType
        final IntBuffer intBuffer = IntBuffer.allocate(data.length * Integer.BYTES);
        intBuffer.put(data);
        return new IntTensor(shape, intBuffer);
    }

    public static Tensor fromBlob(float[] data, long[] shape) {
        final FloatBuffer floatBuffer = FloatBuffer.allocate(data.length * Float.BYTES);
        floatBuffer.put(data);
        return new FloatTensor(shape, floatBuffer);
    }

    // --------------------------- Get Information ---------------------------

    public long[] getShape() {
        return shape;
    }

    public abstract DType getDataType();

    public abstract Buffer getRawDataBuffer();

    public abstract byte[] getBytes();

    public int[] getDataAsIntArray() {
        throw new IllegalStateException(getClass().getSimpleName() +
                " cannot return data as int array");
    }

    public float[] getDataAsFloatArray(){
        throw new IllegalStateException(getClass().getSimpleName() +
                " cannot return data as float array");
    }
}
