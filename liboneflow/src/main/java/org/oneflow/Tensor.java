package org.oneflow;


import org.oneflow.tensor.FloatTensor;
import org.oneflow.tensor.IntTensor;

import java.nio.*;

public abstract class Tensor {
    private final long[] shape;

    protected Tensor(long[] shape) {
        this.shape = shape;
    }

    public static Tensor fromBlob(int[] data, long[] shape) {
        final IntBuffer intBuffer = ByteBuffer.allocateDirect(data.length * DType.kInt32.bytes)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asIntBuffer();
        intBuffer.put(data);
        return new IntTensor(shape, intBuffer);
    }

    public static Tensor fromBlob(float[] data, long[] shape) {
        final FloatBuffer floatBuffer = ByteBuffer.allocateDirect(data.length * DType.kFloat.bytes)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer();
        floatBuffer.put(data);
        return new FloatTensor(shape, floatBuffer);
    }

    public long[] getShape() {
        return shape;
    }

    public abstract DType getDataType();

    public abstract Buffer getDataBuffer();

    public boolean[] getDataAsBooleanArray() {
        throw new IllegalStateException(getClass().getSimpleName() +
                " cannot return data as boolean array");
    }

    public byte[] getDataAsByteArray() {
        throw new IllegalStateException(getClass().getSimpleName() +
                " cannot return data as byte array");
    }

    public short[] getDataAsShortArray() {
        throw new IllegalStateException(getClass().getSimpleName() +
                " cannot return data as short array");
    }

    public int[] getDataAsIntArray() {
        throw new IllegalStateException(getClass().getSimpleName() +
                " cannot return data as int array");
    }

    public long[] getDataAsLongArray() {
        throw new IllegalStateException(getClass().getSimpleName() +
                " cannot return data as long array");
    }

    public float[] getDataAsFloatArray() {
        throw new IllegalStateException(getClass().getSimpleName() +
                " cannot return data as float array");
    }

    public double[] getDataAsDoubleArray() {
        throw new IllegalStateException(getClass().getSimpleName() +
                " cannot return data as double array");
    }

    /**
     * This function will be called from native code, so when the function
     * signature changed, you need to changed the native code too
     */
    public static Tensor nativeNewTensor(byte[] data, long[] shape, int dType) {
        Tensor tensor = null;
        ByteBuffer byteBuffer = ByteBuffer.wrap(data);

        if (DType.kFloat.code == dType) {
            tensor = new FloatTensor(shape, byteBuffer.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer());
        }
        else if (DType.kInt32.code == dType) {
            tensor = new IntTensor(shape, byteBuffer.order(ByteOrder.LITTLE_ENDIAN).asIntBuffer());
        }

        return tensor;
    }
}
