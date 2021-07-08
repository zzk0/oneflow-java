package org.oneflow;


import org.oneflow.tensor.FloatTensor;
import org.oneflow.tensor.IntTensor;

import java.nio.*;

public abstract class Tensor {
    private final long[] shape;

    protected Tensor(long[] shape) {
        this.shape = shape;
    }

    // --------------------------- Constructor Methods ---------------------------

    public static Tensor fromBlob(int[] data, long[] shape) {
        final IntBuffer intBuffer = IntBuffer.allocate(data.length * DType.kInt32.bytes);
        intBuffer.put(data);
        return new IntTensor(shape, intBuffer);
    }

    public static Tensor fromBlob(float[] data, long[] shape) {
        final FloatBuffer floatBuffer = FloatBuffer.allocate(data.length * DType.kFloat.bytes);
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
