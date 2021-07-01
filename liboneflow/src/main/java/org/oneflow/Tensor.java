package org.oneflow;

public class Tensor {
    private long tensorPtr;
    private long[] shape;
    private DType dType;
    private Tensor() {}

    // --------------------------- Constructor Methods ---------------------------

    public static Tensor fromBlob(byte[] data, long[] shape, DType dType) {
        return null;
    }

    public static Tensor fromBlob(int[] data, long[] shape, DType dType) {
        return null;
    }

    public static Tensor fromBlob(long[] data, long[] shape, DType dType) {
        return null;
    }

    public static Tensor fromBlob(short[] data, long[] shape, DType dType) {
        return null;
    }

    public static Tensor fromBlob(float[] data, long[] shape, DType dType) {
        return null;
    }

    public static Tensor fromBlob(double[] data, long[] shape, DType dType) {
        return null;
    }

    // --------------------------- Get Information ---------------------------

    public long[] getShape() {
        return shape;
    }

    public DType getDataType() {
        return dType;
    }

    public byte[] getByteData() {
        return null;
    }

    public int[] getIntData() {
        return null;
    }

    public int[] getLongData() {
        return null;
    }

    public float[] getFloat16Data() {
        return null;
    }

    public float[] getFloatData() {
        return null;
    }

    public double[] getDoubleData() {
        return null;
    }

    public void free() {}

    @Override
    public String toString() {
        return "Tensor{" +
                "tensorPtr=" + tensorPtr +
                '}';
    }

    // --------------------------- Native Methods ---------------------------

    private native long newTensor(byte[] data, long[] shape, int dType);
    private native void free(long tensorPtr);
    private native byte[] getData();
}
