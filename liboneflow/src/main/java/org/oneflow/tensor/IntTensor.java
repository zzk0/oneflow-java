package org.oneflow.tensor;

import org.oneflow.DType;
import org.oneflow.Tensor;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;

public class IntTensor extends Tensor {
    private final IntBuffer data;

    public IntTensor(long[] shape, IntBuffer data) {
        super(shape);
        this.data = data;
    }

    @Override
    public DType getDataType() {
        return DType.kInt32;
    }

    @Override
    public Buffer getRawDataBuffer() {
        return data;
    }

    @Override
    public int[] getDataAsIntArray() {
        data.rewind();
        int[] arr = new int[data.remaining()];
        data.get(arr);
        return arr;
    }

    @Override
    public byte[] getBytes() {
        ByteBuffer byteBuffer = ByteBuffer.allocate(data.capacity() * DType.kInt32.bytes);
        data.rewind();
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
        byteBuffer.asIntBuffer().put(data);
        return byteBuffer.array();
    }
}
