package org.oneflow;

public enum DType {
    BOOL(10),
    UINT8(11),
    INT8(12),
    INT(13),
    INT32(14),
    INT64(15),
    LONG(16),

    FLOAT16(20),
    HALF(21),
    FLOAT(22),
    FLOAT32(23),
    FLOAT64(24),
    DOUBLE(25);

    final int code;

    DType(int code) {
        this.code = code;
    }
}
