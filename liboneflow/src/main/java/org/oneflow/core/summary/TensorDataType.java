// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/summary/tensor.proto

package org.oneflow.core.summary;

/**
 * Protobuf enum {@code oneflow.summary.TensorDataType}
 */
public enum TensorDataType
    implements com.google.protobuf.ProtocolMessageEnum {
  /**
   * <code>DT_INVALID = 0;</code>
   */
  DT_INVALID(0),
  /**
   * <code>DT_FLOAT = 1;</code>
   */
  DT_FLOAT(1),
  /**
   * <code>DT_DOUBLE = 2;</code>
   */
  DT_DOUBLE(2),
  /**
   * <code>DT_INT32 = 3;</code>
   */
  DT_INT32(3),
  /**
   * <code>DT_UINT8 = 4;</code>
   */
  DT_UINT8(4),
  /**
   * <code>DT_INT16 = 5;</code>
   */
  DT_INT16(5),
  /**
   * <code>DT_INT8 = 6;</code>
   */
  DT_INT8(6),
  /**
   * <code>DT_STRING = 7;</code>
   */
  DT_STRING(7),
  /**
   * <code>DT_INT64 = 8;</code>
   */
  DT_INT64(8),
  /**
   * <code>DT_UINT16 = 9;</code>
   */
  DT_UINT16(9),
  /**
   * <code>DT_HALF = 10;</code>
   */
  DT_HALF(10),
  /**
   * <code>DT_UINT32 = 11;</code>
   */
  DT_UINT32(11),
  /**
   * <code>DT_UINT64 = 12;</code>
   */
  DT_UINT64(12),
  ;

  /**
   * <code>DT_INVALID = 0;</code>
   */
  public static final int DT_INVALID_VALUE = 0;
  /**
   * <code>DT_FLOAT = 1;</code>
   */
  public static final int DT_FLOAT_VALUE = 1;
  /**
   * <code>DT_DOUBLE = 2;</code>
   */
  public static final int DT_DOUBLE_VALUE = 2;
  /**
   * <code>DT_INT32 = 3;</code>
   */
  public static final int DT_INT32_VALUE = 3;
  /**
   * <code>DT_UINT8 = 4;</code>
   */
  public static final int DT_UINT8_VALUE = 4;
  /**
   * <code>DT_INT16 = 5;</code>
   */
  public static final int DT_INT16_VALUE = 5;
  /**
   * <code>DT_INT8 = 6;</code>
   */
  public static final int DT_INT8_VALUE = 6;
  /**
   * <code>DT_STRING = 7;</code>
   */
  public static final int DT_STRING_VALUE = 7;
  /**
   * <code>DT_INT64 = 8;</code>
   */
  public static final int DT_INT64_VALUE = 8;
  /**
   * <code>DT_UINT16 = 9;</code>
   */
  public static final int DT_UINT16_VALUE = 9;
  /**
   * <code>DT_HALF = 10;</code>
   */
  public static final int DT_HALF_VALUE = 10;
  /**
   * <code>DT_UINT32 = 11;</code>
   */
  public static final int DT_UINT32_VALUE = 11;
  /**
   * <code>DT_UINT64 = 12;</code>
   */
  public static final int DT_UINT64_VALUE = 12;


  public final int getNumber() {
    return value;
  }

  /**
   * @deprecated Use {@link #forNumber(int)} instead.
   */
  @java.lang.Deprecated
  public static TensorDataType valueOf(int value) {
    return forNumber(value);
  }

  public static TensorDataType forNumber(int value) {
    switch (value) {
      case 0: return DT_INVALID;
      case 1: return DT_FLOAT;
      case 2: return DT_DOUBLE;
      case 3: return DT_INT32;
      case 4: return DT_UINT8;
      case 5: return DT_INT16;
      case 6: return DT_INT8;
      case 7: return DT_STRING;
      case 8: return DT_INT64;
      case 9: return DT_UINT16;
      case 10: return DT_HALF;
      case 11: return DT_UINT32;
      case 12: return DT_UINT64;
      default: return null;
    }
  }

  public static com.google.protobuf.Internal.EnumLiteMap<TensorDataType>
      internalGetValueMap() {
    return internalValueMap;
  }
  private static final com.google.protobuf.Internal.EnumLiteMap<
      TensorDataType> internalValueMap =
        new com.google.protobuf.Internal.EnumLiteMap<TensorDataType>() {
          public TensorDataType findValueByNumber(int number) {
            return TensorDataType.forNumber(number);
          }
        };

  public final com.google.protobuf.Descriptors.EnumValueDescriptor
      getValueDescriptor() {
    return getDescriptor().getValues().get(ordinal());
  }
  public final com.google.protobuf.Descriptors.EnumDescriptor
      getDescriptorForType() {
    return getDescriptor();
  }
  public static final com.google.protobuf.Descriptors.EnumDescriptor
      getDescriptor() {
    return org.oneflow.core.summary.Tensor.getDescriptor()
        .getEnumTypes().get(0);
  }

  private static final TensorDataType[] VALUES = values();

  public static TensorDataType valueOf(
      com.google.protobuf.Descriptors.EnumValueDescriptor desc) {
    if (desc.getType() != getDescriptor()) {
      throw new java.lang.IllegalArgumentException(
        "EnumValueDescriptor is not for this type.");
    }
    return VALUES[desc.getIndex()];
  }

  private final int value;

  private TensorDataType(int value) {
    this.value = value;
  }

  // @@protoc_insertion_point(enum_scope:oneflow.summary.TensorDataType)
}
