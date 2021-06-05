// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/initializer_conf.proto

package org.oneflow.core.job;

/**
 * Protobuf enum {@code oneflow.RandomDistribution}
 */
public enum RandomDistribution
    implements com.google.protobuf.ProtocolMessageEnum {
  /**
   * <code>kRandomUniform = 0;</code>
   */
  kRandomUniform(0),
  /**
   * <code>kRandomNormal = 1;</code>
   */
  kRandomNormal(1),
  /**
   * <code>kTruncatedNormal = 2;</code>
   */
  kTruncatedNormal(2),
  ;

  /**
   * <code>kRandomUniform = 0;</code>
   */
  public static final int kRandomUniform_VALUE = 0;
  /**
   * <code>kRandomNormal = 1;</code>
   */
  public static final int kRandomNormal_VALUE = 1;
  /**
   * <code>kTruncatedNormal = 2;</code>
   */
  public static final int kTruncatedNormal_VALUE = 2;


  public final int getNumber() {
    return value;
  }

  /**
   * @deprecated Use {@link #forNumber(int)} instead.
   */
  @java.lang.Deprecated
  public static RandomDistribution valueOf(int value) {
    return forNumber(value);
  }

  public static RandomDistribution forNumber(int value) {
    switch (value) {
      case 0: return kRandomUniform;
      case 1: return kRandomNormal;
      case 2: return kTruncatedNormal;
      default: return null;
    }
  }

  public static com.google.protobuf.Internal.EnumLiteMap<RandomDistribution>
      internalGetValueMap() {
    return internalValueMap;
  }
  private static final com.google.protobuf.Internal.EnumLiteMap<
      RandomDistribution> internalValueMap =
        new com.google.protobuf.Internal.EnumLiteMap<RandomDistribution>() {
          public RandomDistribution findValueByNumber(int number) {
            return RandomDistribution.forNumber(number);
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
    return org.oneflow.core.job.InitializerConfOuterClass.getDescriptor()
        .getEnumTypes().get(1);
  }

  private static final RandomDistribution[] VALUES = values();

  public static RandomDistribution valueOf(
      com.google.protobuf.Descriptors.EnumValueDescriptor desc) {
    if (desc.getType() != getDescriptor()) {
      throw new java.lang.IllegalArgumentException(
        "EnumValueDescriptor is not for this type.");
    }
    return VALUES[desc.getIndex()];
  }

  private final int value;

  private RandomDistribution(int value) {
    this.value = value;
  }

  // @@protoc_insertion_point(enum_scope:oneflow.RandomDistribution)
}

