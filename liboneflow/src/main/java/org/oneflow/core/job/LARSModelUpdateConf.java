// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/job_conf.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.LARSModelUpdateConf}
 */
public  final class LARSModelUpdateConf extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.LARSModelUpdateConf)
    LARSModelUpdateConfOrBuilder {
  // Use LARSModelUpdateConf.newBuilder() to construct.
  private LARSModelUpdateConf(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private LARSModelUpdateConf() {
    momentumBeta_ = 0.9F;
    epsilon_ = 1e-09F;
    larsCoefficient_ = 0.0001F;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private LARSModelUpdateConf(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    this();
    int mutable_bitField0_ = 0;
    com.google.protobuf.UnknownFieldSet.Builder unknownFields =
        com.google.protobuf.UnknownFieldSet.newBuilder();
    try {
      boolean done = false;
      while (!done) {
        int tag = input.readTag();
        switch (tag) {
          case 0:
            done = true;
            break;
          default: {
            if (!parseUnknownField(input, unknownFields,
                                   extensionRegistry, tag)) {
              done = true;
            }
            break;
          }
          case 13: {
            bitField0_ |= 0x00000001;
            momentumBeta_ = input.readFloat();
            break;
          }
          case 21: {
            bitField0_ |= 0x00000002;
            epsilon_ = input.readFloat();
            break;
          }
          case 29: {
            bitField0_ |= 0x00000004;
            larsCoefficient_ = input.readFloat();
            break;
          }
        }
      }
    } catch (com.google.protobuf.InvalidProtocolBufferException e) {
      throw e.setUnfinishedMessage(this);
    } catch (java.io.IOException e) {
      throw new com.google.protobuf.InvalidProtocolBufferException(
          e).setUnfinishedMessage(this);
    } finally {
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.oneflow.core.job.JobConf.internal_static_oneflow_LARSModelUpdateConf_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.JobConf.internal_static_oneflow_LARSModelUpdateConf_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.LARSModelUpdateConf.class, org.oneflow.core.job.LARSModelUpdateConf.Builder.class);
  }

  private int bitField0_;
  public static final int MOMENTUM_BETA_FIELD_NUMBER = 1;
  private float momentumBeta_;
  /**
   * <code>optional float momentum_beta = 1 [default = 0.9];</code>
   */
  public boolean hasMomentumBeta() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>optional float momentum_beta = 1 [default = 0.9];</code>
   */
  public float getMomentumBeta() {
    return momentumBeta_;
  }

  public static final int EPSILON_FIELD_NUMBER = 2;
  private float epsilon_;
  /**
   * <code>optional float epsilon = 2 [default = 1e-09];</code>
   */
  public boolean hasEpsilon() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>optional float epsilon = 2 [default = 1e-09];</code>
   */
  public float getEpsilon() {
    return epsilon_;
  }

  public static final int LARS_COEFFICIENT_FIELD_NUMBER = 3;
  private float larsCoefficient_;
  /**
   * <code>optional float lars_coefficient = 3 [default = 0.0001];</code>
   */
  public boolean hasLarsCoefficient() {
    return ((bitField0_ & 0x00000004) == 0x00000004);
  }
  /**
   * <code>optional float lars_coefficient = 3 [default = 0.0001];</code>
   */
  public float getLarsCoefficient() {
    return larsCoefficient_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      output.writeFloat(1, momentumBeta_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      output.writeFloat(2, epsilon_);
    }
    if (((bitField0_ & 0x00000004) == 0x00000004)) {
      output.writeFloat(3, larsCoefficient_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.CodedOutputStream
        .computeFloatSize(1, momentumBeta_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.CodedOutputStream
        .computeFloatSize(2, epsilon_);
    }
    if (((bitField0_ & 0x00000004) == 0x00000004)) {
      size += com.google.protobuf.CodedOutputStream
        .computeFloatSize(3, larsCoefficient_);
    }
    size += unknownFields.getSerializedSize();
    memoizedSize = size;
    return size;
  }

  private static final long serialVersionUID = 0L;
  @java.lang.Override
  public boolean equals(final java.lang.Object obj) {
    if (obj == this) {
     return true;
    }
    if (!(obj instanceof org.oneflow.core.job.LARSModelUpdateConf)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.LARSModelUpdateConf other = (org.oneflow.core.job.LARSModelUpdateConf) obj;

    boolean result = true;
    result = result && (hasMomentumBeta() == other.hasMomentumBeta());
    if (hasMomentumBeta()) {
      result = result && (
          java.lang.Float.floatToIntBits(getMomentumBeta())
          == java.lang.Float.floatToIntBits(
              other.getMomentumBeta()));
    }
    result = result && (hasEpsilon() == other.hasEpsilon());
    if (hasEpsilon()) {
      result = result && (
          java.lang.Float.floatToIntBits(getEpsilon())
          == java.lang.Float.floatToIntBits(
              other.getEpsilon()));
    }
    result = result && (hasLarsCoefficient() == other.hasLarsCoefficient());
    if (hasLarsCoefficient()) {
      result = result && (
          java.lang.Float.floatToIntBits(getLarsCoefficient())
          == java.lang.Float.floatToIntBits(
              other.getLarsCoefficient()));
    }
    result = result && unknownFields.equals(other.unknownFields);
    return result;
  }

  @java.lang.Override
  public int hashCode() {
    if (memoizedHashCode != 0) {
      return memoizedHashCode;
    }
    int hash = 41;
    hash = (19 * hash) + getDescriptorForType().hashCode();
    if (hasMomentumBeta()) {
      hash = (37 * hash) + MOMENTUM_BETA_FIELD_NUMBER;
      hash = (53 * hash) + java.lang.Float.floatToIntBits(
          getMomentumBeta());
    }
    if (hasEpsilon()) {
      hash = (37 * hash) + EPSILON_FIELD_NUMBER;
      hash = (53 * hash) + java.lang.Float.floatToIntBits(
          getEpsilon());
    }
    if (hasLarsCoefficient()) {
      hash = (37 * hash) + LARS_COEFFICIENT_FIELD_NUMBER;
      hash = (53 * hash) + java.lang.Float.floatToIntBits(
          getLarsCoefficient());
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.LARSModelUpdateConf parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.LARSModelUpdateConf parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.LARSModelUpdateConf parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.LARSModelUpdateConf parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.LARSModelUpdateConf parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.LARSModelUpdateConf parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.LARSModelUpdateConf parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.LARSModelUpdateConf parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.LARSModelUpdateConf parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.LARSModelUpdateConf parseFrom(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  public Builder newBuilderForType() { return newBuilder(); }
  public static Builder newBuilder() {
    return DEFAULT_INSTANCE.toBuilder();
  }
  public static Builder newBuilder(org.oneflow.core.job.LARSModelUpdateConf prototype) {
    return DEFAULT_INSTANCE.toBuilder().mergeFrom(prototype);
  }
  public Builder toBuilder() {
    return this == DEFAULT_INSTANCE
        ? new Builder() : new Builder().mergeFrom(this);
  }

  @java.lang.Override
  protected Builder newBuilderForType(
      com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
    Builder builder = new Builder(parent);
    return builder;
  }
  /**
   * Protobuf type {@code oneflow.LARSModelUpdateConf}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.LARSModelUpdateConf)
      org.oneflow.core.job.LARSModelUpdateConfOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.JobConf.internal_static_oneflow_LARSModelUpdateConf_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.JobConf.internal_static_oneflow_LARSModelUpdateConf_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.LARSModelUpdateConf.class, org.oneflow.core.job.LARSModelUpdateConf.Builder.class);
    }

    // Construct using org.oneflow.core.job.LARSModelUpdateConf.newBuilder()
    private Builder() {
      maybeForceBuilderInitialization();
    }

    private Builder(
        com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
      super(parent);
      maybeForceBuilderInitialization();
    }
    private void maybeForceBuilderInitialization() {
      if (com.google.protobuf.GeneratedMessageV3
              .alwaysUseFieldBuilders) {
      }
    }
    public Builder clear() {
      super.clear();
      momentumBeta_ = 0.9F;
      bitField0_ = (bitField0_ & ~0x00000001);
      epsilon_ = 1e-09F;
      bitField0_ = (bitField0_ & ~0x00000002);
      larsCoefficient_ = 0.0001F;
      bitField0_ = (bitField0_ & ~0x00000004);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.JobConf.internal_static_oneflow_LARSModelUpdateConf_descriptor;
    }

    public org.oneflow.core.job.LARSModelUpdateConf getDefaultInstanceForType() {
      return org.oneflow.core.job.LARSModelUpdateConf.getDefaultInstance();
    }

    public org.oneflow.core.job.LARSModelUpdateConf build() {
      org.oneflow.core.job.LARSModelUpdateConf result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.LARSModelUpdateConf buildPartial() {
      org.oneflow.core.job.LARSModelUpdateConf result = new org.oneflow.core.job.LARSModelUpdateConf(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      result.momentumBeta_ = momentumBeta_;
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000002;
      }
      result.epsilon_ = epsilon_;
      if (((from_bitField0_ & 0x00000004) == 0x00000004)) {
        to_bitField0_ |= 0x00000004;
      }
      result.larsCoefficient_ = larsCoefficient_;
      result.bitField0_ = to_bitField0_;
      onBuilt();
      return result;
    }

    public Builder clone() {
      return (Builder) super.clone();
    }
    public Builder setField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        Object value) {
      return (Builder) super.setField(field, value);
    }
    public Builder clearField(
        com.google.protobuf.Descriptors.FieldDescriptor field) {
      return (Builder) super.clearField(field);
    }
    public Builder clearOneof(
        com.google.protobuf.Descriptors.OneofDescriptor oneof) {
      return (Builder) super.clearOneof(oneof);
    }
    public Builder setRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        int index, Object value) {
      return (Builder) super.setRepeatedField(field, index, value);
    }
    public Builder addRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        Object value) {
      return (Builder) super.addRepeatedField(field, value);
    }
    public Builder mergeFrom(com.google.protobuf.Message other) {
      if (other instanceof org.oneflow.core.job.LARSModelUpdateConf) {
        return mergeFrom((org.oneflow.core.job.LARSModelUpdateConf)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.LARSModelUpdateConf other) {
      if (other == org.oneflow.core.job.LARSModelUpdateConf.getDefaultInstance()) return this;
      if (other.hasMomentumBeta()) {
        setMomentumBeta(other.getMomentumBeta());
      }
      if (other.hasEpsilon()) {
        setEpsilon(other.getEpsilon());
      }
      if (other.hasLarsCoefficient()) {
        setLarsCoefficient(other.getLarsCoefficient());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.job.LARSModelUpdateConf parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.LARSModelUpdateConf) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private float momentumBeta_ = 0.9F;
    /**
     * <code>optional float momentum_beta = 1 [default = 0.9];</code>
     */
    public boolean hasMomentumBeta() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>optional float momentum_beta = 1 [default = 0.9];</code>
     */
    public float getMomentumBeta() {
      return momentumBeta_;
    }
    /**
     * <code>optional float momentum_beta = 1 [default = 0.9];</code>
     */
    public Builder setMomentumBeta(float value) {
      bitField0_ |= 0x00000001;
      momentumBeta_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional float momentum_beta = 1 [default = 0.9];</code>
     */
    public Builder clearMomentumBeta() {
      bitField0_ = (bitField0_ & ~0x00000001);
      momentumBeta_ = 0.9F;
      onChanged();
      return this;
    }

    private float epsilon_ = 1e-09F;
    /**
     * <code>optional float epsilon = 2 [default = 1e-09];</code>
     */
    public boolean hasEpsilon() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>optional float epsilon = 2 [default = 1e-09];</code>
     */
    public float getEpsilon() {
      return epsilon_;
    }
    /**
     * <code>optional float epsilon = 2 [default = 1e-09];</code>
     */
    public Builder setEpsilon(float value) {
      bitField0_ |= 0x00000002;
      epsilon_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional float epsilon = 2 [default = 1e-09];</code>
     */
    public Builder clearEpsilon() {
      bitField0_ = (bitField0_ & ~0x00000002);
      epsilon_ = 1e-09F;
      onChanged();
      return this;
    }

    private float larsCoefficient_ = 0.0001F;
    /**
     * <code>optional float lars_coefficient = 3 [default = 0.0001];</code>
     */
    public boolean hasLarsCoefficient() {
      return ((bitField0_ & 0x00000004) == 0x00000004);
    }
    /**
     * <code>optional float lars_coefficient = 3 [default = 0.0001];</code>
     */
    public float getLarsCoefficient() {
      return larsCoefficient_;
    }
    /**
     * <code>optional float lars_coefficient = 3 [default = 0.0001];</code>
     */
    public Builder setLarsCoefficient(float value) {
      bitField0_ |= 0x00000004;
      larsCoefficient_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional float lars_coefficient = 3 [default = 0.0001];</code>
     */
    public Builder clearLarsCoefficient() {
      bitField0_ = (bitField0_ & ~0x00000004);
      larsCoefficient_ = 0.0001F;
      onChanged();
      return this;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:oneflow.LARSModelUpdateConf)
  }

  // @@protoc_insertion_point(class_scope:oneflow.LARSModelUpdateConf)
  private static final org.oneflow.core.job.LARSModelUpdateConf DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.LARSModelUpdateConf();
  }

  public static org.oneflow.core.job.LARSModelUpdateConf getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<LARSModelUpdateConf>
      PARSER = new com.google.protobuf.AbstractParser<LARSModelUpdateConf>() {
    public LARSModelUpdateConf parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new LARSModelUpdateConf(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<LARSModelUpdateConf> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<LARSModelUpdateConf> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.LARSModelUpdateConf getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

