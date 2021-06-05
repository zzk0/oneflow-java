// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/job_conf.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.AdamModelUpdateConf}
 */
public  final class AdamModelUpdateConf extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.AdamModelUpdateConf)
    AdamModelUpdateConfOrBuilder {
  // Use AdamModelUpdateConf.newBuilder() to construct.
  private AdamModelUpdateConf(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private AdamModelUpdateConf() {
    beta1_ = 0.9F;
    beta2_ = 0.999F;
    epsilon_ = 1e-08F;
    doBiasCorrection_ = false;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private AdamModelUpdateConf(
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
            beta1_ = input.readFloat();
            break;
          }
          case 21: {
            bitField0_ |= 0x00000002;
            beta2_ = input.readFloat();
            break;
          }
          case 29: {
            bitField0_ |= 0x00000004;
            epsilon_ = input.readFloat();
            break;
          }
          case 32: {
            bitField0_ |= 0x00000008;
            doBiasCorrection_ = input.readBool();
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
    return org.oneflow.core.job.JobConf.internal_static_oneflow_AdamModelUpdateConf_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.JobConf.internal_static_oneflow_AdamModelUpdateConf_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.AdamModelUpdateConf.class, org.oneflow.core.job.AdamModelUpdateConf.Builder.class);
  }

  private int bitField0_;
  public static final int BETA1_FIELD_NUMBER = 1;
  private float beta1_;
  /**
   * <code>optional float beta1 = 1 [default = 0.9];</code>
   */
  public boolean hasBeta1() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>optional float beta1 = 1 [default = 0.9];</code>
   */
  public float getBeta1() {
    return beta1_;
  }

  public static final int BETA2_FIELD_NUMBER = 2;
  private float beta2_;
  /**
   * <code>optional float beta2 = 2 [default = 0.999];</code>
   */
  public boolean hasBeta2() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>optional float beta2 = 2 [default = 0.999];</code>
   */
  public float getBeta2() {
    return beta2_;
  }

  public static final int EPSILON_FIELD_NUMBER = 3;
  private float epsilon_;
  /**
   * <code>optional float epsilon = 3 [default = 1e-08];</code>
   */
  public boolean hasEpsilon() {
    return ((bitField0_ & 0x00000004) == 0x00000004);
  }
  /**
   * <code>optional float epsilon = 3 [default = 1e-08];</code>
   */
  public float getEpsilon() {
    return epsilon_;
  }

  public static final int DO_BIAS_CORRECTION_FIELD_NUMBER = 4;
  private boolean doBiasCorrection_;
  /**
   * <code>optional bool do_bias_correction = 4 [default = false];</code>
   */
  public boolean hasDoBiasCorrection() {
    return ((bitField0_ & 0x00000008) == 0x00000008);
  }
  /**
   * <code>optional bool do_bias_correction = 4 [default = false];</code>
   */
  public boolean getDoBiasCorrection() {
    return doBiasCorrection_;
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
      output.writeFloat(1, beta1_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      output.writeFloat(2, beta2_);
    }
    if (((bitField0_ & 0x00000004) == 0x00000004)) {
      output.writeFloat(3, epsilon_);
    }
    if (((bitField0_ & 0x00000008) == 0x00000008)) {
      output.writeBool(4, doBiasCorrection_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.CodedOutputStream
        .computeFloatSize(1, beta1_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.CodedOutputStream
        .computeFloatSize(2, beta2_);
    }
    if (((bitField0_ & 0x00000004) == 0x00000004)) {
      size += com.google.protobuf.CodedOutputStream
        .computeFloatSize(3, epsilon_);
    }
    if (((bitField0_ & 0x00000008) == 0x00000008)) {
      size += com.google.protobuf.CodedOutputStream
        .computeBoolSize(4, doBiasCorrection_);
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
    if (!(obj instanceof org.oneflow.core.job.AdamModelUpdateConf)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.AdamModelUpdateConf other = (org.oneflow.core.job.AdamModelUpdateConf) obj;

    boolean result = true;
    result = result && (hasBeta1() == other.hasBeta1());
    if (hasBeta1()) {
      result = result && (
          java.lang.Float.floatToIntBits(getBeta1())
          == java.lang.Float.floatToIntBits(
              other.getBeta1()));
    }
    result = result && (hasBeta2() == other.hasBeta2());
    if (hasBeta2()) {
      result = result && (
          java.lang.Float.floatToIntBits(getBeta2())
          == java.lang.Float.floatToIntBits(
              other.getBeta2()));
    }
    result = result && (hasEpsilon() == other.hasEpsilon());
    if (hasEpsilon()) {
      result = result && (
          java.lang.Float.floatToIntBits(getEpsilon())
          == java.lang.Float.floatToIntBits(
              other.getEpsilon()));
    }
    result = result && (hasDoBiasCorrection() == other.hasDoBiasCorrection());
    if (hasDoBiasCorrection()) {
      result = result && (getDoBiasCorrection()
          == other.getDoBiasCorrection());
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
    if (hasBeta1()) {
      hash = (37 * hash) + BETA1_FIELD_NUMBER;
      hash = (53 * hash) + java.lang.Float.floatToIntBits(
          getBeta1());
    }
    if (hasBeta2()) {
      hash = (37 * hash) + BETA2_FIELD_NUMBER;
      hash = (53 * hash) + java.lang.Float.floatToIntBits(
          getBeta2());
    }
    if (hasEpsilon()) {
      hash = (37 * hash) + EPSILON_FIELD_NUMBER;
      hash = (53 * hash) + java.lang.Float.floatToIntBits(
          getEpsilon());
    }
    if (hasDoBiasCorrection()) {
      hash = (37 * hash) + DO_BIAS_CORRECTION_FIELD_NUMBER;
      hash = (53 * hash) + com.google.protobuf.Internal.hashBoolean(
          getDoBiasCorrection());
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.AdamModelUpdateConf parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.AdamModelUpdateConf parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.AdamModelUpdateConf parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.AdamModelUpdateConf parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.AdamModelUpdateConf parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.AdamModelUpdateConf parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.AdamModelUpdateConf parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.AdamModelUpdateConf parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.AdamModelUpdateConf parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.AdamModelUpdateConf parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.AdamModelUpdateConf prototype) {
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
   * Protobuf type {@code oneflow.AdamModelUpdateConf}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.AdamModelUpdateConf)
      org.oneflow.core.job.AdamModelUpdateConfOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.JobConf.internal_static_oneflow_AdamModelUpdateConf_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.JobConf.internal_static_oneflow_AdamModelUpdateConf_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.AdamModelUpdateConf.class, org.oneflow.core.job.AdamModelUpdateConf.Builder.class);
    }

    // Construct using org.oneflow.core.job.AdamModelUpdateConf.newBuilder()
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
      beta1_ = 0.9F;
      bitField0_ = (bitField0_ & ~0x00000001);
      beta2_ = 0.999F;
      bitField0_ = (bitField0_ & ~0x00000002);
      epsilon_ = 1e-08F;
      bitField0_ = (bitField0_ & ~0x00000004);
      doBiasCorrection_ = false;
      bitField0_ = (bitField0_ & ~0x00000008);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.JobConf.internal_static_oneflow_AdamModelUpdateConf_descriptor;
    }

    public org.oneflow.core.job.AdamModelUpdateConf getDefaultInstanceForType() {
      return org.oneflow.core.job.AdamModelUpdateConf.getDefaultInstance();
    }

    public org.oneflow.core.job.AdamModelUpdateConf build() {
      org.oneflow.core.job.AdamModelUpdateConf result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.AdamModelUpdateConf buildPartial() {
      org.oneflow.core.job.AdamModelUpdateConf result = new org.oneflow.core.job.AdamModelUpdateConf(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      result.beta1_ = beta1_;
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000002;
      }
      result.beta2_ = beta2_;
      if (((from_bitField0_ & 0x00000004) == 0x00000004)) {
        to_bitField0_ |= 0x00000004;
      }
      result.epsilon_ = epsilon_;
      if (((from_bitField0_ & 0x00000008) == 0x00000008)) {
        to_bitField0_ |= 0x00000008;
      }
      result.doBiasCorrection_ = doBiasCorrection_;
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
      if (other instanceof org.oneflow.core.job.AdamModelUpdateConf) {
        return mergeFrom((org.oneflow.core.job.AdamModelUpdateConf)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.AdamModelUpdateConf other) {
      if (other == org.oneflow.core.job.AdamModelUpdateConf.getDefaultInstance()) return this;
      if (other.hasBeta1()) {
        setBeta1(other.getBeta1());
      }
      if (other.hasBeta2()) {
        setBeta2(other.getBeta2());
      }
      if (other.hasEpsilon()) {
        setEpsilon(other.getEpsilon());
      }
      if (other.hasDoBiasCorrection()) {
        setDoBiasCorrection(other.getDoBiasCorrection());
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
      org.oneflow.core.job.AdamModelUpdateConf parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.AdamModelUpdateConf) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private float beta1_ = 0.9F;
    /**
     * <code>optional float beta1 = 1 [default = 0.9];</code>
     */
    public boolean hasBeta1() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>optional float beta1 = 1 [default = 0.9];</code>
     */
    public float getBeta1() {
      return beta1_;
    }
    /**
     * <code>optional float beta1 = 1 [default = 0.9];</code>
     */
    public Builder setBeta1(float value) {
      bitField0_ |= 0x00000001;
      beta1_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional float beta1 = 1 [default = 0.9];</code>
     */
    public Builder clearBeta1() {
      bitField0_ = (bitField0_ & ~0x00000001);
      beta1_ = 0.9F;
      onChanged();
      return this;
    }

    private float beta2_ = 0.999F;
    /**
     * <code>optional float beta2 = 2 [default = 0.999];</code>
     */
    public boolean hasBeta2() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>optional float beta2 = 2 [default = 0.999];</code>
     */
    public float getBeta2() {
      return beta2_;
    }
    /**
     * <code>optional float beta2 = 2 [default = 0.999];</code>
     */
    public Builder setBeta2(float value) {
      bitField0_ |= 0x00000002;
      beta2_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional float beta2 = 2 [default = 0.999];</code>
     */
    public Builder clearBeta2() {
      bitField0_ = (bitField0_ & ~0x00000002);
      beta2_ = 0.999F;
      onChanged();
      return this;
    }

    private float epsilon_ = 1e-08F;
    /**
     * <code>optional float epsilon = 3 [default = 1e-08];</code>
     */
    public boolean hasEpsilon() {
      return ((bitField0_ & 0x00000004) == 0x00000004);
    }
    /**
     * <code>optional float epsilon = 3 [default = 1e-08];</code>
     */
    public float getEpsilon() {
      return epsilon_;
    }
    /**
     * <code>optional float epsilon = 3 [default = 1e-08];</code>
     */
    public Builder setEpsilon(float value) {
      bitField0_ |= 0x00000004;
      epsilon_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional float epsilon = 3 [default = 1e-08];</code>
     */
    public Builder clearEpsilon() {
      bitField0_ = (bitField0_ & ~0x00000004);
      epsilon_ = 1e-08F;
      onChanged();
      return this;
    }

    private boolean doBiasCorrection_ ;
    /**
     * <code>optional bool do_bias_correction = 4 [default = false];</code>
     */
    public boolean hasDoBiasCorrection() {
      return ((bitField0_ & 0x00000008) == 0x00000008);
    }
    /**
     * <code>optional bool do_bias_correction = 4 [default = false];</code>
     */
    public boolean getDoBiasCorrection() {
      return doBiasCorrection_;
    }
    /**
     * <code>optional bool do_bias_correction = 4 [default = false];</code>
     */
    public Builder setDoBiasCorrection(boolean value) {
      bitField0_ |= 0x00000008;
      doBiasCorrection_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional bool do_bias_correction = 4 [default = false];</code>
     */
    public Builder clearDoBiasCorrection() {
      bitField0_ = (bitField0_ & ~0x00000008);
      doBiasCorrection_ = false;
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


    // @@protoc_insertion_point(builder_scope:oneflow.AdamModelUpdateConf)
  }

  // @@protoc_insertion_point(class_scope:oneflow.AdamModelUpdateConf)
  private static final org.oneflow.core.job.AdamModelUpdateConf DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.AdamModelUpdateConf();
  }

  public static org.oneflow.core.job.AdamModelUpdateConf getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<AdamModelUpdateConf>
      PARSER = new com.google.protobuf.AbstractParser<AdamModelUpdateConf>() {
    public AdamModelUpdateConf parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new AdamModelUpdateConf(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<AdamModelUpdateConf> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<AdamModelUpdateConf> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.AdamModelUpdateConf getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

