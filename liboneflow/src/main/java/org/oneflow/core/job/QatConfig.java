// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/job_conf.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.QatConfig}
 */
public  final class QatConfig extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.QatConfig)
    QatConfigOrBuilder {
  // Use QatConfig.newBuilder() to construct.
  private QatConfig(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private QatConfig() {
    perChannelWeightQuantization_ = false;
    symmetric_ = true;
    movingMinMaxMomentum_ = 0.95F;
    movingMinMaxStopUpdateAfterIters_ = 0L;
    targetBackend_ = "";
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private QatConfig(
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
          case 8: {
            bitField0_ |= 0x00000001;
            perChannelWeightQuantization_ = input.readBool();
            break;
          }
          case 16: {
            bitField0_ |= 0x00000002;
            symmetric_ = input.readBool();
            break;
          }
          case 29: {
            bitField0_ |= 0x00000004;
            movingMinMaxMomentum_ = input.readFloat();
            break;
          }
          case 32: {
            bitField0_ |= 0x00000008;
            movingMinMaxStopUpdateAfterIters_ = input.readInt64();
            break;
          }
          case 42: {
            com.google.protobuf.ByteString bs = input.readBytes();
            bitField0_ |= 0x00000010;
            targetBackend_ = bs;
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
    return org.oneflow.core.job.JobConf.internal_static_oneflow_QatConfig_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.JobConf.internal_static_oneflow_QatConfig_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.QatConfig.class, org.oneflow.core.job.QatConfig.Builder.class);
  }

  private int bitField0_;
  public static final int PER_CHANNEL_WEIGHT_QUANTIZATION_FIELD_NUMBER = 1;
  private boolean perChannelWeightQuantization_;
  /**
   * <code>optional bool per_channel_weight_quantization = 1 [default = false];</code>
   */
  public boolean hasPerChannelWeightQuantization() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>optional bool per_channel_weight_quantization = 1 [default = false];</code>
   */
  public boolean getPerChannelWeightQuantization() {
    return perChannelWeightQuantization_;
  }

  public static final int SYMMETRIC_FIELD_NUMBER = 2;
  private boolean symmetric_;
  /**
   * <code>optional bool symmetric = 2 [default = true];</code>
   */
  public boolean hasSymmetric() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>optional bool symmetric = 2 [default = true];</code>
   */
  public boolean getSymmetric() {
    return symmetric_;
  }

  public static final int MOVING_MIN_MAX_MOMENTUM_FIELD_NUMBER = 3;
  private float movingMinMaxMomentum_;
  /**
   * <code>optional float moving_min_max_momentum = 3 [default = 0.95];</code>
   */
  public boolean hasMovingMinMaxMomentum() {
    return ((bitField0_ & 0x00000004) == 0x00000004);
  }
  /**
   * <code>optional float moving_min_max_momentum = 3 [default = 0.95];</code>
   */
  public float getMovingMinMaxMomentum() {
    return movingMinMaxMomentum_;
  }

  public static final int MOVING_MIN_MAX_STOP_UPDATE_AFTER_ITERS_FIELD_NUMBER = 4;
  private long movingMinMaxStopUpdateAfterIters_;
  /**
   * <code>optional int64 moving_min_max_stop_update_after_iters = 4;</code>
   */
  public boolean hasMovingMinMaxStopUpdateAfterIters() {
    return ((bitField0_ & 0x00000008) == 0x00000008);
  }
  /**
   * <code>optional int64 moving_min_max_stop_update_after_iters = 4;</code>
   */
  public long getMovingMinMaxStopUpdateAfterIters() {
    return movingMinMaxStopUpdateAfterIters_;
  }

  public static final int TARGET_BACKEND_FIELD_NUMBER = 5;
  private volatile java.lang.Object targetBackend_;
  /**
   * <code>optional string target_backend = 5 [default = ""];</code>
   */
  public boolean hasTargetBackend() {
    return ((bitField0_ & 0x00000010) == 0x00000010);
  }
  /**
   * <code>optional string target_backend = 5 [default = ""];</code>
   */
  public java.lang.String getTargetBackend() {
    java.lang.Object ref = targetBackend_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      if (bs.isValidUtf8()) {
        targetBackend_ = s;
      }
      return s;
    }
  }
  /**
   * <code>optional string target_backend = 5 [default = ""];</code>
   */
  public com.google.protobuf.ByteString
      getTargetBackendBytes() {
    java.lang.Object ref = targetBackend_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      targetBackend_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
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
      output.writeBool(1, perChannelWeightQuantization_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      output.writeBool(2, symmetric_);
    }
    if (((bitField0_ & 0x00000004) == 0x00000004)) {
      output.writeFloat(3, movingMinMaxMomentum_);
    }
    if (((bitField0_ & 0x00000008) == 0x00000008)) {
      output.writeInt64(4, movingMinMaxStopUpdateAfterIters_);
    }
    if (((bitField0_ & 0x00000010) == 0x00000010)) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 5, targetBackend_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.CodedOutputStream
        .computeBoolSize(1, perChannelWeightQuantization_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.CodedOutputStream
        .computeBoolSize(2, symmetric_);
    }
    if (((bitField0_ & 0x00000004) == 0x00000004)) {
      size += com.google.protobuf.CodedOutputStream
        .computeFloatSize(3, movingMinMaxMomentum_);
    }
    if (((bitField0_ & 0x00000008) == 0x00000008)) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt64Size(4, movingMinMaxStopUpdateAfterIters_);
    }
    if (((bitField0_ & 0x00000010) == 0x00000010)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(5, targetBackend_);
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
    if (!(obj instanceof org.oneflow.core.job.QatConfig)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.QatConfig other = (org.oneflow.core.job.QatConfig) obj;

    boolean result = true;
    result = result && (hasPerChannelWeightQuantization() == other.hasPerChannelWeightQuantization());
    if (hasPerChannelWeightQuantization()) {
      result = result && (getPerChannelWeightQuantization()
          == other.getPerChannelWeightQuantization());
    }
    result = result && (hasSymmetric() == other.hasSymmetric());
    if (hasSymmetric()) {
      result = result && (getSymmetric()
          == other.getSymmetric());
    }
    result = result && (hasMovingMinMaxMomentum() == other.hasMovingMinMaxMomentum());
    if (hasMovingMinMaxMomentum()) {
      result = result && (
          java.lang.Float.floatToIntBits(getMovingMinMaxMomentum())
          == java.lang.Float.floatToIntBits(
              other.getMovingMinMaxMomentum()));
    }
    result = result && (hasMovingMinMaxStopUpdateAfterIters() == other.hasMovingMinMaxStopUpdateAfterIters());
    if (hasMovingMinMaxStopUpdateAfterIters()) {
      result = result && (getMovingMinMaxStopUpdateAfterIters()
          == other.getMovingMinMaxStopUpdateAfterIters());
    }
    result = result && (hasTargetBackend() == other.hasTargetBackend());
    if (hasTargetBackend()) {
      result = result && getTargetBackend()
          .equals(other.getTargetBackend());
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
    if (hasPerChannelWeightQuantization()) {
      hash = (37 * hash) + PER_CHANNEL_WEIGHT_QUANTIZATION_FIELD_NUMBER;
      hash = (53 * hash) + com.google.protobuf.Internal.hashBoolean(
          getPerChannelWeightQuantization());
    }
    if (hasSymmetric()) {
      hash = (37 * hash) + SYMMETRIC_FIELD_NUMBER;
      hash = (53 * hash) + com.google.protobuf.Internal.hashBoolean(
          getSymmetric());
    }
    if (hasMovingMinMaxMomentum()) {
      hash = (37 * hash) + MOVING_MIN_MAX_MOMENTUM_FIELD_NUMBER;
      hash = (53 * hash) + java.lang.Float.floatToIntBits(
          getMovingMinMaxMomentum());
    }
    if (hasMovingMinMaxStopUpdateAfterIters()) {
      hash = (37 * hash) + MOVING_MIN_MAX_STOP_UPDATE_AFTER_ITERS_FIELD_NUMBER;
      hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
          getMovingMinMaxStopUpdateAfterIters());
    }
    if (hasTargetBackend()) {
      hash = (37 * hash) + TARGET_BACKEND_FIELD_NUMBER;
      hash = (53 * hash) + getTargetBackend().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.QatConfig parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.QatConfig parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.QatConfig parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.QatConfig parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.QatConfig parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.QatConfig parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.QatConfig parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.QatConfig parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.QatConfig parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.QatConfig parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.QatConfig prototype) {
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
   * Protobuf type {@code oneflow.QatConfig}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.QatConfig)
      org.oneflow.core.job.QatConfigOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.JobConf.internal_static_oneflow_QatConfig_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.JobConf.internal_static_oneflow_QatConfig_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.QatConfig.class, org.oneflow.core.job.QatConfig.Builder.class);
    }

    // Construct using org.oneflow.core.job.QatConfig.newBuilder()
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
      perChannelWeightQuantization_ = false;
      bitField0_ = (bitField0_ & ~0x00000001);
      symmetric_ = true;
      bitField0_ = (bitField0_ & ~0x00000002);
      movingMinMaxMomentum_ = 0.95F;
      bitField0_ = (bitField0_ & ~0x00000004);
      movingMinMaxStopUpdateAfterIters_ = 0L;
      bitField0_ = (bitField0_ & ~0x00000008);
      targetBackend_ = "";
      bitField0_ = (bitField0_ & ~0x00000010);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.JobConf.internal_static_oneflow_QatConfig_descriptor;
    }

    public org.oneflow.core.job.QatConfig getDefaultInstanceForType() {
      return org.oneflow.core.job.QatConfig.getDefaultInstance();
    }

    public org.oneflow.core.job.QatConfig build() {
      org.oneflow.core.job.QatConfig result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.QatConfig buildPartial() {
      org.oneflow.core.job.QatConfig result = new org.oneflow.core.job.QatConfig(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      result.perChannelWeightQuantization_ = perChannelWeightQuantization_;
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000002;
      }
      result.symmetric_ = symmetric_;
      if (((from_bitField0_ & 0x00000004) == 0x00000004)) {
        to_bitField0_ |= 0x00000004;
      }
      result.movingMinMaxMomentum_ = movingMinMaxMomentum_;
      if (((from_bitField0_ & 0x00000008) == 0x00000008)) {
        to_bitField0_ |= 0x00000008;
      }
      result.movingMinMaxStopUpdateAfterIters_ = movingMinMaxStopUpdateAfterIters_;
      if (((from_bitField0_ & 0x00000010) == 0x00000010)) {
        to_bitField0_ |= 0x00000010;
      }
      result.targetBackend_ = targetBackend_;
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
      if (other instanceof org.oneflow.core.job.QatConfig) {
        return mergeFrom((org.oneflow.core.job.QatConfig)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.QatConfig other) {
      if (other == org.oneflow.core.job.QatConfig.getDefaultInstance()) return this;
      if (other.hasPerChannelWeightQuantization()) {
        setPerChannelWeightQuantization(other.getPerChannelWeightQuantization());
      }
      if (other.hasSymmetric()) {
        setSymmetric(other.getSymmetric());
      }
      if (other.hasMovingMinMaxMomentum()) {
        setMovingMinMaxMomentum(other.getMovingMinMaxMomentum());
      }
      if (other.hasMovingMinMaxStopUpdateAfterIters()) {
        setMovingMinMaxStopUpdateAfterIters(other.getMovingMinMaxStopUpdateAfterIters());
      }
      if (other.hasTargetBackend()) {
        bitField0_ |= 0x00000010;
        targetBackend_ = other.targetBackend_;
        onChanged();
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
      org.oneflow.core.job.QatConfig parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.QatConfig) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private boolean perChannelWeightQuantization_ ;
    /**
     * <code>optional bool per_channel_weight_quantization = 1 [default = false];</code>
     */
    public boolean hasPerChannelWeightQuantization() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>optional bool per_channel_weight_quantization = 1 [default = false];</code>
     */
    public boolean getPerChannelWeightQuantization() {
      return perChannelWeightQuantization_;
    }
    /**
     * <code>optional bool per_channel_weight_quantization = 1 [default = false];</code>
     */
    public Builder setPerChannelWeightQuantization(boolean value) {
      bitField0_ |= 0x00000001;
      perChannelWeightQuantization_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional bool per_channel_weight_quantization = 1 [default = false];</code>
     */
    public Builder clearPerChannelWeightQuantization() {
      bitField0_ = (bitField0_ & ~0x00000001);
      perChannelWeightQuantization_ = false;
      onChanged();
      return this;
    }

    private boolean symmetric_ = true;
    /**
     * <code>optional bool symmetric = 2 [default = true];</code>
     */
    public boolean hasSymmetric() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>optional bool symmetric = 2 [default = true];</code>
     */
    public boolean getSymmetric() {
      return symmetric_;
    }
    /**
     * <code>optional bool symmetric = 2 [default = true];</code>
     */
    public Builder setSymmetric(boolean value) {
      bitField0_ |= 0x00000002;
      symmetric_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional bool symmetric = 2 [default = true];</code>
     */
    public Builder clearSymmetric() {
      bitField0_ = (bitField0_ & ~0x00000002);
      symmetric_ = true;
      onChanged();
      return this;
    }

    private float movingMinMaxMomentum_ = 0.95F;
    /**
     * <code>optional float moving_min_max_momentum = 3 [default = 0.95];</code>
     */
    public boolean hasMovingMinMaxMomentum() {
      return ((bitField0_ & 0x00000004) == 0x00000004);
    }
    /**
     * <code>optional float moving_min_max_momentum = 3 [default = 0.95];</code>
     */
    public float getMovingMinMaxMomentum() {
      return movingMinMaxMomentum_;
    }
    /**
     * <code>optional float moving_min_max_momentum = 3 [default = 0.95];</code>
     */
    public Builder setMovingMinMaxMomentum(float value) {
      bitField0_ |= 0x00000004;
      movingMinMaxMomentum_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional float moving_min_max_momentum = 3 [default = 0.95];</code>
     */
    public Builder clearMovingMinMaxMomentum() {
      bitField0_ = (bitField0_ & ~0x00000004);
      movingMinMaxMomentum_ = 0.95F;
      onChanged();
      return this;
    }

    private long movingMinMaxStopUpdateAfterIters_ ;
    /**
     * <code>optional int64 moving_min_max_stop_update_after_iters = 4;</code>
     */
    public boolean hasMovingMinMaxStopUpdateAfterIters() {
      return ((bitField0_ & 0x00000008) == 0x00000008);
    }
    /**
     * <code>optional int64 moving_min_max_stop_update_after_iters = 4;</code>
     */
    public long getMovingMinMaxStopUpdateAfterIters() {
      return movingMinMaxStopUpdateAfterIters_;
    }
    /**
     * <code>optional int64 moving_min_max_stop_update_after_iters = 4;</code>
     */
    public Builder setMovingMinMaxStopUpdateAfterIters(long value) {
      bitField0_ |= 0x00000008;
      movingMinMaxStopUpdateAfterIters_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional int64 moving_min_max_stop_update_after_iters = 4;</code>
     */
    public Builder clearMovingMinMaxStopUpdateAfterIters() {
      bitField0_ = (bitField0_ & ~0x00000008);
      movingMinMaxStopUpdateAfterIters_ = 0L;
      onChanged();
      return this;
    }

    private java.lang.Object targetBackend_ = "";
    /**
     * <code>optional string target_backend = 5 [default = ""];</code>
     */
    public boolean hasTargetBackend() {
      return ((bitField0_ & 0x00000010) == 0x00000010);
    }
    /**
     * <code>optional string target_backend = 5 [default = ""];</code>
     */
    public java.lang.String getTargetBackend() {
      java.lang.Object ref = targetBackend_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        if (bs.isValidUtf8()) {
          targetBackend_ = s;
        }
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>optional string target_backend = 5 [default = ""];</code>
     */
    public com.google.protobuf.ByteString
        getTargetBackendBytes() {
      java.lang.Object ref = targetBackend_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        targetBackend_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>optional string target_backend = 5 [default = ""];</code>
     */
    public Builder setTargetBackend(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000010;
      targetBackend_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional string target_backend = 5 [default = ""];</code>
     */
    public Builder clearTargetBackend() {
      bitField0_ = (bitField0_ & ~0x00000010);
      targetBackend_ = getDefaultInstance().getTargetBackend();
      onChanged();
      return this;
    }
    /**
     * <code>optional string target_backend = 5 [default = ""];</code>
     */
    public Builder setTargetBackendBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000010;
      targetBackend_ = value;
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


    // @@protoc_insertion_point(builder_scope:oneflow.QatConfig)
  }

  // @@protoc_insertion_point(class_scope:oneflow.QatConfig)
  private static final org.oneflow.core.job.QatConfig DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.QatConfig();
  }

  public static org.oneflow.core.job.QatConfig getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<QatConfig>
      PARSER = new com.google.protobuf.AbstractParser<QatConfig>() {
    public QatConfig parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new QatConfig(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<QatConfig> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<QatConfig> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.QatConfig getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
