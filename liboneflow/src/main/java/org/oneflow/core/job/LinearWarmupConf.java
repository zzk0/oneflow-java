// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/learning_rate_schedule_conf.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.LinearWarmupConf}
 */
public  final class LinearWarmupConf extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.LinearWarmupConf)
    LinearWarmupConfOrBuilder {
  // Use LinearWarmupConf.newBuilder() to construct.
  private LinearWarmupConf(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private LinearWarmupConf() {
    warmupBatches_ = 0L;
    startMultiplier_ = 0D;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private LinearWarmupConf(
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
            warmupBatches_ = input.readInt64();
            break;
          }
          case 17: {
            bitField0_ |= 0x00000002;
            startMultiplier_ = input.readDouble();
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
    return org.oneflow.core.job.LearningRateScheduleConf.internal_static_oneflow_LinearWarmupConf_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.LearningRateScheduleConf.internal_static_oneflow_LinearWarmupConf_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.LinearWarmupConf.class, org.oneflow.core.job.LinearWarmupConf.Builder.class);
  }

  private int bitField0_;
  public static final int WARMUP_BATCHES_FIELD_NUMBER = 1;
  private long warmupBatches_;
  /**
   * <code>required int64 warmup_batches = 1;</code>
   */
  public boolean hasWarmupBatches() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>required int64 warmup_batches = 1;</code>
   */
  public long getWarmupBatches() {
    return warmupBatches_;
  }

  public static final int START_MULTIPLIER_FIELD_NUMBER = 2;
  private double startMultiplier_;
  /**
   * <code>required double start_multiplier = 2;</code>
   */
  public boolean hasStartMultiplier() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>required double start_multiplier = 2;</code>
   */
  public double getStartMultiplier() {
    return startMultiplier_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (!hasWarmupBatches()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!hasStartMultiplier()) {
      memoizedIsInitialized = 0;
      return false;
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      output.writeInt64(1, warmupBatches_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      output.writeDouble(2, startMultiplier_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt64Size(1, warmupBatches_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.CodedOutputStream
        .computeDoubleSize(2, startMultiplier_);
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
    if (!(obj instanceof org.oneflow.core.job.LinearWarmupConf)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.LinearWarmupConf other = (org.oneflow.core.job.LinearWarmupConf) obj;

    boolean result = true;
    result = result && (hasWarmupBatches() == other.hasWarmupBatches());
    if (hasWarmupBatches()) {
      result = result && (getWarmupBatches()
          == other.getWarmupBatches());
    }
    result = result && (hasStartMultiplier() == other.hasStartMultiplier());
    if (hasStartMultiplier()) {
      result = result && (
          java.lang.Double.doubleToLongBits(getStartMultiplier())
          == java.lang.Double.doubleToLongBits(
              other.getStartMultiplier()));
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
    if (hasWarmupBatches()) {
      hash = (37 * hash) + WARMUP_BATCHES_FIELD_NUMBER;
      hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
          getWarmupBatches());
    }
    if (hasStartMultiplier()) {
      hash = (37 * hash) + START_MULTIPLIER_FIELD_NUMBER;
      hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
          java.lang.Double.doubleToLongBits(getStartMultiplier()));
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.LinearWarmupConf parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.LinearWarmupConf parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.LinearWarmupConf parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.LinearWarmupConf parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.LinearWarmupConf parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.LinearWarmupConf parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.LinearWarmupConf parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.LinearWarmupConf parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.LinearWarmupConf parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.LinearWarmupConf parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.LinearWarmupConf prototype) {
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
   * Protobuf type {@code oneflow.LinearWarmupConf}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.LinearWarmupConf)
      org.oneflow.core.job.LinearWarmupConfOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.LearningRateScheduleConf.internal_static_oneflow_LinearWarmupConf_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.LearningRateScheduleConf.internal_static_oneflow_LinearWarmupConf_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.LinearWarmupConf.class, org.oneflow.core.job.LinearWarmupConf.Builder.class);
    }

    // Construct using org.oneflow.core.job.LinearWarmupConf.newBuilder()
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
      warmupBatches_ = 0L;
      bitField0_ = (bitField0_ & ~0x00000001);
      startMultiplier_ = 0D;
      bitField0_ = (bitField0_ & ~0x00000002);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.LearningRateScheduleConf.internal_static_oneflow_LinearWarmupConf_descriptor;
    }

    public org.oneflow.core.job.LinearWarmupConf getDefaultInstanceForType() {
      return org.oneflow.core.job.LinearWarmupConf.getDefaultInstance();
    }

    public org.oneflow.core.job.LinearWarmupConf build() {
      org.oneflow.core.job.LinearWarmupConf result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.LinearWarmupConf buildPartial() {
      org.oneflow.core.job.LinearWarmupConf result = new org.oneflow.core.job.LinearWarmupConf(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      result.warmupBatches_ = warmupBatches_;
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000002;
      }
      result.startMultiplier_ = startMultiplier_;
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
      if (other instanceof org.oneflow.core.job.LinearWarmupConf) {
        return mergeFrom((org.oneflow.core.job.LinearWarmupConf)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.LinearWarmupConf other) {
      if (other == org.oneflow.core.job.LinearWarmupConf.getDefaultInstance()) return this;
      if (other.hasWarmupBatches()) {
        setWarmupBatches(other.getWarmupBatches());
      }
      if (other.hasStartMultiplier()) {
        setStartMultiplier(other.getStartMultiplier());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (!hasWarmupBatches()) {
        return false;
      }
      if (!hasStartMultiplier()) {
        return false;
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.job.LinearWarmupConf parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.LinearWarmupConf) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private long warmupBatches_ ;
    /**
     * <code>required int64 warmup_batches = 1;</code>
     */
    public boolean hasWarmupBatches() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>required int64 warmup_batches = 1;</code>
     */
    public long getWarmupBatches() {
      return warmupBatches_;
    }
    /**
     * <code>required int64 warmup_batches = 1;</code>
     */
    public Builder setWarmupBatches(long value) {
      bitField0_ |= 0x00000001;
      warmupBatches_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>required int64 warmup_batches = 1;</code>
     */
    public Builder clearWarmupBatches() {
      bitField0_ = (bitField0_ & ~0x00000001);
      warmupBatches_ = 0L;
      onChanged();
      return this;
    }

    private double startMultiplier_ ;
    /**
     * <code>required double start_multiplier = 2;</code>
     */
    public boolean hasStartMultiplier() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>required double start_multiplier = 2;</code>
     */
    public double getStartMultiplier() {
      return startMultiplier_;
    }
    /**
     * <code>required double start_multiplier = 2;</code>
     */
    public Builder setStartMultiplier(double value) {
      bitField0_ |= 0x00000002;
      startMultiplier_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>required double start_multiplier = 2;</code>
     */
    public Builder clearStartMultiplier() {
      bitField0_ = (bitField0_ & ~0x00000002);
      startMultiplier_ = 0D;
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


    // @@protoc_insertion_point(builder_scope:oneflow.LinearWarmupConf)
  }

  // @@protoc_insertion_point(class_scope:oneflow.LinearWarmupConf)
  private static final org.oneflow.core.job.LinearWarmupConf DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.LinearWarmupConf();
  }

  public static org.oneflow.core.job.LinearWarmupConf getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<LinearWarmupConf>
      PARSER = new com.google.protobuf.AbstractParser<LinearWarmupConf>() {
    public LinearWarmupConf parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new LinearWarmupConf(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<LinearWarmupConf> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<LinearWarmupConf> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.LinearWarmupConf getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

