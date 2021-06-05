// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/distribute_hirarchy.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.DistributeDim}
 */
public  final class DistributeDim extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.DistributeDim)
    DistributeDimOrBuilder {
  // Use DistributeDim.newBuilder() to construct.
  private DistributeDim(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private DistributeDim() {
    distributeType_ = 0;
    distributeNum_ = 0L;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private DistributeDim(
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
            int rawValue = input.readEnum();
            org.oneflow.core.job.DistributeType value = org.oneflow.core.job.DistributeType.valueOf(rawValue);
            if (value == null) {
              unknownFields.mergeVarintField(1, rawValue);
            } else {
              bitField0_ |= 0x00000001;
              distributeType_ = rawValue;
            }
            break;
          }
          case 18: {
            org.oneflow.core.job.SbpParallel.Builder subBuilder = null;
            if (((bitField0_ & 0x00000002) == 0x00000002)) {
              subBuilder = sbpParallel_.toBuilder();
            }
            sbpParallel_ = input.readMessage(org.oneflow.core.job.SbpParallel.PARSER, extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(sbpParallel_);
              sbpParallel_ = subBuilder.buildPartial();
            }
            bitField0_ |= 0x00000002;
            break;
          }
          case 24: {
            bitField0_ |= 0x00000004;
            distributeNum_ = input.readInt64();
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
    return org.oneflow.core.job.DistributeHirarchyOuterClass.internal_static_oneflow_DistributeDim_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.DistributeHirarchyOuterClass.internal_static_oneflow_DistributeDim_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.DistributeDim.class, org.oneflow.core.job.DistributeDim.Builder.class);
  }

  private int bitField0_;
  public static final int DISTRIBUTE_TYPE_FIELD_NUMBER = 1;
  private int distributeType_;
  /**
   * <code>required .oneflow.DistributeType distribute_type = 1;</code>
   */
  public boolean hasDistributeType() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>required .oneflow.DistributeType distribute_type = 1;</code>
   */
  public org.oneflow.core.job.DistributeType getDistributeType() {
    org.oneflow.core.job.DistributeType result = org.oneflow.core.job.DistributeType.valueOf(distributeType_);
    return result == null ? org.oneflow.core.job.DistributeType.kInvalidDistributeType : result;
  }

  public static final int SBP_PARALLEL_FIELD_NUMBER = 2;
  private org.oneflow.core.job.SbpParallel sbpParallel_;
  /**
   * <code>required .oneflow.SbpParallel sbp_parallel = 2;</code>
   */
  public boolean hasSbpParallel() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>required .oneflow.SbpParallel sbp_parallel = 2;</code>
   */
  public org.oneflow.core.job.SbpParallel getSbpParallel() {
    return sbpParallel_ == null ? org.oneflow.core.job.SbpParallel.getDefaultInstance() : sbpParallel_;
  }
  /**
   * <code>required .oneflow.SbpParallel sbp_parallel = 2;</code>
   */
  public org.oneflow.core.job.SbpParallelOrBuilder getSbpParallelOrBuilder() {
    return sbpParallel_ == null ? org.oneflow.core.job.SbpParallel.getDefaultInstance() : sbpParallel_;
  }

  public static final int DISTRIBUTE_NUM_FIELD_NUMBER = 3;
  private long distributeNum_;
  /**
   * <code>required int64 distribute_num = 3;</code>
   */
  public boolean hasDistributeNum() {
    return ((bitField0_ & 0x00000004) == 0x00000004);
  }
  /**
   * <code>required int64 distribute_num = 3;</code>
   */
  public long getDistributeNum() {
    return distributeNum_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (!hasDistributeType()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!hasSbpParallel()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!hasDistributeNum()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!getSbpParallel().isInitialized()) {
      memoizedIsInitialized = 0;
      return false;
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      output.writeEnum(1, distributeType_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      output.writeMessage(2, getSbpParallel());
    }
    if (((bitField0_ & 0x00000004) == 0x00000004)) {
      output.writeInt64(3, distributeNum_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.CodedOutputStream
        .computeEnumSize(1, distributeType_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(2, getSbpParallel());
    }
    if (((bitField0_ & 0x00000004) == 0x00000004)) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt64Size(3, distributeNum_);
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
    if (!(obj instanceof org.oneflow.core.job.DistributeDim)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.DistributeDim other = (org.oneflow.core.job.DistributeDim) obj;

    boolean result = true;
    result = result && (hasDistributeType() == other.hasDistributeType());
    if (hasDistributeType()) {
      result = result && distributeType_ == other.distributeType_;
    }
    result = result && (hasSbpParallel() == other.hasSbpParallel());
    if (hasSbpParallel()) {
      result = result && getSbpParallel()
          .equals(other.getSbpParallel());
    }
    result = result && (hasDistributeNum() == other.hasDistributeNum());
    if (hasDistributeNum()) {
      result = result && (getDistributeNum()
          == other.getDistributeNum());
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
    if (hasDistributeType()) {
      hash = (37 * hash) + DISTRIBUTE_TYPE_FIELD_NUMBER;
      hash = (53 * hash) + distributeType_;
    }
    if (hasSbpParallel()) {
      hash = (37 * hash) + SBP_PARALLEL_FIELD_NUMBER;
      hash = (53 * hash) + getSbpParallel().hashCode();
    }
    if (hasDistributeNum()) {
      hash = (37 * hash) + DISTRIBUTE_NUM_FIELD_NUMBER;
      hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
          getDistributeNum());
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.DistributeDim parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.DistributeDim parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.DistributeDim parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.DistributeDim parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.DistributeDim parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.DistributeDim parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.DistributeDim parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.DistributeDim parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.DistributeDim parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.DistributeDim parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.DistributeDim prototype) {
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
   * Protobuf type {@code oneflow.DistributeDim}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.DistributeDim)
      org.oneflow.core.job.DistributeDimOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.DistributeHirarchyOuterClass.internal_static_oneflow_DistributeDim_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.DistributeHirarchyOuterClass.internal_static_oneflow_DistributeDim_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.DistributeDim.class, org.oneflow.core.job.DistributeDim.Builder.class);
    }

    // Construct using org.oneflow.core.job.DistributeDim.newBuilder()
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
        getSbpParallelFieldBuilder();
      }
    }
    public Builder clear() {
      super.clear();
      distributeType_ = 0;
      bitField0_ = (bitField0_ & ~0x00000001);
      if (sbpParallelBuilder_ == null) {
        sbpParallel_ = null;
      } else {
        sbpParallelBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000002);
      distributeNum_ = 0L;
      bitField0_ = (bitField0_ & ~0x00000004);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.DistributeHirarchyOuterClass.internal_static_oneflow_DistributeDim_descriptor;
    }

    public org.oneflow.core.job.DistributeDim getDefaultInstanceForType() {
      return org.oneflow.core.job.DistributeDim.getDefaultInstance();
    }

    public org.oneflow.core.job.DistributeDim build() {
      org.oneflow.core.job.DistributeDim result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.DistributeDim buildPartial() {
      org.oneflow.core.job.DistributeDim result = new org.oneflow.core.job.DistributeDim(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      result.distributeType_ = distributeType_;
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000002;
      }
      if (sbpParallelBuilder_ == null) {
        result.sbpParallel_ = sbpParallel_;
      } else {
        result.sbpParallel_ = sbpParallelBuilder_.build();
      }
      if (((from_bitField0_ & 0x00000004) == 0x00000004)) {
        to_bitField0_ |= 0x00000004;
      }
      result.distributeNum_ = distributeNum_;
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
      if (other instanceof org.oneflow.core.job.DistributeDim) {
        return mergeFrom((org.oneflow.core.job.DistributeDim)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.DistributeDim other) {
      if (other == org.oneflow.core.job.DistributeDim.getDefaultInstance()) return this;
      if (other.hasDistributeType()) {
        setDistributeType(other.getDistributeType());
      }
      if (other.hasSbpParallel()) {
        mergeSbpParallel(other.getSbpParallel());
      }
      if (other.hasDistributeNum()) {
        setDistributeNum(other.getDistributeNum());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (!hasDistributeType()) {
        return false;
      }
      if (!hasSbpParallel()) {
        return false;
      }
      if (!hasDistributeNum()) {
        return false;
      }
      if (!getSbpParallel().isInitialized()) {
        return false;
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.job.DistributeDim parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.DistributeDim) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private int distributeType_ = 0;
    /**
     * <code>required .oneflow.DistributeType distribute_type = 1;</code>
     */
    public boolean hasDistributeType() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>required .oneflow.DistributeType distribute_type = 1;</code>
     */
    public org.oneflow.core.job.DistributeType getDistributeType() {
      org.oneflow.core.job.DistributeType result = org.oneflow.core.job.DistributeType.valueOf(distributeType_);
      return result == null ? org.oneflow.core.job.DistributeType.kInvalidDistributeType : result;
    }
    /**
     * <code>required .oneflow.DistributeType distribute_type = 1;</code>
     */
    public Builder setDistributeType(org.oneflow.core.job.DistributeType value) {
      if (value == null) {
        throw new NullPointerException();
      }
      bitField0_ |= 0x00000001;
      distributeType_ = value.getNumber();
      onChanged();
      return this;
    }
    /**
     * <code>required .oneflow.DistributeType distribute_type = 1;</code>
     */
    public Builder clearDistributeType() {
      bitField0_ = (bitField0_ & ~0x00000001);
      distributeType_ = 0;
      onChanged();
      return this;
    }

    private org.oneflow.core.job.SbpParallel sbpParallel_ = null;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.job.SbpParallel, org.oneflow.core.job.SbpParallel.Builder, org.oneflow.core.job.SbpParallelOrBuilder> sbpParallelBuilder_;
    /**
     * <code>required .oneflow.SbpParallel sbp_parallel = 2;</code>
     */
    public boolean hasSbpParallel() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>required .oneflow.SbpParallel sbp_parallel = 2;</code>
     */
    public org.oneflow.core.job.SbpParallel getSbpParallel() {
      if (sbpParallelBuilder_ == null) {
        return sbpParallel_ == null ? org.oneflow.core.job.SbpParallel.getDefaultInstance() : sbpParallel_;
      } else {
        return sbpParallelBuilder_.getMessage();
      }
    }
    /**
     * <code>required .oneflow.SbpParallel sbp_parallel = 2;</code>
     */
    public Builder setSbpParallel(org.oneflow.core.job.SbpParallel value) {
      if (sbpParallelBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        sbpParallel_ = value;
        onChanged();
      } else {
        sbpParallelBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000002;
      return this;
    }
    /**
     * <code>required .oneflow.SbpParallel sbp_parallel = 2;</code>
     */
    public Builder setSbpParallel(
        org.oneflow.core.job.SbpParallel.Builder builderForValue) {
      if (sbpParallelBuilder_ == null) {
        sbpParallel_ = builderForValue.build();
        onChanged();
      } else {
        sbpParallelBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000002;
      return this;
    }
    /**
     * <code>required .oneflow.SbpParallel sbp_parallel = 2;</code>
     */
    public Builder mergeSbpParallel(org.oneflow.core.job.SbpParallel value) {
      if (sbpParallelBuilder_ == null) {
        if (((bitField0_ & 0x00000002) == 0x00000002) &&
            sbpParallel_ != null &&
            sbpParallel_ != org.oneflow.core.job.SbpParallel.getDefaultInstance()) {
          sbpParallel_ =
            org.oneflow.core.job.SbpParallel.newBuilder(sbpParallel_).mergeFrom(value).buildPartial();
        } else {
          sbpParallel_ = value;
        }
        onChanged();
      } else {
        sbpParallelBuilder_.mergeFrom(value);
      }
      bitField0_ |= 0x00000002;
      return this;
    }
    /**
     * <code>required .oneflow.SbpParallel sbp_parallel = 2;</code>
     */
    public Builder clearSbpParallel() {
      if (sbpParallelBuilder_ == null) {
        sbpParallel_ = null;
        onChanged();
      } else {
        sbpParallelBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000002);
      return this;
    }
    /**
     * <code>required .oneflow.SbpParallel sbp_parallel = 2;</code>
     */
    public org.oneflow.core.job.SbpParallel.Builder getSbpParallelBuilder() {
      bitField0_ |= 0x00000002;
      onChanged();
      return getSbpParallelFieldBuilder().getBuilder();
    }
    /**
     * <code>required .oneflow.SbpParallel sbp_parallel = 2;</code>
     */
    public org.oneflow.core.job.SbpParallelOrBuilder getSbpParallelOrBuilder() {
      if (sbpParallelBuilder_ != null) {
        return sbpParallelBuilder_.getMessageOrBuilder();
      } else {
        return sbpParallel_ == null ?
            org.oneflow.core.job.SbpParallel.getDefaultInstance() : sbpParallel_;
      }
    }
    /**
     * <code>required .oneflow.SbpParallel sbp_parallel = 2;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.job.SbpParallel, org.oneflow.core.job.SbpParallel.Builder, org.oneflow.core.job.SbpParallelOrBuilder> 
        getSbpParallelFieldBuilder() {
      if (sbpParallelBuilder_ == null) {
        sbpParallelBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.oneflow.core.job.SbpParallel, org.oneflow.core.job.SbpParallel.Builder, org.oneflow.core.job.SbpParallelOrBuilder>(
                getSbpParallel(),
                getParentForChildren(),
                isClean());
        sbpParallel_ = null;
      }
      return sbpParallelBuilder_;
    }

    private long distributeNum_ ;
    /**
     * <code>required int64 distribute_num = 3;</code>
     */
    public boolean hasDistributeNum() {
      return ((bitField0_ & 0x00000004) == 0x00000004);
    }
    /**
     * <code>required int64 distribute_num = 3;</code>
     */
    public long getDistributeNum() {
      return distributeNum_;
    }
    /**
     * <code>required int64 distribute_num = 3;</code>
     */
    public Builder setDistributeNum(long value) {
      bitField0_ |= 0x00000004;
      distributeNum_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>required int64 distribute_num = 3;</code>
     */
    public Builder clearDistributeNum() {
      bitField0_ = (bitField0_ & ~0x00000004);
      distributeNum_ = 0L;
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


    // @@protoc_insertion_point(builder_scope:oneflow.DistributeDim)
  }

  // @@protoc_insertion_point(class_scope:oneflow.DistributeDim)
  private static final org.oneflow.core.job.DistributeDim DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.DistributeDim();
  }

  public static org.oneflow.core.job.DistributeDim getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<DistributeDim>
      PARSER = new com.google.protobuf.AbstractParser<DistributeDim>() {
    public DistributeDim parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new DistributeDim(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<DistributeDim> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<DistributeDim> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.DistributeDim getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

