// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/job_conf.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.ParallelBlobConf}
 */
public  final class ParallelBlobConf extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.ParallelBlobConf)
    ParallelBlobConfOrBuilder {
  // Use ParallelBlobConf.newBuilder() to construct.
  private ParallelBlobConf(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private ParallelBlobConf() {
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private ParallelBlobConf(
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
          case 10: {
            org.oneflow.core.register.BlobDescProto.Builder subBuilder = null;
            if (((bitField0_ & 0x00000001) == 0x00000001)) {
              subBuilder = logicalBlobDescConf_.toBuilder();
            }
            logicalBlobDescConf_ = input.readMessage(org.oneflow.core.register.BlobDescProto.PARSER, extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(logicalBlobDescConf_);
              logicalBlobDescConf_ = subBuilder.buildPartial();
            }
            bitField0_ |= 0x00000001;
            break;
          }
          case 18: {
            org.oneflow.core.job.ParallelConf.Builder subBuilder = null;
            if (((bitField0_ & 0x00000002) == 0x00000002)) {
              subBuilder = parallelConf_.toBuilder();
            }
            parallelConf_ = input.readMessage(org.oneflow.core.job.ParallelConf.PARSER, extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(parallelConf_);
              parallelConf_ = subBuilder.buildPartial();
            }
            bitField0_ |= 0x00000002;
            break;
          }
          case 26: {
            org.oneflow.core.job.ParallelDistribution.Builder subBuilder = null;
            if (((bitField0_ & 0x00000004) == 0x00000004)) {
              subBuilder = parallelDistribution_.toBuilder();
            }
            parallelDistribution_ = input.readMessage(org.oneflow.core.job.ParallelDistribution.PARSER, extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(parallelDistribution_);
              parallelDistribution_ = subBuilder.buildPartial();
            }
            bitField0_ |= 0x00000004;
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
    return org.oneflow.core.job.JobConf.internal_static_oneflow_ParallelBlobConf_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.JobConf.internal_static_oneflow_ParallelBlobConf_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.ParallelBlobConf.class, org.oneflow.core.job.ParallelBlobConf.Builder.class);
  }

  private int bitField0_;
  public static final int LOGICAL_BLOB_DESC_CONF_FIELD_NUMBER = 1;
  private org.oneflow.core.register.BlobDescProto logicalBlobDescConf_;
  /**
   * <code>required .oneflow.BlobDescProto logical_blob_desc_conf = 1;</code>
   */
  public boolean hasLogicalBlobDescConf() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>required .oneflow.BlobDescProto logical_blob_desc_conf = 1;</code>
   */
  public org.oneflow.core.register.BlobDescProto getLogicalBlobDescConf() {
    return logicalBlobDescConf_ == null ? org.oneflow.core.register.BlobDescProto.getDefaultInstance() : logicalBlobDescConf_;
  }
  /**
   * <code>required .oneflow.BlobDescProto logical_blob_desc_conf = 1;</code>
   */
  public org.oneflow.core.register.BlobDescProtoOrBuilder getLogicalBlobDescConfOrBuilder() {
    return logicalBlobDescConf_ == null ? org.oneflow.core.register.BlobDescProto.getDefaultInstance() : logicalBlobDescConf_;
  }

  public static final int PARALLEL_CONF_FIELD_NUMBER = 2;
  private org.oneflow.core.job.ParallelConf parallelConf_;
  /**
   * <code>required .oneflow.ParallelConf parallel_conf = 2;</code>
   */
  public boolean hasParallelConf() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>required .oneflow.ParallelConf parallel_conf = 2;</code>
   */
  public org.oneflow.core.job.ParallelConf getParallelConf() {
    return parallelConf_ == null ? org.oneflow.core.job.ParallelConf.getDefaultInstance() : parallelConf_;
  }
  /**
   * <code>required .oneflow.ParallelConf parallel_conf = 2;</code>
   */
  public org.oneflow.core.job.ParallelConfOrBuilder getParallelConfOrBuilder() {
    return parallelConf_ == null ? org.oneflow.core.job.ParallelConf.getDefaultInstance() : parallelConf_;
  }

  public static final int PARALLEL_DISTRIBUTION_FIELD_NUMBER = 3;
  private org.oneflow.core.job.ParallelDistribution parallelDistribution_;
  /**
   * <code>required .oneflow.ParallelDistribution parallel_distribution = 3;</code>
   */
  public boolean hasParallelDistribution() {
    return ((bitField0_ & 0x00000004) == 0x00000004);
  }
  /**
   * <code>required .oneflow.ParallelDistribution parallel_distribution = 3;</code>
   */
  public org.oneflow.core.job.ParallelDistribution getParallelDistribution() {
    return parallelDistribution_ == null ? org.oneflow.core.job.ParallelDistribution.getDefaultInstance() : parallelDistribution_;
  }
  /**
   * <code>required .oneflow.ParallelDistribution parallel_distribution = 3;</code>
   */
  public org.oneflow.core.job.ParallelDistributionOrBuilder getParallelDistributionOrBuilder() {
    return parallelDistribution_ == null ? org.oneflow.core.job.ParallelDistribution.getDefaultInstance() : parallelDistribution_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (!hasLogicalBlobDescConf()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!hasParallelConf()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!hasParallelDistribution()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!getLogicalBlobDescConf().isInitialized()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!getParallelConf().isInitialized()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!getParallelDistribution().isInitialized()) {
      memoizedIsInitialized = 0;
      return false;
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      output.writeMessage(1, getLogicalBlobDescConf());
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      output.writeMessage(2, getParallelConf());
    }
    if (((bitField0_ & 0x00000004) == 0x00000004)) {
      output.writeMessage(3, getParallelDistribution());
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(1, getLogicalBlobDescConf());
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(2, getParallelConf());
    }
    if (((bitField0_ & 0x00000004) == 0x00000004)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(3, getParallelDistribution());
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
    if (!(obj instanceof org.oneflow.core.job.ParallelBlobConf)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.ParallelBlobConf other = (org.oneflow.core.job.ParallelBlobConf) obj;

    boolean result = true;
    result = result && (hasLogicalBlobDescConf() == other.hasLogicalBlobDescConf());
    if (hasLogicalBlobDescConf()) {
      result = result && getLogicalBlobDescConf()
          .equals(other.getLogicalBlobDescConf());
    }
    result = result && (hasParallelConf() == other.hasParallelConf());
    if (hasParallelConf()) {
      result = result && getParallelConf()
          .equals(other.getParallelConf());
    }
    result = result && (hasParallelDistribution() == other.hasParallelDistribution());
    if (hasParallelDistribution()) {
      result = result && getParallelDistribution()
          .equals(other.getParallelDistribution());
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
    if (hasLogicalBlobDescConf()) {
      hash = (37 * hash) + LOGICAL_BLOB_DESC_CONF_FIELD_NUMBER;
      hash = (53 * hash) + getLogicalBlobDescConf().hashCode();
    }
    if (hasParallelConf()) {
      hash = (37 * hash) + PARALLEL_CONF_FIELD_NUMBER;
      hash = (53 * hash) + getParallelConf().hashCode();
    }
    if (hasParallelDistribution()) {
      hash = (37 * hash) + PARALLEL_DISTRIBUTION_FIELD_NUMBER;
      hash = (53 * hash) + getParallelDistribution().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.ParallelBlobConf parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.ParallelBlobConf parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.ParallelBlobConf parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.ParallelBlobConf parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.ParallelBlobConf parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.ParallelBlobConf parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.ParallelBlobConf parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.ParallelBlobConf parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.ParallelBlobConf parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.ParallelBlobConf parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.ParallelBlobConf prototype) {
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
   * Protobuf type {@code oneflow.ParallelBlobConf}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.ParallelBlobConf)
      org.oneflow.core.job.ParallelBlobConfOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.JobConf.internal_static_oneflow_ParallelBlobConf_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.JobConf.internal_static_oneflow_ParallelBlobConf_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.ParallelBlobConf.class, org.oneflow.core.job.ParallelBlobConf.Builder.class);
    }

    // Construct using org.oneflow.core.job.ParallelBlobConf.newBuilder()
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
        getLogicalBlobDescConfFieldBuilder();
        getParallelConfFieldBuilder();
        getParallelDistributionFieldBuilder();
      }
    }
    public Builder clear() {
      super.clear();
      if (logicalBlobDescConfBuilder_ == null) {
        logicalBlobDescConf_ = null;
      } else {
        logicalBlobDescConfBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000001);
      if (parallelConfBuilder_ == null) {
        parallelConf_ = null;
      } else {
        parallelConfBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000002);
      if (parallelDistributionBuilder_ == null) {
        parallelDistribution_ = null;
      } else {
        parallelDistributionBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000004);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.JobConf.internal_static_oneflow_ParallelBlobConf_descriptor;
    }

    public org.oneflow.core.job.ParallelBlobConf getDefaultInstanceForType() {
      return org.oneflow.core.job.ParallelBlobConf.getDefaultInstance();
    }

    public org.oneflow.core.job.ParallelBlobConf build() {
      org.oneflow.core.job.ParallelBlobConf result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.ParallelBlobConf buildPartial() {
      org.oneflow.core.job.ParallelBlobConf result = new org.oneflow.core.job.ParallelBlobConf(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      if (logicalBlobDescConfBuilder_ == null) {
        result.logicalBlobDescConf_ = logicalBlobDescConf_;
      } else {
        result.logicalBlobDescConf_ = logicalBlobDescConfBuilder_.build();
      }
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000002;
      }
      if (parallelConfBuilder_ == null) {
        result.parallelConf_ = parallelConf_;
      } else {
        result.parallelConf_ = parallelConfBuilder_.build();
      }
      if (((from_bitField0_ & 0x00000004) == 0x00000004)) {
        to_bitField0_ |= 0x00000004;
      }
      if (parallelDistributionBuilder_ == null) {
        result.parallelDistribution_ = parallelDistribution_;
      } else {
        result.parallelDistribution_ = parallelDistributionBuilder_.build();
      }
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
      if (other instanceof org.oneflow.core.job.ParallelBlobConf) {
        return mergeFrom((org.oneflow.core.job.ParallelBlobConf)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.ParallelBlobConf other) {
      if (other == org.oneflow.core.job.ParallelBlobConf.getDefaultInstance()) return this;
      if (other.hasLogicalBlobDescConf()) {
        mergeLogicalBlobDescConf(other.getLogicalBlobDescConf());
      }
      if (other.hasParallelConf()) {
        mergeParallelConf(other.getParallelConf());
      }
      if (other.hasParallelDistribution()) {
        mergeParallelDistribution(other.getParallelDistribution());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (!hasLogicalBlobDescConf()) {
        return false;
      }
      if (!hasParallelConf()) {
        return false;
      }
      if (!hasParallelDistribution()) {
        return false;
      }
      if (!getLogicalBlobDescConf().isInitialized()) {
        return false;
      }
      if (!getParallelConf().isInitialized()) {
        return false;
      }
      if (!getParallelDistribution().isInitialized()) {
        return false;
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.job.ParallelBlobConf parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.ParallelBlobConf) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private org.oneflow.core.register.BlobDescProto logicalBlobDescConf_ = null;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.register.BlobDescProto, org.oneflow.core.register.BlobDescProto.Builder, org.oneflow.core.register.BlobDescProtoOrBuilder> logicalBlobDescConfBuilder_;
    /**
     * <code>required .oneflow.BlobDescProto logical_blob_desc_conf = 1;</code>
     */
    public boolean hasLogicalBlobDescConf() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>required .oneflow.BlobDescProto logical_blob_desc_conf = 1;</code>
     */
    public org.oneflow.core.register.BlobDescProto getLogicalBlobDescConf() {
      if (logicalBlobDescConfBuilder_ == null) {
        return logicalBlobDescConf_ == null ? org.oneflow.core.register.BlobDescProto.getDefaultInstance() : logicalBlobDescConf_;
      } else {
        return logicalBlobDescConfBuilder_.getMessage();
      }
    }
    /**
     * <code>required .oneflow.BlobDescProto logical_blob_desc_conf = 1;</code>
     */
    public Builder setLogicalBlobDescConf(org.oneflow.core.register.BlobDescProto value) {
      if (logicalBlobDescConfBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        logicalBlobDescConf_ = value;
        onChanged();
      } else {
        logicalBlobDescConfBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>required .oneflow.BlobDescProto logical_blob_desc_conf = 1;</code>
     */
    public Builder setLogicalBlobDescConf(
        org.oneflow.core.register.BlobDescProto.Builder builderForValue) {
      if (logicalBlobDescConfBuilder_ == null) {
        logicalBlobDescConf_ = builderForValue.build();
        onChanged();
      } else {
        logicalBlobDescConfBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>required .oneflow.BlobDescProto logical_blob_desc_conf = 1;</code>
     */
    public Builder mergeLogicalBlobDescConf(org.oneflow.core.register.BlobDescProto value) {
      if (logicalBlobDescConfBuilder_ == null) {
        if (((bitField0_ & 0x00000001) == 0x00000001) &&
            logicalBlobDescConf_ != null &&
            logicalBlobDescConf_ != org.oneflow.core.register.BlobDescProto.getDefaultInstance()) {
          logicalBlobDescConf_ =
            org.oneflow.core.register.BlobDescProto.newBuilder(logicalBlobDescConf_).mergeFrom(value).buildPartial();
        } else {
          logicalBlobDescConf_ = value;
        }
        onChanged();
      } else {
        logicalBlobDescConfBuilder_.mergeFrom(value);
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>required .oneflow.BlobDescProto logical_blob_desc_conf = 1;</code>
     */
    public Builder clearLogicalBlobDescConf() {
      if (logicalBlobDescConfBuilder_ == null) {
        logicalBlobDescConf_ = null;
        onChanged();
      } else {
        logicalBlobDescConfBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000001);
      return this;
    }
    /**
     * <code>required .oneflow.BlobDescProto logical_blob_desc_conf = 1;</code>
     */
    public org.oneflow.core.register.BlobDescProto.Builder getLogicalBlobDescConfBuilder() {
      bitField0_ |= 0x00000001;
      onChanged();
      return getLogicalBlobDescConfFieldBuilder().getBuilder();
    }
    /**
     * <code>required .oneflow.BlobDescProto logical_blob_desc_conf = 1;</code>
     */
    public org.oneflow.core.register.BlobDescProtoOrBuilder getLogicalBlobDescConfOrBuilder() {
      if (logicalBlobDescConfBuilder_ != null) {
        return logicalBlobDescConfBuilder_.getMessageOrBuilder();
      } else {
        return logicalBlobDescConf_ == null ?
            org.oneflow.core.register.BlobDescProto.getDefaultInstance() : logicalBlobDescConf_;
      }
    }
    /**
     * <code>required .oneflow.BlobDescProto logical_blob_desc_conf = 1;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.register.BlobDescProto, org.oneflow.core.register.BlobDescProto.Builder, org.oneflow.core.register.BlobDescProtoOrBuilder> 
        getLogicalBlobDescConfFieldBuilder() {
      if (logicalBlobDescConfBuilder_ == null) {
        logicalBlobDescConfBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.oneflow.core.register.BlobDescProto, org.oneflow.core.register.BlobDescProto.Builder, org.oneflow.core.register.BlobDescProtoOrBuilder>(
                getLogicalBlobDescConf(),
                getParentForChildren(),
                isClean());
        logicalBlobDescConf_ = null;
      }
      return logicalBlobDescConfBuilder_;
    }

    private org.oneflow.core.job.ParallelConf parallelConf_ = null;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.job.ParallelConf, org.oneflow.core.job.ParallelConf.Builder, org.oneflow.core.job.ParallelConfOrBuilder> parallelConfBuilder_;
    /**
     * <code>required .oneflow.ParallelConf parallel_conf = 2;</code>
     */
    public boolean hasParallelConf() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>required .oneflow.ParallelConf parallel_conf = 2;</code>
     */
    public org.oneflow.core.job.ParallelConf getParallelConf() {
      if (parallelConfBuilder_ == null) {
        return parallelConf_ == null ? org.oneflow.core.job.ParallelConf.getDefaultInstance() : parallelConf_;
      } else {
        return parallelConfBuilder_.getMessage();
      }
    }
    /**
     * <code>required .oneflow.ParallelConf parallel_conf = 2;</code>
     */
    public Builder setParallelConf(org.oneflow.core.job.ParallelConf value) {
      if (parallelConfBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        parallelConf_ = value;
        onChanged();
      } else {
        parallelConfBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000002;
      return this;
    }
    /**
     * <code>required .oneflow.ParallelConf parallel_conf = 2;</code>
     */
    public Builder setParallelConf(
        org.oneflow.core.job.ParallelConf.Builder builderForValue) {
      if (parallelConfBuilder_ == null) {
        parallelConf_ = builderForValue.build();
        onChanged();
      } else {
        parallelConfBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000002;
      return this;
    }
    /**
     * <code>required .oneflow.ParallelConf parallel_conf = 2;</code>
     */
    public Builder mergeParallelConf(org.oneflow.core.job.ParallelConf value) {
      if (parallelConfBuilder_ == null) {
        if (((bitField0_ & 0x00000002) == 0x00000002) &&
            parallelConf_ != null &&
            parallelConf_ != org.oneflow.core.job.ParallelConf.getDefaultInstance()) {
          parallelConf_ =
            org.oneflow.core.job.ParallelConf.newBuilder(parallelConf_).mergeFrom(value).buildPartial();
        } else {
          parallelConf_ = value;
        }
        onChanged();
      } else {
        parallelConfBuilder_.mergeFrom(value);
      }
      bitField0_ |= 0x00000002;
      return this;
    }
    /**
     * <code>required .oneflow.ParallelConf parallel_conf = 2;</code>
     */
    public Builder clearParallelConf() {
      if (parallelConfBuilder_ == null) {
        parallelConf_ = null;
        onChanged();
      } else {
        parallelConfBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000002);
      return this;
    }
    /**
     * <code>required .oneflow.ParallelConf parallel_conf = 2;</code>
     */
    public org.oneflow.core.job.ParallelConf.Builder getParallelConfBuilder() {
      bitField0_ |= 0x00000002;
      onChanged();
      return getParallelConfFieldBuilder().getBuilder();
    }
    /**
     * <code>required .oneflow.ParallelConf parallel_conf = 2;</code>
     */
    public org.oneflow.core.job.ParallelConfOrBuilder getParallelConfOrBuilder() {
      if (parallelConfBuilder_ != null) {
        return parallelConfBuilder_.getMessageOrBuilder();
      } else {
        return parallelConf_ == null ?
            org.oneflow.core.job.ParallelConf.getDefaultInstance() : parallelConf_;
      }
    }
    /**
     * <code>required .oneflow.ParallelConf parallel_conf = 2;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.job.ParallelConf, org.oneflow.core.job.ParallelConf.Builder, org.oneflow.core.job.ParallelConfOrBuilder> 
        getParallelConfFieldBuilder() {
      if (parallelConfBuilder_ == null) {
        parallelConfBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.oneflow.core.job.ParallelConf, org.oneflow.core.job.ParallelConf.Builder, org.oneflow.core.job.ParallelConfOrBuilder>(
                getParallelConf(),
                getParentForChildren(),
                isClean());
        parallelConf_ = null;
      }
      return parallelConfBuilder_;
    }

    private org.oneflow.core.job.ParallelDistribution parallelDistribution_ = null;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.job.ParallelDistribution, org.oneflow.core.job.ParallelDistribution.Builder, org.oneflow.core.job.ParallelDistributionOrBuilder> parallelDistributionBuilder_;
    /**
     * <code>required .oneflow.ParallelDistribution parallel_distribution = 3;</code>
     */
    public boolean hasParallelDistribution() {
      return ((bitField0_ & 0x00000004) == 0x00000004);
    }
    /**
     * <code>required .oneflow.ParallelDistribution parallel_distribution = 3;</code>
     */
    public org.oneflow.core.job.ParallelDistribution getParallelDistribution() {
      if (parallelDistributionBuilder_ == null) {
        return parallelDistribution_ == null ? org.oneflow.core.job.ParallelDistribution.getDefaultInstance() : parallelDistribution_;
      } else {
        return parallelDistributionBuilder_.getMessage();
      }
    }
    /**
     * <code>required .oneflow.ParallelDistribution parallel_distribution = 3;</code>
     */
    public Builder setParallelDistribution(org.oneflow.core.job.ParallelDistribution value) {
      if (parallelDistributionBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        parallelDistribution_ = value;
        onChanged();
      } else {
        parallelDistributionBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000004;
      return this;
    }
    /**
     * <code>required .oneflow.ParallelDistribution parallel_distribution = 3;</code>
     */
    public Builder setParallelDistribution(
        org.oneflow.core.job.ParallelDistribution.Builder builderForValue) {
      if (parallelDistributionBuilder_ == null) {
        parallelDistribution_ = builderForValue.build();
        onChanged();
      } else {
        parallelDistributionBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000004;
      return this;
    }
    /**
     * <code>required .oneflow.ParallelDistribution parallel_distribution = 3;</code>
     */
    public Builder mergeParallelDistribution(org.oneflow.core.job.ParallelDistribution value) {
      if (parallelDistributionBuilder_ == null) {
        if (((bitField0_ & 0x00000004) == 0x00000004) &&
            parallelDistribution_ != null &&
            parallelDistribution_ != org.oneflow.core.job.ParallelDistribution.getDefaultInstance()) {
          parallelDistribution_ =
            org.oneflow.core.job.ParallelDistribution.newBuilder(parallelDistribution_).mergeFrom(value).buildPartial();
        } else {
          parallelDistribution_ = value;
        }
        onChanged();
      } else {
        parallelDistributionBuilder_.mergeFrom(value);
      }
      bitField0_ |= 0x00000004;
      return this;
    }
    /**
     * <code>required .oneflow.ParallelDistribution parallel_distribution = 3;</code>
     */
    public Builder clearParallelDistribution() {
      if (parallelDistributionBuilder_ == null) {
        parallelDistribution_ = null;
        onChanged();
      } else {
        parallelDistributionBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000004);
      return this;
    }
    /**
     * <code>required .oneflow.ParallelDistribution parallel_distribution = 3;</code>
     */
    public org.oneflow.core.job.ParallelDistribution.Builder getParallelDistributionBuilder() {
      bitField0_ |= 0x00000004;
      onChanged();
      return getParallelDistributionFieldBuilder().getBuilder();
    }
    /**
     * <code>required .oneflow.ParallelDistribution parallel_distribution = 3;</code>
     */
    public org.oneflow.core.job.ParallelDistributionOrBuilder getParallelDistributionOrBuilder() {
      if (parallelDistributionBuilder_ != null) {
        return parallelDistributionBuilder_.getMessageOrBuilder();
      } else {
        return parallelDistribution_ == null ?
            org.oneflow.core.job.ParallelDistribution.getDefaultInstance() : parallelDistribution_;
      }
    }
    /**
     * <code>required .oneflow.ParallelDistribution parallel_distribution = 3;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.job.ParallelDistribution, org.oneflow.core.job.ParallelDistribution.Builder, org.oneflow.core.job.ParallelDistributionOrBuilder> 
        getParallelDistributionFieldBuilder() {
      if (parallelDistributionBuilder_ == null) {
        parallelDistributionBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.oneflow.core.job.ParallelDistribution, org.oneflow.core.job.ParallelDistribution.Builder, org.oneflow.core.job.ParallelDistributionOrBuilder>(
                getParallelDistribution(),
                getParentForChildren(),
                isClean());
        parallelDistribution_ = null;
      }
      return parallelDistributionBuilder_;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:oneflow.ParallelBlobConf)
  }

  // @@protoc_insertion_point(class_scope:oneflow.ParallelBlobConf)
  private static final org.oneflow.core.job.ParallelBlobConf DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.ParallelBlobConf();
  }

  public static org.oneflow.core.job.ParallelBlobConf getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<ParallelBlobConf>
      PARSER = new com.google.protobuf.AbstractParser<ParallelBlobConf>() {
    public ParallelBlobConf parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new ParallelBlobConf(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<ParallelBlobConf> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<ParallelBlobConf> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.ParallelBlobConf getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

