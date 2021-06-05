// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/job_conf.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.JobOutputDef}
 */
public  final class JobOutputDef extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.JobOutputDef)
    JobOutputDefOrBuilder {
  // Use JobOutputDef.newBuilder() to construct.
  private JobOutputDef(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private JobOutputDef() {
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private JobOutputDef(
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
            org.oneflow.core.register.LogicalBlobId.Builder subBuilder = null;
            if (((bitField0_ & 0x00000001) == 0x00000001)) {
              subBuilder = lbi_.toBuilder();
            }
            lbi_ = input.readMessage(org.oneflow.core.register.LogicalBlobId.PARSER, extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(lbi_);
              lbi_ = subBuilder.buildPartial();
            }
            bitField0_ |= 0x00000001;
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
    return org.oneflow.core.job.JobConf.internal_static_oneflow_JobOutputDef_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.JobConf.internal_static_oneflow_JobOutputDef_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.JobOutputDef.class, org.oneflow.core.job.JobOutputDef.Builder.class);
  }

  private int bitField0_;
  public static final int LBI_FIELD_NUMBER = 1;
  private org.oneflow.core.register.LogicalBlobId lbi_;
  /**
   * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
   */
  public boolean hasLbi() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
   */
  public org.oneflow.core.register.LogicalBlobId getLbi() {
    return lbi_ == null ? org.oneflow.core.register.LogicalBlobId.getDefaultInstance() : lbi_;
  }
  /**
   * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
   */
  public org.oneflow.core.register.LogicalBlobIdOrBuilder getLbiOrBuilder() {
    return lbi_ == null ? org.oneflow.core.register.LogicalBlobId.getDefaultInstance() : lbi_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (!hasLbi()) {
      memoizedIsInitialized = 0;
      return false;
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      output.writeMessage(1, getLbi());
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(1, getLbi());
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
    if (!(obj instanceof org.oneflow.core.job.JobOutputDef)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.JobOutputDef other = (org.oneflow.core.job.JobOutputDef) obj;

    boolean result = true;
    result = result && (hasLbi() == other.hasLbi());
    if (hasLbi()) {
      result = result && getLbi()
          .equals(other.getLbi());
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
    if (hasLbi()) {
      hash = (37 * hash) + LBI_FIELD_NUMBER;
      hash = (53 * hash) + getLbi().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.JobOutputDef parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.JobOutputDef parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.JobOutputDef parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.JobOutputDef parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.JobOutputDef parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.JobOutputDef parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.JobOutputDef parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.JobOutputDef parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.JobOutputDef parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.JobOutputDef parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.JobOutputDef prototype) {
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
   * Protobuf type {@code oneflow.JobOutputDef}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.JobOutputDef)
      org.oneflow.core.job.JobOutputDefOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.JobConf.internal_static_oneflow_JobOutputDef_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.JobConf.internal_static_oneflow_JobOutputDef_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.JobOutputDef.class, org.oneflow.core.job.JobOutputDef.Builder.class);
    }

    // Construct using org.oneflow.core.job.JobOutputDef.newBuilder()
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
        getLbiFieldBuilder();
      }
    }
    public Builder clear() {
      super.clear();
      if (lbiBuilder_ == null) {
        lbi_ = null;
      } else {
        lbiBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000001);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.JobConf.internal_static_oneflow_JobOutputDef_descriptor;
    }

    public org.oneflow.core.job.JobOutputDef getDefaultInstanceForType() {
      return org.oneflow.core.job.JobOutputDef.getDefaultInstance();
    }

    public org.oneflow.core.job.JobOutputDef build() {
      org.oneflow.core.job.JobOutputDef result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.JobOutputDef buildPartial() {
      org.oneflow.core.job.JobOutputDef result = new org.oneflow.core.job.JobOutputDef(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      if (lbiBuilder_ == null) {
        result.lbi_ = lbi_;
      } else {
        result.lbi_ = lbiBuilder_.build();
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
      if (other instanceof org.oneflow.core.job.JobOutputDef) {
        return mergeFrom((org.oneflow.core.job.JobOutputDef)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.JobOutputDef other) {
      if (other == org.oneflow.core.job.JobOutputDef.getDefaultInstance()) return this;
      if (other.hasLbi()) {
        mergeLbi(other.getLbi());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (!hasLbi()) {
        return false;
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.job.JobOutputDef parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.JobOutputDef) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private org.oneflow.core.register.LogicalBlobId lbi_ = null;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.register.LogicalBlobId, org.oneflow.core.register.LogicalBlobId.Builder, org.oneflow.core.register.LogicalBlobIdOrBuilder> lbiBuilder_;
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    public boolean hasLbi() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    public org.oneflow.core.register.LogicalBlobId getLbi() {
      if (lbiBuilder_ == null) {
        return lbi_ == null ? org.oneflow.core.register.LogicalBlobId.getDefaultInstance() : lbi_;
      } else {
        return lbiBuilder_.getMessage();
      }
    }
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    public Builder setLbi(org.oneflow.core.register.LogicalBlobId value) {
      if (lbiBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        lbi_ = value;
        onChanged();
      } else {
        lbiBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    public Builder setLbi(
        org.oneflow.core.register.LogicalBlobId.Builder builderForValue) {
      if (lbiBuilder_ == null) {
        lbi_ = builderForValue.build();
        onChanged();
      } else {
        lbiBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    public Builder mergeLbi(org.oneflow.core.register.LogicalBlobId value) {
      if (lbiBuilder_ == null) {
        if (((bitField0_ & 0x00000001) == 0x00000001) &&
            lbi_ != null &&
            lbi_ != org.oneflow.core.register.LogicalBlobId.getDefaultInstance()) {
          lbi_ =
            org.oneflow.core.register.LogicalBlobId.newBuilder(lbi_).mergeFrom(value).buildPartial();
        } else {
          lbi_ = value;
        }
        onChanged();
      } else {
        lbiBuilder_.mergeFrom(value);
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    public Builder clearLbi() {
      if (lbiBuilder_ == null) {
        lbi_ = null;
        onChanged();
      } else {
        lbiBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000001);
      return this;
    }
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    public org.oneflow.core.register.LogicalBlobId.Builder getLbiBuilder() {
      bitField0_ |= 0x00000001;
      onChanged();
      return getLbiFieldBuilder().getBuilder();
    }
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    public org.oneflow.core.register.LogicalBlobIdOrBuilder getLbiOrBuilder() {
      if (lbiBuilder_ != null) {
        return lbiBuilder_.getMessageOrBuilder();
      } else {
        return lbi_ == null ?
            org.oneflow.core.register.LogicalBlobId.getDefaultInstance() : lbi_;
      }
    }
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.register.LogicalBlobId, org.oneflow.core.register.LogicalBlobId.Builder, org.oneflow.core.register.LogicalBlobIdOrBuilder> 
        getLbiFieldBuilder() {
      if (lbiBuilder_ == null) {
        lbiBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.oneflow.core.register.LogicalBlobId, org.oneflow.core.register.LogicalBlobId.Builder, org.oneflow.core.register.LogicalBlobIdOrBuilder>(
                getLbi(),
                getParentForChildren(),
                isClean());
        lbi_ = null;
      }
      return lbiBuilder_;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:oneflow.JobOutputDef)
  }

  // @@protoc_insertion_point(class_scope:oneflow.JobOutputDef)
  private static final org.oneflow.core.job.JobOutputDef DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.JobOutputDef();
  }

  public static org.oneflow.core.job.JobOutputDef getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<JobOutputDef>
      PARSER = new com.google.protobuf.AbstractParser<JobOutputDef>() {
    public JobOutputDef parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new JobOutputDef(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<JobOutputDef> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<JobOutputDef> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.JobOutputDef getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
