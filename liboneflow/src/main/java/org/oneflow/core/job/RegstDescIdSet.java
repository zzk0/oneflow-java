// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/task.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.RegstDescIdSet}
 */
public  final class RegstDescIdSet extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.RegstDescIdSet)
    RegstDescIdSetOrBuilder {
  // Use RegstDescIdSet.newBuilder() to construct.
  private RegstDescIdSet(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private RegstDescIdSet() {
    regstDescId_ = java.util.Collections.emptyList();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private RegstDescIdSet(
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
            if (!((mutable_bitField0_ & 0x00000001) == 0x00000001)) {
              regstDescId_ = new java.util.ArrayList<java.lang.Long>();
              mutable_bitField0_ |= 0x00000001;
            }
            regstDescId_.add(input.readInt64());
            break;
          }
          case 10: {
            int length = input.readRawVarint32();
            int limit = input.pushLimit(length);
            if (!((mutable_bitField0_ & 0x00000001) == 0x00000001) && input.getBytesUntilLimit() > 0) {
              regstDescId_ = new java.util.ArrayList<java.lang.Long>();
              mutable_bitField0_ |= 0x00000001;
            }
            while (input.getBytesUntilLimit() > 0) {
              regstDescId_.add(input.readInt64());
            }
            input.popLimit(limit);
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
      if (((mutable_bitField0_ & 0x00000001) == 0x00000001)) {
        regstDescId_ = java.util.Collections.unmodifiableList(regstDescId_);
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.oneflow.core.job.Task.internal_static_oneflow_RegstDescIdSet_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.Task.internal_static_oneflow_RegstDescIdSet_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.RegstDescIdSet.class, org.oneflow.core.job.RegstDescIdSet.Builder.class);
  }

  public static final int REGST_DESC_ID_FIELD_NUMBER = 1;
  private java.util.List<java.lang.Long> regstDescId_;
  /**
   * <code>repeated int64 regst_desc_id = 1;</code>
   */
  public java.util.List<java.lang.Long>
      getRegstDescIdList() {
    return regstDescId_;
  }
  /**
   * <code>repeated int64 regst_desc_id = 1;</code>
   */
  public int getRegstDescIdCount() {
    return regstDescId_.size();
  }
  /**
   * <code>repeated int64 regst_desc_id = 1;</code>
   */
  public long getRegstDescId(int index) {
    return regstDescId_.get(index);
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
    for (int i = 0; i < regstDescId_.size(); i++) {
      output.writeInt64(1, regstDescId_.get(i));
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    {
      int dataSize = 0;
      for (int i = 0; i < regstDescId_.size(); i++) {
        dataSize += com.google.protobuf.CodedOutputStream
          .computeInt64SizeNoTag(regstDescId_.get(i));
      }
      size += dataSize;
      size += 1 * getRegstDescIdList().size();
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
    if (!(obj instanceof org.oneflow.core.job.RegstDescIdSet)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.RegstDescIdSet other = (org.oneflow.core.job.RegstDescIdSet) obj;

    boolean result = true;
    result = result && getRegstDescIdList()
        .equals(other.getRegstDescIdList());
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
    if (getRegstDescIdCount() > 0) {
      hash = (37 * hash) + REGST_DESC_ID_FIELD_NUMBER;
      hash = (53 * hash) + getRegstDescIdList().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.RegstDescIdSet parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.RegstDescIdSet parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.RegstDescIdSet parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.RegstDescIdSet parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.RegstDescIdSet parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.RegstDescIdSet parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.RegstDescIdSet parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.RegstDescIdSet parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.RegstDescIdSet parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.RegstDescIdSet parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.RegstDescIdSet prototype) {
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
   * Protobuf type {@code oneflow.RegstDescIdSet}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.RegstDescIdSet)
      org.oneflow.core.job.RegstDescIdSetOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.Task.internal_static_oneflow_RegstDescIdSet_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.Task.internal_static_oneflow_RegstDescIdSet_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.RegstDescIdSet.class, org.oneflow.core.job.RegstDescIdSet.Builder.class);
    }

    // Construct using org.oneflow.core.job.RegstDescIdSet.newBuilder()
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
      regstDescId_ = java.util.Collections.emptyList();
      bitField0_ = (bitField0_ & ~0x00000001);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.Task.internal_static_oneflow_RegstDescIdSet_descriptor;
    }

    public org.oneflow.core.job.RegstDescIdSet getDefaultInstanceForType() {
      return org.oneflow.core.job.RegstDescIdSet.getDefaultInstance();
    }

    public org.oneflow.core.job.RegstDescIdSet build() {
      org.oneflow.core.job.RegstDescIdSet result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.RegstDescIdSet buildPartial() {
      org.oneflow.core.job.RegstDescIdSet result = new org.oneflow.core.job.RegstDescIdSet(this);
      int from_bitField0_ = bitField0_;
      if (((bitField0_ & 0x00000001) == 0x00000001)) {
        regstDescId_ = java.util.Collections.unmodifiableList(regstDescId_);
        bitField0_ = (bitField0_ & ~0x00000001);
      }
      result.regstDescId_ = regstDescId_;
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
      if (other instanceof org.oneflow.core.job.RegstDescIdSet) {
        return mergeFrom((org.oneflow.core.job.RegstDescIdSet)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.RegstDescIdSet other) {
      if (other == org.oneflow.core.job.RegstDescIdSet.getDefaultInstance()) return this;
      if (!other.regstDescId_.isEmpty()) {
        if (regstDescId_.isEmpty()) {
          regstDescId_ = other.regstDescId_;
          bitField0_ = (bitField0_ & ~0x00000001);
        } else {
          ensureRegstDescIdIsMutable();
          regstDescId_.addAll(other.regstDescId_);
        }
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
      org.oneflow.core.job.RegstDescIdSet parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.RegstDescIdSet) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private java.util.List<java.lang.Long> regstDescId_ = java.util.Collections.emptyList();
    private void ensureRegstDescIdIsMutable() {
      if (!((bitField0_ & 0x00000001) == 0x00000001)) {
        regstDescId_ = new java.util.ArrayList<java.lang.Long>(regstDescId_);
        bitField0_ |= 0x00000001;
       }
    }
    /**
     * <code>repeated int64 regst_desc_id = 1;</code>
     */
    public java.util.List<java.lang.Long>
        getRegstDescIdList() {
      return java.util.Collections.unmodifiableList(regstDescId_);
    }
    /**
     * <code>repeated int64 regst_desc_id = 1;</code>
     */
    public int getRegstDescIdCount() {
      return regstDescId_.size();
    }
    /**
     * <code>repeated int64 regst_desc_id = 1;</code>
     */
    public long getRegstDescId(int index) {
      return regstDescId_.get(index);
    }
    /**
     * <code>repeated int64 regst_desc_id = 1;</code>
     */
    public Builder setRegstDescId(
        int index, long value) {
      ensureRegstDescIdIsMutable();
      regstDescId_.set(index, value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated int64 regst_desc_id = 1;</code>
     */
    public Builder addRegstDescId(long value) {
      ensureRegstDescIdIsMutable();
      regstDescId_.add(value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated int64 regst_desc_id = 1;</code>
     */
    public Builder addAllRegstDescId(
        java.lang.Iterable<? extends java.lang.Long> values) {
      ensureRegstDescIdIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, regstDescId_);
      onChanged();
      return this;
    }
    /**
     * <code>repeated int64 regst_desc_id = 1;</code>
     */
    public Builder clearRegstDescId() {
      regstDescId_ = java.util.Collections.emptyList();
      bitField0_ = (bitField0_ & ~0x00000001);
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


    // @@protoc_insertion_point(builder_scope:oneflow.RegstDescIdSet)
  }

  // @@protoc_insertion_point(class_scope:oneflow.RegstDescIdSet)
  private static final org.oneflow.core.job.RegstDescIdSet DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.RegstDescIdSet();
  }

  public static org.oneflow.core.job.RegstDescIdSet getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<RegstDescIdSet>
      PARSER = new com.google.protobuf.AbstractParser<RegstDescIdSet>() {
    public RegstDescIdSet parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new RegstDescIdSet(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<RegstDescIdSet> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<RegstDescIdSet> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.RegstDescIdSet getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
