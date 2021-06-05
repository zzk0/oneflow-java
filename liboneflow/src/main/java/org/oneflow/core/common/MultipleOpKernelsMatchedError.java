// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/common/error.proto

package org.oneflow.core.common;

/**
 * Protobuf type {@code oneflow.MultipleOpKernelsMatchedError}
 */
public  final class MultipleOpKernelsMatchedError extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.MultipleOpKernelsMatchedError)
    MultipleOpKernelsMatchedErrorOrBuilder {
  // Use MultipleOpKernelsMatchedError.newBuilder() to construct.
  private MultipleOpKernelsMatchedError(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private MultipleOpKernelsMatchedError() {
    matchedOpKernelsDebugStr_ = com.google.protobuf.LazyStringArrayList.EMPTY;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private MultipleOpKernelsMatchedError(
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
            com.google.protobuf.ByteString bs = input.readBytes();
            if (!((mutable_bitField0_ & 0x00000001) == 0x00000001)) {
              matchedOpKernelsDebugStr_ = new com.google.protobuf.LazyStringArrayList();
              mutable_bitField0_ |= 0x00000001;
            }
            matchedOpKernelsDebugStr_.add(bs);
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
        matchedOpKernelsDebugStr_ = matchedOpKernelsDebugStr_.getUnmodifiableView();
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.oneflow.core.common.Error.internal_static_oneflow_MultipleOpKernelsMatchedError_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.common.Error.internal_static_oneflow_MultipleOpKernelsMatchedError_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.common.MultipleOpKernelsMatchedError.class, org.oneflow.core.common.MultipleOpKernelsMatchedError.Builder.class);
  }

  public static final int MATCHED_OP_KERNELS_DEBUG_STR_FIELD_NUMBER = 1;
  private com.google.protobuf.LazyStringList matchedOpKernelsDebugStr_;
  /**
   * <code>repeated string matched_op_kernels_debug_str = 1;</code>
   */
  public com.google.protobuf.ProtocolStringList
      getMatchedOpKernelsDebugStrList() {
    return matchedOpKernelsDebugStr_;
  }
  /**
   * <code>repeated string matched_op_kernels_debug_str = 1;</code>
   */
  public int getMatchedOpKernelsDebugStrCount() {
    return matchedOpKernelsDebugStr_.size();
  }
  /**
   * <code>repeated string matched_op_kernels_debug_str = 1;</code>
   */
  public java.lang.String getMatchedOpKernelsDebugStr(int index) {
    return matchedOpKernelsDebugStr_.get(index);
  }
  /**
   * <code>repeated string matched_op_kernels_debug_str = 1;</code>
   */
  public com.google.protobuf.ByteString
      getMatchedOpKernelsDebugStrBytes(int index) {
    return matchedOpKernelsDebugStr_.getByteString(index);
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
    for (int i = 0; i < matchedOpKernelsDebugStr_.size(); i++) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, matchedOpKernelsDebugStr_.getRaw(i));
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    {
      int dataSize = 0;
      for (int i = 0; i < matchedOpKernelsDebugStr_.size(); i++) {
        dataSize += computeStringSizeNoTag(matchedOpKernelsDebugStr_.getRaw(i));
      }
      size += dataSize;
      size += 1 * getMatchedOpKernelsDebugStrList().size();
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
    if (!(obj instanceof org.oneflow.core.common.MultipleOpKernelsMatchedError)) {
      return super.equals(obj);
    }
    org.oneflow.core.common.MultipleOpKernelsMatchedError other = (org.oneflow.core.common.MultipleOpKernelsMatchedError) obj;

    boolean result = true;
    result = result && getMatchedOpKernelsDebugStrList()
        .equals(other.getMatchedOpKernelsDebugStrList());
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
    if (getMatchedOpKernelsDebugStrCount() > 0) {
      hash = (37 * hash) + MATCHED_OP_KERNELS_DEBUG_STR_FIELD_NUMBER;
      hash = (53 * hash) + getMatchedOpKernelsDebugStrList().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.common.MultipleOpKernelsMatchedError parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.common.MultipleOpKernelsMatchedError parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.common.MultipleOpKernelsMatchedError parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.common.MultipleOpKernelsMatchedError parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.common.MultipleOpKernelsMatchedError parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.common.MultipleOpKernelsMatchedError parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.common.MultipleOpKernelsMatchedError parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.common.MultipleOpKernelsMatchedError parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.common.MultipleOpKernelsMatchedError parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.common.MultipleOpKernelsMatchedError parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.common.MultipleOpKernelsMatchedError prototype) {
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
   * Protobuf type {@code oneflow.MultipleOpKernelsMatchedError}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.MultipleOpKernelsMatchedError)
      org.oneflow.core.common.MultipleOpKernelsMatchedErrorOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.common.Error.internal_static_oneflow_MultipleOpKernelsMatchedError_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.common.Error.internal_static_oneflow_MultipleOpKernelsMatchedError_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.common.MultipleOpKernelsMatchedError.class, org.oneflow.core.common.MultipleOpKernelsMatchedError.Builder.class);
    }

    // Construct using org.oneflow.core.common.MultipleOpKernelsMatchedError.newBuilder()
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
      matchedOpKernelsDebugStr_ = com.google.protobuf.LazyStringArrayList.EMPTY;
      bitField0_ = (bitField0_ & ~0x00000001);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.common.Error.internal_static_oneflow_MultipleOpKernelsMatchedError_descriptor;
    }

    public org.oneflow.core.common.MultipleOpKernelsMatchedError getDefaultInstanceForType() {
      return org.oneflow.core.common.MultipleOpKernelsMatchedError.getDefaultInstance();
    }

    public org.oneflow.core.common.MultipleOpKernelsMatchedError build() {
      org.oneflow.core.common.MultipleOpKernelsMatchedError result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.common.MultipleOpKernelsMatchedError buildPartial() {
      org.oneflow.core.common.MultipleOpKernelsMatchedError result = new org.oneflow.core.common.MultipleOpKernelsMatchedError(this);
      int from_bitField0_ = bitField0_;
      if (((bitField0_ & 0x00000001) == 0x00000001)) {
        matchedOpKernelsDebugStr_ = matchedOpKernelsDebugStr_.getUnmodifiableView();
        bitField0_ = (bitField0_ & ~0x00000001);
      }
      result.matchedOpKernelsDebugStr_ = matchedOpKernelsDebugStr_;
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
      if (other instanceof org.oneflow.core.common.MultipleOpKernelsMatchedError) {
        return mergeFrom((org.oneflow.core.common.MultipleOpKernelsMatchedError)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.common.MultipleOpKernelsMatchedError other) {
      if (other == org.oneflow.core.common.MultipleOpKernelsMatchedError.getDefaultInstance()) return this;
      if (!other.matchedOpKernelsDebugStr_.isEmpty()) {
        if (matchedOpKernelsDebugStr_.isEmpty()) {
          matchedOpKernelsDebugStr_ = other.matchedOpKernelsDebugStr_;
          bitField0_ = (bitField0_ & ~0x00000001);
        } else {
          ensureMatchedOpKernelsDebugStrIsMutable();
          matchedOpKernelsDebugStr_.addAll(other.matchedOpKernelsDebugStr_);
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
      org.oneflow.core.common.MultipleOpKernelsMatchedError parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.common.MultipleOpKernelsMatchedError) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private com.google.protobuf.LazyStringList matchedOpKernelsDebugStr_ = com.google.protobuf.LazyStringArrayList.EMPTY;
    private void ensureMatchedOpKernelsDebugStrIsMutable() {
      if (!((bitField0_ & 0x00000001) == 0x00000001)) {
        matchedOpKernelsDebugStr_ = new com.google.protobuf.LazyStringArrayList(matchedOpKernelsDebugStr_);
        bitField0_ |= 0x00000001;
       }
    }
    /**
     * <code>repeated string matched_op_kernels_debug_str = 1;</code>
     */
    public com.google.protobuf.ProtocolStringList
        getMatchedOpKernelsDebugStrList() {
      return matchedOpKernelsDebugStr_.getUnmodifiableView();
    }
    /**
     * <code>repeated string matched_op_kernels_debug_str = 1;</code>
     */
    public int getMatchedOpKernelsDebugStrCount() {
      return matchedOpKernelsDebugStr_.size();
    }
    /**
     * <code>repeated string matched_op_kernels_debug_str = 1;</code>
     */
    public java.lang.String getMatchedOpKernelsDebugStr(int index) {
      return matchedOpKernelsDebugStr_.get(index);
    }
    /**
     * <code>repeated string matched_op_kernels_debug_str = 1;</code>
     */
    public com.google.protobuf.ByteString
        getMatchedOpKernelsDebugStrBytes(int index) {
      return matchedOpKernelsDebugStr_.getByteString(index);
    }
    /**
     * <code>repeated string matched_op_kernels_debug_str = 1;</code>
     */
    public Builder setMatchedOpKernelsDebugStr(
        int index, java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureMatchedOpKernelsDebugStrIsMutable();
      matchedOpKernelsDebugStr_.set(index, value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string matched_op_kernels_debug_str = 1;</code>
     */
    public Builder addMatchedOpKernelsDebugStr(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureMatchedOpKernelsDebugStrIsMutable();
      matchedOpKernelsDebugStr_.add(value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string matched_op_kernels_debug_str = 1;</code>
     */
    public Builder addAllMatchedOpKernelsDebugStr(
        java.lang.Iterable<java.lang.String> values) {
      ensureMatchedOpKernelsDebugStrIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, matchedOpKernelsDebugStr_);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string matched_op_kernels_debug_str = 1;</code>
     */
    public Builder clearMatchedOpKernelsDebugStr() {
      matchedOpKernelsDebugStr_ = com.google.protobuf.LazyStringArrayList.EMPTY;
      bitField0_ = (bitField0_ & ~0x00000001);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string matched_op_kernels_debug_str = 1;</code>
     */
    public Builder addMatchedOpKernelsDebugStrBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureMatchedOpKernelsDebugStrIsMutable();
      matchedOpKernelsDebugStr_.add(value);
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


    // @@protoc_insertion_point(builder_scope:oneflow.MultipleOpKernelsMatchedError)
  }

  // @@protoc_insertion_point(class_scope:oneflow.MultipleOpKernelsMatchedError)
  private static final org.oneflow.core.common.MultipleOpKernelsMatchedError DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.common.MultipleOpKernelsMatchedError();
  }

  public static org.oneflow.core.common.MultipleOpKernelsMatchedError getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<MultipleOpKernelsMatchedError>
      PARSER = new com.google.protobuf.AbstractParser<MultipleOpKernelsMatchedError>() {
    public MultipleOpKernelsMatchedError parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new MultipleOpKernelsMatchedError(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<MultipleOpKernelsMatchedError> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<MultipleOpKernelsMatchedError> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.common.MultipleOpKernelsMatchedError getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
