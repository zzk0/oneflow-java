// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/register/logical_blob_id.proto

package org.oneflow.core.register;

/**
 * Protobuf type {@code oneflow.LogicalBlobIdPairs}
 */
public  final class LogicalBlobIdPairs extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.LogicalBlobIdPairs)
    LogicalBlobIdPairsOrBuilder {
  // Use LogicalBlobIdPairs.newBuilder() to construct.
  private LogicalBlobIdPairs(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private LogicalBlobIdPairs() {
    pair_ = java.util.Collections.emptyList();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private LogicalBlobIdPairs(
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
            if (!((mutable_bitField0_ & 0x00000001) == 0x00000001)) {
              pair_ = new java.util.ArrayList<org.oneflow.core.register.LogicalBlobIdPair>();
              mutable_bitField0_ |= 0x00000001;
            }
            pair_.add(
                input.readMessage(org.oneflow.core.register.LogicalBlobIdPair.PARSER, extensionRegistry));
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
        pair_ = java.util.Collections.unmodifiableList(pair_);
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.oneflow.core.register.LogicalBlobIdOuterClass.internal_static_oneflow_LogicalBlobIdPairs_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.register.LogicalBlobIdOuterClass.internal_static_oneflow_LogicalBlobIdPairs_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.register.LogicalBlobIdPairs.class, org.oneflow.core.register.LogicalBlobIdPairs.Builder.class);
  }

  public static final int PAIR_FIELD_NUMBER = 1;
  private java.util.List<org.oneflow.core.register.LogicalBlobIdPair> pair_;
  /**
   * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
   */
  public java.util.List<org.oneflow.core.register.LogicalBlobIdPair> getPairList() {
    return pair_;
  }
  /**
   * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
   */
  public java.util.List<? extends org.oneflow.core.register.LogicalBlobIdPairOrBuilder> 
      getPairOrBuilderList() {
    return pair_;
  }
  /**
   * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
   */
  public int getPairCount() {
    return pair_.size();
  }
  /**
   * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
   */
  public org.oneflow.core.register.LogicalBlobIdPair getPair(int index) {
    return pair_.get(index);
  }
  /**
   * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
   */
  public org.oneflow.core.register.LogicalBlobIdPairOrBuilder getPairOrBuilder(
      int index) {
    return pair_.get(index);
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    for (int i = 0; i < getPairCount(); i++) {
      if (!getPair(i).isInitialized()) {
        memoizedIsInitialized = 0;
        return false;
      }
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    for (int i = 0; i < pair_.size(); i++) {
      output.writeMessage(1, pair_.get(i));
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    for (int i = 0; i < pair_.size(); i++) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(1, pair_.get(i));
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
    if (!(obj instanceof org.oneflow.core.register.LogicalBlobIdPairs)) {
      return super.equals(obj);
    }
    org.oneflow.core.register.LogicalBlobIdPairs other = (org.oneflow.core.register.LogicalBlobIdPairs) obj;

    boolean result = true;
    result = result && getPairList()
        .equals(other.getPairList());
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
    if (getPairCount() > 0) {
      hash = (37 * hash) + PAIR_FIELD_NUMBER;
      hash = (53 * hash) + getPairList().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.register.LogicalBlobIdPairs parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.register.LogicalBlobIdPairs parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.register.LogicalBlobIdPairs parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.register.LogicalBlobIdPairs parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.register.LogicalBlobIdPairs parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.LogicalBlobIdPairs parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.register.LogicalBlobIdPairs parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.LogicalBlobIdPairs parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.register.LogicalBlobIdPairs parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.LogicalBlobIdPairs parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.register.LogicalBlobIdPairs prototype) {
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
   * Protobuf type {@code oneflow.LogicalBlobIdPairs}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.LogicalBlobIdPairs)
      org.oneflow.core.register.LogicalBlobIdPairsOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.register.LogicalBlobIdOuterClass.internal_static_oneflow_LogicalBlobIdPairs_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.register.LogicalBlobIdOuterClass.internal_static_oneflow_LogicalBlobIdPairs_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.register.LogicalBlobIdPairs.class, org.oneflow.core.register.LogicalBlobIdPairs.Builder.class);
    }

    // Construct using org.oneflow.core.register.LogicalBlobIdPairs.newBuilder()
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
        getPairFieldBuilder();
      }
    }
    public Builder clear() {
      super.clear();
      if (pairBuilder_ == null) {
        pair_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000001);
      } else {
        pairBuilder_.clear();
      }
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.register.LogicalBlobIdOuterClass.internal_static_oneflow_LogicalBlobIdPairs_descriptor;
    }

    public org.oneflow.core.register.LogicalBlobIdPairs getDefaultInstanceForType() {
      return org.oneflow.core.register.LogicalBlobIdPairs.getDefaultInstance();
    }

    public org.oneflow.core.register.LogicalBlobIdPairs build() {
      org.oneflow.core.register.LogicalBlobIdPairs result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.register.LogicalBlobIdPairs buildPartial() {
      org.oneflow.core.register.LogicalBlobIdPairs result = new org.oneflow.core.register.LogicalBlobIdPairs(this);
      int from_bitField0_ = bitField0_;
      if (pairBuilder_ == null) {
        if (((bitField0_ & 0x00000001) == 0x00000001)) {
          pair_ = java.util.Collections.unmodifiableList(pair_);
          bitField0_ = (bitField0_ & ~0x00000001);
        }
        result.pair_ = pair_;
      } else {
        result.pair_ = pairBuilder_.build();
      }
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
      if (other instanceof org.oneflow.core.register.LogicalBlobIdPairs) {
        return mergeFrom((org.oneflow.core.register.LogicalBlobIdPairs)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.register.LogicalBlobIdPairs other) {
      if (other == org.oneflow.core.register.LogicalBlobIdPairs.getDefaultInstance()) return this;
      if (pairBuilder_ == null) {
        if (!other.pair_.isEmpty()) {
          if (pair_.isEmpty()) {
            pair_ = other.pair_;
            bitField0_ = (bitField0_ & ~0x00000001);
          } else {
            ensurePairIsMutable();
            pair_.addAll(other.pair_);
          }
          onChanged();
        }
      } else {
        if (!other.pair_.isEmpty()) {
          if (pairBuilder_.isEmpty()) {
            pairBuilder_.dispose();
            pairBuilder_ = null;
            pair_ = other.pair_;
            bitField0_ = (bitField0_ & ~0x00000001);
            pairBuilder_ = 
              com.google.protobuf.GeneratedMessageV3.alwaysUseFieldBuilders ?
                 getPairFieldBuilder() : null;
          } else {
            pairBuilder_.addAllMessages(other.pair_);
          }
        }
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      for (int i = 0; i < getPairCount(); i++) {
        if (!getPair(i).isInitialized()) {
          return false;
        }
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.register.LogicalBlobIdPairs parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.register.LogicalBlobIdPairs) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private java.util.List<org.oneflow.core.register.LogicalBlobIdPair> pair_ =
      java.util.Collections.emptyList();
    private void ensurePairIsMutable() {
      if (!((bitField0_ & 0x00000001) == 0x00000001)) {
        pair_ = new java.util.ArrayList<org.oneflow.core.register.LogicalBlobIdPair>(pair_);
        bitField0_ |= 0x00000001;
       }
    }

    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.oneflow.core.register.LogicalBlobIdPair, org.oneflow.core.register.LogicalBlobIdPair.Builder, org.oneflow.core.register.LogicalBlobIdPairOrBuilder> pairBuilder_;

    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public java.util.List<org.oneflow.core.register.LogicalBlobIdPair> getPairList() {
      if (pairBuilder_ == null) {
        return java.util.Collections.unmodifiableList(pair_);
      } else {
        return pairBuilder_.getMessageList();
      }
    }
    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public int getPairCount() {
      if (pairBuilder_ == null) {
        return pair_.size();
      } else {
        return pairBuilder_.getCount();
      }
    }
    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public org.oneflow.core.register.LogicalBlobIdPair getPair(int index) {
      if (pairBuilder_ == null) {
        return pair_.get(index);
      } else {
        return pairBuilder_.getMessage(index);
      }
    }
    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public Builder setPair(
        int index, org.oneflow.core.register.LogicalBlobIdPair value) {
      if (pairBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensurePairIsMutable();
        pair_.set(index, value);
        onChanged();
      } else {
        pairBuilder_.setMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public Builder setPair(
        int index, org.oneflow.core.register.LogicalBlobIdPair.Builder builderForValue) {
      if (pairBuilder_ == null) {
        ensurePairIsMutable();
        pair_.set(index, builderForValue.build());
        onChanged();
      } else {
        pairBuilder_.setMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public Builder addPair(org.oneflow.core.register.LogicalBlobIdPair value) {
      if (pairBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensurePairIsMutable();
        pair_.add(value);
        onChanged();
      } else {
        pairBuilder_.addMessage(value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public Builder addPair(
        int index, org.oneflow.core.register.LogicalBlobIdPair value) {
      if (pairBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensurePairIsMutable();
        pair_.add(index, value);
        onChanged();
      } else {
        pairBuilder_.addMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public Builder addPair(
        org.oneflow.core.register.LogicalBlobIdPair.Builder builderForValue) {
      if (pairBuilder_ == null) {
        ensurePairIsMutable();
        pair_.add(builderForValue.build());
        onChanged();
      } else {
        pairBuilder_.addMessage(builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public Builder addPair(
        int index, org.oneflow.core.register.LogicalBlobIdPair.Builder builderForValue) {
      if (pairBuilder_ == null) {
        ensurePairIsMutable();
        pair_.add(index, builderForValue.build());
        onChanged();
      } else {
        pairBuilder_.addMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public Builder addAllPair(
        java.lang.Iterable<? extends org.oneflow.core.register.LogicalBlobIdPair> values) {
      if (pairBuilder_ == null) {
        ensurePairIsMutable();
        com.google.protobuf.AbstractMessageLite.Builder.addAll(
            values, pair_);
        onChanged();
      } else {
        pairBuilder_.addAllMessages(values);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public Builder clearPair() {
      if (pairBuilder_ == null) {
        pair_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000001);
        onChanged();
      } else {
        pairBuilder_.clear();
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public Builder removePair(int index) {
      if (pairBuilder_ == null) {
        ensurePairIsMutable();
        pair_.remove(index);
        onChanged();
      } else {
        pairBuilder_.remove(index);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public org.oneflow.core.register.LogicalBlobIdPair.Builder getPairBuilder(
        int index) {
      return getPairFieldBuilder().getBuilder(index);
    }
    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public org.oneflow.core.register.LogicalBlobIdPairOrBuilder getPairOrBuilder(
        int index) {
      if (pairBuilder_ == null) {
        return pair_.get(index);  } else {
        return pairBuilder_.getMessageOrBuilder(index);
      }
    }
    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public java.util.List<? extends org.oneflow.core.register.LogicalBlobIdPairOrBuilder> 
         getPairOrBuilderList() {
      if (pairBuilder_ != null) {
        return pairBuilder_.getMessageOrBuilderList();
      } else {
        return java.util.Collections.unmodifiableList(pair_);
      }
    }
    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public org.oneflow.core.register.LogicalBlobIdPair.Builder addPairBuilder() {
      return getPairFieldBuilder().addBuilder(
          org.oneflow.core.register.LogicalBlobIdPair.getDefaultInstance());
    }
    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public org.oneflow.core.register.LogicalBlobIdPair.Builder addPairBuilder(
        int index) {
      return getPairFieldBuilder().addBuilder(
          index, org.oneflow.core.register.LogicalBlobIdPair.getDefaultInstance());
    }
    /**
     * <code>repeated .oneflow.LogicalBlobIdPair pair = 1;</code>
     */
    public java.util.List<org.oneflow.core.register.LogicalBlobIdPair.Builder> 
         getPairBuilderList() {
      return getPairFieldBuilder().getBuilderList();
    }
    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.oneflow.core.register.LogicalBlobIdPair, org.oneflow.core.register.LogicalBlobIdPair.Builder, org.oneflow.core.register.LogicalBlobIdPairOrBuilder> 
        getPairFieldBuilder() {
      if (pairBuilder_ == null) {
        pairBuilder_ = new com.google.protobuf.RepeatedFieldBuilderV3<
            org.oneflow.core.register.LogicalBlobIdPair, org.oneflow.core.register.LogicalBlobIdPair.Builder, org.oneflow.core.register.LogicalBlobIdPairOrBuilder>(
                pair_,
                ((bitField0_ & 0x00000001) == 0x00000001),
                getParentForChildren(),
                isClean());
        pair_ = null;
      }
      return pairBuilder_;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:oneflow.LogicalBlobIdPairs)
  }

  // @@protoc_insertion_point(class_scope:oneflow.LogicalBlobIdPairs)
  private static final org.oneflow.core.register.LogicalBlobIdPairs DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.register.LogicalBlobIdPairs();
  }

  public static org.oneflow.core.register.LogicalBlobIdPairs getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<LogicalBlobIdPairs>
      PARSER = new com.google.protobuf.AbstractParser<LogicalBlobIdPairs>() {
    public LogicalBlobIdPairs parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new LogicalBlobIdPairs(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<LogicalBlobIdPairs> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<LogicalBlobIdPairs> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.register.LogicalBlobIdPairs getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

