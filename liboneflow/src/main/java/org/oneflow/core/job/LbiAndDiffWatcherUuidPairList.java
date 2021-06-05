// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/lbi_diff_watcher_info.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.LbiAndDiffWatcherUuidPairList}
 */
public  final class LbiAndDiffWatcherUuidPairList extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.LbiAndDiffWatcherUuidPairList)
    LbiAndDiffWatcherUuidPairListOrBuilder {
  // Use LbiAndDiffWatcherUuidPairList.newBuilder() to construct.
  private LbiAndDiffWatcherUuidPairList(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private LbiAndDiffWatcherUuidPairList() {
    lbiAndUuidPair_ = java.util.Collections.emptyList();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private LbiAndDiffWatcherUuidPairList(
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
              lbiAndUuidPair_ = new java.util.ArrayList<org.oneflow.core.job.LbiAndDiffWatcherUuidPair>();
              mutable_bitField0_ |= 0x00000001;
            }
            lbiAndUuidPair_.add(
                input.readMessage(org.oneflow.core.job.LbiAndDiffWatcherUuidPair.PARSER, extensionRegistry));
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
        lbiAndUuidPair_ = java.util.Collections.unmodifiableList(lbiAndUuidPair_);
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.oneflow.core.job.LbiDiffWatcherInfoOuterClass.internal_static_oneflow_LbiAndDiffWatcherUuidPairList_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.LbiDiffWatcherInfoOuterClass.internal_static_oneflow_LbiAndDiffWatcherUuidPairList_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.LbiAndDiffWatcherUuidPairList.class, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList.Builder.class);
  }

  public static final int LBI_AND_UUID_PAIR_FIELD_NUMBER = 1;
  private java.util.List<org.oneflow.core.job.LbiAndDiffWatcherUuidPair> lbiAndUuidPair_;
  /**
   * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
   */
  public java.util.List<org.oneflow.core.job.LbiAndDiffWatcherUuidPair> getLbiAndUuidPairList() {
    return lbiAndUuidPair_;
  }
  /**
   * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
   */
  public java.util.List<? extends org.oneflow.core.job.LbiAndDiffWatcherUuidPairOrBuilder> 
      getLbiAndUuidPairOrBuilderList() {
    return lbiAndUuidPair_;
  }
  /**
   * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
   */
  public int getLbiAndUuidPairCount() {
    return lbiAndUuidPair_.size();
  }
  /**
   * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
   */
  public org.oneflow.core.job.LbiAndDiffWatcherUuidPair getLbiAndUuidPair(int index) {
    return lbiAndUuidPair_.get(index);
  }
  /**
   * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
   */
  public org.oneflow.core.job.LbiAndDiffWatcherUuidPairOrBuilder getLbiAndUuidPairOrBuilder(
      int index) {
    return lbiAndUuidPair_.get(index);
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    for (int i = 0; i < getLbiAndUuidPairCount(); i++) {
      if (!getLbiAndUuidPair(i).isInitialized()) {
        memoizedIsInitialized = 0;
        return false;
      }
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    for (int i = 0; i < lbiAndUuidPair_.size(); i++) {
      output.writeMessage(1, lbiAndUuidPair_.get(i));
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    for (int i = 0; i < lbiAndUuidPair_.size(); i++) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(1, lbiAndUuidPair_.get(i));
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
    if (!(obj instanceof org.oneflow.core.job.LbiAndDiffWatcherUuidPairList)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.LbiAndDiffWatcherUuidPairList other = (org.oneflow.core.job.LbiAndDiffWatcherUuidPairList) obj;

    boolean result = true;
    result = result && getLbiAndUuidPairList()
        .equals(other.getLbiAndUuidPairList());
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
    if (getLbiAndUuidPairCount() > 0) {
      hash = (37 * hash) + LBI_AND_UUID_PAIR_FIELD_NUMBER;
      hash = (53 * hash) + getLbiAndUuidPairList().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.LbiAndDiffWatcherUuidPairList parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.LbiAndDiffWatcherUuidPairList parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.LbiAndDiffWatcherUuidPairList parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.LbiAndDiffWatcherUuidPairList parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.LbiAndDiffWatcherUuidPairList parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.LbiAndDiffWatcherUuidPairList parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.LbiAndDiffWatcherUuidPairList parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.LbiAndDiffWatcherUuidPairList parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.LbiAndDiffWatcherUuidPairList parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.LbiAndDiffWatcherUuidPairList parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.LbiAndDiffWatcherUuidPairList prototype) {
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
   * Protobuf type {@code oneflow.LbiAndDiffWatcherUuidPairList}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.LbiAndDiffWatcherUuidPairList)
      org.oneflow.core.job.LbiAndDiffWatcherUuidPairListOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.LbiDiffWatcherInfoOuterClass.internal_static_oneflow_LbiAndDiffWatcherUuidPairList_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.LbiDiffWatcherInfoOuterClass.internal_static_oneflow_LbiAndDiffWatcherUuidPairList_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.LbiAndDiffWatcherUuidPairList.class, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList.Builder.class);
    }

    // Construct using org.oneflow.core.job.LbiAndDiffWatcherUuidPairList.newBuilder()
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
        getLbiAndUuidPairFieldBuilder();
      }
    }
    public Builder clear() {
      super.clear();
      if (lbiAndUuidPairBuilder_ == null) {
        lbiAndUuidPair_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000001);
      } else {
        lbiAndUuidPairBuilder_.clear();
      }
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.LbiDiffWatcherInfoOuterClass.internal_static_oneflow_LbiAndDiffWatcherUuidPairList_descriptor;
    }

    public org.oneflow.core.job.LbiAndDiffWatcherUuidPairList getDefaultInstanceForType() {
      return org.oneflow.core.job.LbiAndDiffWatcherUuidPairList.getDefaultInstance();
    }

    public org.oneflow.core.job.LbiAndDiffWatcherUuidPairList build() {
      org.oneflow.core.job.LbiAndDiffWatcherUuidPairList result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.LbiAndDiffWatcherUuidPairList buildPartial() {
      org.oneflow.core.job.LbiAndDiffWatcherUuidPairList result = new org.oneflow.core.job.LbiAndDiffWatcherUuidPairList(this);
      int from_bitField0_ = bitField0_;
      if (lbiAndUuidPairBuilder_ == null) {
        if (((bitField0_ & 0x00000001) == 0x00000001)) {
          lbiAndUuidPair_ = java.util.Collections.unmodifiableList(lbiAndUuidPair_);
          bitField0_ = (bitField0_ & ~0x00000001);
        }
        result.lbiAndUuidPair_ = lbiAndUuidPair_;
      } else {
        result.lbiAndUuidPair_ = lbiAndUuidPairBuilder_.build();
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
      if (other instanceof org.oneflow.core.job.LbiAndDiffWatcherUuidPairList) {
        return mergeFrom((org.oneflow.core.job.LbiAndDiffWatcherUuidPairList)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.LbiAndDiffWatcherUuidPairList other) {
      if (other == org.oneflow.core.job.LbiAndDiffWatcherUuidPairList.getDefaultInstance()) return this;
      if (lbiAndUuidPairBuilder_ == null) {
        if (!other.lbiAndUuidPair_.isEmpty()) {
          if (lbiAndUuidPair_.isEmpty()) {
            lbiAndUuidPair_ = other.lbiAndUuidPair_;
            bitField0_ = (bitField0_ & ~0x00000001);
          } else {
            ensureLbiAndUuidPairIsMutable();
            lbiAndUuidPair_.addAll(other.lbiAndUuidPair_);
          }
          onChanged();
        }
      } else {
        if (!other.lbiAndUuidPair_.isEmpty()) {
          if (lbiAndUuidPairBuilder_.isEmpty()) {
            lbiAndUuidPairBuilder_.dispose();
            lbiAndUuidPairBuilder_ = null;
            lbiAndUuidPair_ = other.lbiAndUuidPair_;
            bitField0_ = (bitField0_ & ~0x00000001);
            lbiAndUuidPairBuilder_ = 
              com.google.protobuf.GeneratedMessageV3.alwaysUseFieldBuilders ?
                 getLbiAndUuidPairFieldBuilder() : null;
          } else {
            lbiAndUuidPairBuilder_.addAllMessages(other.lbiAndUuidPair_);
          }
        }
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      for (int i = 0; i < getLbiAndUuidPairCount(); i++) {
        if (!getLbiAndUuidPair(i).isInitialized()) {
          return false;
        }
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.job.LbiAndDiffWatcherUuidPairList parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.LbiAndDiffWatcherUuidPairList) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private java.util.List<org.oneflow.core.job.LbiAndDiffWatcherUuidPair> lbiAndUuidPair_ =
      java.util.Collections.emptyList();
    private void ensureLbiAndUuidPairIsMutable() {
      if (!((bitField0_ & 0x00000001) == 0x00000001)) {
        lbiAndUuidPair_ = new java.util.ArrayList<org.oneflow.core.job.LbiAndDiffWatcherUuidPair>(lbiAndUuidPair_);
        bitField0_ |= 0x00000001;
       }
    }

    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.oneflow.core.job.LbiAndDiffWatcherUuidPair, org.oneflow.core.job.LbiAndDiffWatcherUuidPair.Builder, org.oneflow.core.job.LbiAndDiffWatcherUuidPairOrBuilder> lbiAndUuidPairBuilder_;

    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public java.util.List<org.oneflow.core.job.LbiAndDiffWatcherUuidPair> getLbiAndUuidPairList() {
      if (lbiAndUuidPairBuilder_ == null) {
        return java.util.Collections.unmodifiableList(lbiAndUuidPair_);
      } else {
        return lbiAndUuidPairBuilder_.getMessageList();
      }
    }
    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public int getLbiAndUuidPairCount() {
      if (lbiAndUuidPairBuilder_ == null) {
        return lbiAndUuidPair_.size();
      } else {
        return lbiAndUuidPairBuilder_.getCount();
      }
    }
    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public org.oneflow.core.job.LbiAndDiffWatcherUuidPair getLbiAndUuidPair(int index) {
      if (lbiAndUuidPairBuilder_ == null) {
        return lbiAndUuidPair_.get(index);
      } else {
        return lbiAndUuidPairBuilder_.getMessage(index);
      }
    }
    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public Builder setLbiAndUuidPair(
        int index, org.oneflow.core.job.LbiAndDiffWatcherUuidPair value) {
      if (lbiAndUuidPairBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureLbiAndUuidPairIsMutable();
        lbiAndUuidPair_.set(index, value);
        onChanged();
      } else {
        lbiAndUuidPairBuilder_.setMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public Builder setLbiAndUuidPair(
        int index, org.oneflow.core.job.LbiAndDiffWatcherUuidPair.Builder builderForValue) {
      if (lbiAndUuidPairBuilder_ == null) {
        ensureLbiAndUuidPairIsMutable();
        lbiAndUuidPair_.set(index, builderForValue.build());
        onChanged();
      } else {
        lbiAndUuidPairBuilder_.setMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public Builder addLbiAndUuidPair(org.oneflow.core.job.LbiAndDiffWatcherUuidPair value) {
      if (lbiAndUuidPairBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureLbiAndUuidPairIsMutable();
        lbiAndUuidPair_.add(value);
        onChanged();
      } else {
        lbiAndUuidPairBuilder_.addMessage(value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public Builder addLbiAndUuidPair(
        int index, org.oneflow.core.job.LbiAndDiffWatcherUuidPair value) {
      if (lbiAndUuidPairBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureLbiAndUuidPairIsMutable();
        lbiAndUuidPair_.add(index, value);
        onChanged();
      } else {
        lbiAndUuidPairBuilder_.addMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public Builder addLbiAndUuidPair(
        org.oneflow.core.job.LbiAndDiffWatcherUuidPair.Builder builderForValue) {
      if (lbiAndUuidPairBuilder_ == null) {
        ensureLbiAndUuidPairIsMutable();
        lbiAndUuidPair_.add(builderForValue.build());
        onChanged();
      } else {
        lbiAndUuidPairBuilder_.addMessage(builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public Builder addLbiAndUuidPair(
        int index, org.oneflow.core.job.LbiAndDiffWatcherUuidPair.Builder builderForValue) {
      if (lbiAndUuidPairBuilder_ == null) {
        ensureLbiAndUuidPairIsMutable();
        lbiAndUuidPair_.add(index, builderForValue.build());
        onChanged();
      } else {
        lbiAndUuidPairBuilder_.addMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public Builder addAllLbiAndUuidPair(
        java.lang.Iterable<? extends org.oneflow.core.job.LbiAndDiffWatcherUuidPair> values) {
      if (lbiAndUuidPairBuilder_ == null) {
        ensureLbiAndUuidPairIsMutable();
        com.google.protobuf.AbstractMessageLite.Builder.addAll(
            values, lbiAndUuidPair_);
        onChanged();
      } else {
        lbiAndUuidPairBuilder_.addAllMessages(values);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public Builder clearLbiAndUuidPair() {
      if (lbiAndUuidPairBuilder_ == null) {
        lbiAndUuidPair_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000001);
        onChanged();
      } else {
        lbiAndUuidPairBuilder_.clear();
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public Builder removeLbiAndUuidPair(int index) {
      if (lbiAndUuidPairBuilder_ == null) {
        ensureLbiAndUuidPairIsMutable();
        lbiAndUuidPair_.remove(index);
        onChanged();
      } else {
        lbiAndUuidPairBuilder_.remove(index);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public org.oneflow.core.job.LbiAndDiffWatcherUuidPair.Builder getLbiAndUuidPairBuilder(
        int index) {
      return getLbiAndUuidPairFieldBuilder().getBuilder(index);
    }
    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public org.oneflow.core.job.LbiAndDiffWatcherUuidPairOrBuilder getLbiAndUuidPairOrBuilder(
        int index) {
      if (lbiAndUuidPairBuilder_ == null) {
        return lbiAndUuidPair_.get(index);  } else {
        return lbiAndUuidPairBuilder_.getMessageOrBuilder(index);
      }
    }
    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public java.util.List<? extends org.oneflow.core.job.LbiAndDiffWatcherUuidPairOrBuilder> 
         getLbiAndUuidPairOrBuilderList() {
      if (lbiAndUuidPairBuilder_ != null) {
        return lbiAndUuidPairBuilder_.getMessageOrBuilderList();
      } else {
        return java.util.Collections.unmodifiableList(lbiAndUuidPair_);
      }
    }
    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public org.oneflow.core.job.LbiAndDiffWatcherUuidPair.Builder addLbiAndUuidPairBuilder() {
      return getLbiAndUuidPairFieldBuilder().addBuilder(
          org.oneflow.core.job.LbiAndDiffWatcherUuidPair.getDefaultInstance());
    }
    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public org.oneflow.core.job.LbiAndDiffWatcherUuidPair.Builder addLbiAndUuidPairBuilder(
        int index) {
      return getLbiAndUuidPairFieldBuilder().addBuilder(
          index, org.oneflow.core.job.LbiAndDiffWatcherUuidPair.getDefaultInstance());
    }
    /**
     * <code>repeated .oneflow.LbiAndDiffWatcherUuidPair lbi_and_uuid_pair = 1;</code>
     */
    public java.util.List<org.oneflow.core.job.LbiAndDiffWatcherUuidPair.Builder> 
         getLbiAndUuidPairBuilderList() {
      return getLbiAndUuidPairFieldBuilder().getBuilderList();
    }
    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.oneflow.core.job.LbiAndDiffWatcherUuidPair, org.oneflow.core.job.LbiAndDiffWatcherUuidPair.Builder, org.oneflow.core.job.LbiAndDiffWatcherUuidPairOrBuilder> 
        getLbiAndUuidPairFieldBuilder() {
      if (lbiAndUuidPairBuilder_ == null) {
        lbiAndUuidPairBuilder_ = new com.google.protobuf.RepeatedFieldBuilderV3<
            org.oneflow.core.job.LbiAndDiffWatcherUuidPair, org.oneflow.core.job.LbiAndDiffWatcherUuidPair.Builder, org.oneflow.core.job.LbiAndDiffWatcherUuidPairOrBuilder>(
                lbiAndUuidPair_,
                ((bitField0_ & 0x00000001) == 0x00000001),
                getParentForChildren(),
                isClean());
        lbiAndUuidPair_ = null;
      }
      return lbiAndUuidPairBuilder_;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:oneflow.LbiAndDiffWatcherUuidPairList)
  }

  // @@protoc_insertion_point(class_scope:oneflow.LbiAndDiffWatcherUuidPairList)
  private static final org.oneflow.core.job.LbiAndDiffWatcherUuidPairList DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.LbiAndDiffWatcherUuidPairList();
  }

  public static org.oneflow.core.job.LbiAndDiffWatcherUuidPairList getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<LbiAndDiffWatcherUuidPairList>
      PARSER = new com.google.protobuf.AbstractParser<LbiAndDiffWatcherUuidPairList>() {
    public LbiAndDiffWatcherUuidPairList parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new LbiAndDiffWatcherUuidPairList(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<LbiAndDiffWatcherUuidPairList> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<LbiAndDiffWatcherUuidPairList> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.LbiAndDiffWatcherUuidPairList getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

