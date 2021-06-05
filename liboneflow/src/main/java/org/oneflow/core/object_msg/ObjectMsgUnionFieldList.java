// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/object_msg/object_msg_field_list.proto

package org.oneflow.core.object_msg;

/**
 * Protobuf type {@code oneflow.ObjectMsgUnionFieldList}
 */
public  final class ObjectMsgUnionFieldList extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.ObjectMsgUnionFieldList)
    ObjectMsgUnionFieldListOrBuilder {
  // Use ObjectMsgUnionFieldList.newBuilder() to construct.
  private ObjectMsgUnionFieldList(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private ObjectMsgUnionFieldList() {
    unionName_ = "";
    unionField_ = java.util.Collections.emptyList();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private ObjectMsgUnionFieldList(
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
            bitField0_ |= 0x00000001;
            unionName_ = bs;
            break;
          }
          case 18: {
            if (!((mutable_bitField0_ & 0x00000002) == 0x00000002)) {
              unionField_ = new java.util.ArrayList<org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName>();
              mutable_bitField0_ |= 0x00000002;
            }
            unionField_.add(
                input.readMessage(org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName.PARSER, extensionRegistry));
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
      if (((mutable_bitField0_ & 0x00000002) == 0x00000002)) {
        unionField_ = java.util.Collections.unmodifiableList(unionField_);
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.oneflow.core.object_msg.ObjectMsgFieldListOuterClass.internal_static_oneflow_ObjectMsgUnionFieldList_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.object_msg.ObjectMsgFieldListOuterClass.internal_static_oneflow_ObjectMsgUnionFieldList_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.object_msg.ObjectMsgUnionFieldList.class, org.oneflow.core.object_msg.ObjectMsgUnionFieldList.Builder.class);
  }

  private int bitField0_;
  public static final int UNION_NAME_FIELD_NUMBER = 1;
  private volatile java.lang.Object unionName_;
  /**
   * <code>required string union_name = 1;</code>
   */
  public boolean hasUnionName() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>required string union_name = 1;</code>
   */
  public java.lang.String getUnionName() {
    java.lang.Object ref = unionName_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      if (bs.isValidUtf8()) {
        unionName_ = s;
      }
      return s;
    }
  }
  /**
   * <code>required string union_name = 1;</code>
   */
  public com.google.protobuf.ByteString
      getUnionNameBytes() {
    java.lang.Object ref = unionName_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      unionName_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int UNION_FIELD_FIELD_NUMBER = 2;
  private java.util.List<org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName> unionField_;
  /**
   * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
   */
  public java.util.List<org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName> getUnionFieldList() {
    return unionField_;
  }
  /**
   * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
   */
  public java.util.List<? extends org.oneflow.core.object_msg.ObjectMsgFieldTypeAndNameOrBuilder> 
      getUnionFieldOrBuilderList() {
    return unionField_;
  }
  /**
   * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
   */
  public int getUnionFieldCount() {
    return unionField_.size();
  }
  /**
   * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
   */
  public org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName getUnionField(int index) {
    return unionField_.get(index);
  }
  /**
   * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
   */
  public org.oneflow.core.object_msg.ObjectMsgFieldTypeAndNameOrBuilder getUnionFieldOrBuilder(
      int index) {
    return unionField_.get(index);
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (!hasUnionName()) {
      memoizedIsInitialized = 0;
      return false;
    }
    for (int i = 0; i < getUnionFieldCount(); i++) {
      if (!getUnionField(i).isInitialized()) {
        memoizedIsInitialized = 0;
        return false;
      }
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, unionName_);
    }
    for (int i = 0; i < unionField_.size(); i++) {
      output.writeMessage(2, unionField_.get(i));
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(1, unionName_);
    }
    for (int i = 0; i < unionField_.size(); i++) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(2, unionField_.get(i));
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
    if (!(obj instanceof org.oneflow.core.object_msg.ObjectMsgUnionFieldList)) {
      return super.equals(obj);
    }
    org.oneflow.core.object_msg.ObjectMsgUnionFieldList other = (org.oneflow.core.object_msg.ObjectMsgUnionFieldList) obj;

    boolean result = true;
    result = result && (hasUnionName() == other.hasUnionName());
    if (hasUnionName()) {
      result = result && getUnionName()
          .equals(other.getUnionName());
    }
    result = result && getUnionFieldList()
        .equals(other.getUnionFieldList());
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
    if (hasUnionName()) {
      hash = (37 * hash) + UNION_NAME_FIELD_NUMBER;
      hash = (53 * hash) + getUnionName().hashCode();
    }
    if (getUnionFieldCount() > 0) {
      hash = (37 * hash) + UNION_FIELD_FIELD_NUMBER;
      hash = (53 * hash) + getUnionFieldList().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.object_msg.ObjectMsgUnionFieldList parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.object_msg.ObjectMsgUnionFieldList parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.object_msg.ObjectMsgUnionFieldList parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.object_msg.ObjectMsgUnionFieldList parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.object_msg.ObjectMsgUnionFieldList parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.object_msg.ObjectMsgUnionFieldList parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.object_msg.ObjectMsgUnionFieldList parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.object_msg.ObjectMsgUnionFieldList parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.object_msg.ObjectMsgUnionFieldList parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.object_msg.ObjectMsgUnionFieldList parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.object_msg.ObjectMsgUnionFieldList prototype) {
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
   * Protobuf type {@code oneflow.ObjectMsgUnionFieldList}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.ObjectMsgUnionFieldList)
      org.oneflow.core.object_msg.ObjectMsgUnionFieldListOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.object_msg.ObjectMsgFieldListOuterClass.internal_static_oneflow_ObjectMsgUnionFieldList_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.object_msg.ObjectMsgFieldListOuterClass.internal_static_oneflow_ObjectMsgUnionFieldList_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.object_msg.ObjectMsgUnionFieldList.class, org.oneflow.core.object_msg.ObjectMsgUnionFieldList.Builder.class);
    }

    // Construct using org.oneflow.core.object_msg.ObjectMsgUnionFieldList.newBuilder()
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
        getUnionFieldFieldBuilder();
      }
    }
    public Builder clear() {
      super.clear();
      unionName_ = "";
      bitField0_ = (bitField0_ & ~0x00000001);
      if (unionFieldBuilder_ == null) {
        unionField_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000002);
      } else {
        unionFieldBuilder_.clear();
      }
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.object_msg.ObjectMsgFieldListOuterClass.internal_static_oneflow_ObjectMsgUnionFieldList_descriptor;
    }

    public org.oneflow.core.object_msg.ObjectMsgUnionFieldList getDefaultInstanceForType() {
      return org.oneflow.core.object_msg.ObjectMsgUnionFieldList.getDefaultInstance();
    }

    public org.oneflow.core.object_msg.ObjectMsgUnionFieldList build() {
      org.oneflow.core.object_msg.ObjectMsgUnionFieldList result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.object_msg.ObjectMsgUnionFieldList buildPartial() {
      org.oneflow.core.object_msg.ObjectMsgUnionFieldList result = new org.oneflow.core.object_msg.ObjectMsgUnionFieldList(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      result.unionName_ = unionName_;
      if (unionFieldBuilder_ == null) {
        if (((bitField0_ & 0x00000002) == 0x00000002)) {
          unionField_ = java.util.Collections.unmodifiableList(unionField_);
          bitField0_ = (bitField0_ & ~0x00000002);
        }
        result.unionField_ = unionField_;
      } else {
        result.unionField_ = unionFieldBuilder_.build();
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
      if (other instanceof org.oneflow.core.object_msg.ObjectMsgUnionFieldList) {
        return mergeFrom((org.oneflow.core.object_msg.ObjectMsgUnionFieldList)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.object_msg.ObjectMsgUnionFieldList other) {
      if (other == org.oneflow.core.object_msg.ObjectMsgUnionFieldList.getDefaultInstance()) return this;
      if (other.hasUnionName()) {
        bitField0_ |= 0x00000001;
        unionName_ = other.unionName_;
        onChanged();
      }
      if (unionFieldBuilder_ == null) {
        if (!other.unionField_.isEmpty()) {
          if (unionField_.isEmpty()) {
            unionField_ = other.unionField_;
            bitField0_ = (bitField0_ & ~0x00000002);
          } else {
            ensureUnionFieldIsMutable();
            unionField_.addAll(other.unionField_);
          }
          onChanged();
        }
      } else {
        if (!other.unionField_.isEmpty()) {
          if (unionFieldBuilder_.isEmpty()) {
            unionFieldBuilder_.dispose();
            unionFieldBuilder_ = null;
            unionField_ = other.unionField_;
            bitField0_ = (bitField0_ & ~0x00000002);
            unionFieldBuilder_ = 
              com.google.protobuf.GeneratedMessageV3.alwaysUseFieldBuilders ?
                 getUnionFieldFieldBuilder() : null;
          } else {
            unionFieldBuilder_.addAllMessages(other.unionField_);
          }
        }
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (!hasUnionName()) {
        return false;
      }
      for (int i = 0; i < getUnionFieldCount(); i++) {
        if (!getUnionField(i).isInitialized()) {
          return false;
        }
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.object_msg.ObjectMsgUnionFieldList parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.object_msg.ObjectMsgUnionFieldList) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private java.lang.Object unionName_ = "";
    /**
     * <code>required string union_name = 1;</code>
     */
    public boolean hasUnionName() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>required string union_name = 1;</code>
     */
    public java.lang.String getUnionName() {
      java.lang.Object ref = unionName_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        if (bs.isValidUtf8()) {
          unionName_ = s;
        }
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>required string union_name = 1;</code>
     */
    public com.google.protobuf.ByteString
        getUnionNameBytes() {
      java.lang.Object ref = unionName_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        unionName_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>required string union_name = 1;</code>
     */
    public Builder setUnionName(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000001;
      unionName_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>required string union_name = 1;</code>
     */
    public Builder clearUnionName() {
      bitField0_ = (bitField0_ & ~0x00000001);
      unionName_ = getDefaultInstance().getUnionName();
      onChanged();
      return this;
    }
    /**
     * <code>required string union_name = 1;</code>
     */
    public Builder setUnionNameBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000001;
      unionName_ = value;
      onChanged();
      return this;
    }

    private java.util.List<org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName> unionField_ =
      java.util.Collections.emptyList();
    private void ensureUnionFieldIsMutable() {
      if (!((bitField0_ & 0x00000002) == 0x00000002)) {
        unionField_ = new java.util.ArrayList<org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName>(unionField_);
        bitField0_ |= 0x00000002;
       }
    }

    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName, org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName.Builder, org.oneflow.core.object_msg.ObjectMsgFieldTypeAndNameOrBuilder> unionFieldBuilder_;

    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public java.util.List<org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName> getUnionFieldList() {
      if (unionFieldBuilder_ == null) {
        return java.util.Collections.unmodifiableList(unionField_);
      } else {
        return unionFieldBuilder_.getMessageList();
      }
    }
    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public int getUnionFieldCount() {
      if (unionFieldBuilder_ == null) {
        return unionField_.size();
      } else {
        return unionFieldBuilder_.getCount();
      }
    }
    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName getUnionField(int index) {
      if (unionFieldBuilder_ == null) {
        return unionField_.get(index);
      } else {
        return unionFieldBuilder_.getMessage(index);
      }
    }
    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public Builder setUnionField(
        int index, org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName value) {
      if (unionFieldBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureUnionFieldIsMutable();
        unionField_.set(index, value);
        onChanged();
      } else {
        unionFieldBuilder_.setMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public Builder setUnionField(
        int index, org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName.Builder builderForValue) {
      if (unionFieldBuilder_ == null) {
        ensureUnionFieldIsMutable();
        unionField_.set(index, builderForValue.build());
        onChanged();
      } else {
        unionFieldBuilder_.setMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public Builder addUnionField(org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName value) {
      if (unionFieldBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureUnionFieldIsMutable();
        unionField_.add(value);
        onChanged();
      } else {
        unionFieldBuilder_.addMessage(value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public Builder addUnionField(
        int index, org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName value) {
      if (unionFieldBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureUnionFieldIsMutable();
        unionField_.add(index, value);
        onChanged();
      } else {
        unionFieldBuilder_.addMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public Builder addUnionField(
        org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName.Builder builderForValue) {
      if (unionFieldBuilder_ == null) {
        ensureUnionFieldIsMutable();
        unionField_.add(builderForValue.build());
        onChanged();
      } else {
        unionFieldBuilder_.addMessage(builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public Builder addUnionField(
        int index, org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName.Builder builderForValue) {
      if (unionFieldBuilder_ == null) {
        ensureUnionFieldIsMutable();
        unionField_.add(index, builderForValue.build());
        onChanged();
      } else {
        unionFieldBuilder_.addMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public Builder addAllUnionField(
        java.lang.Iterable<? extends org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName> values) {
      if (unionFieldBuilder_ == null) {
        ensureUnionFieldIsMutable();
        com.google.protobuf.AbstractMessageLite.Builder.addAll(
            values, unionField_);
        onChanged();
      } else {
        unionFieldBuilder_.addAllMessages(values);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public Builder clearUnionField() {
      if (unionFieldBuilder_ == null) {
        unionField_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000002);
        onChanged();
      } else {
        unionFieldBuilder_.clear();
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public Builder removeUnionField(int index) {
      if (unionFieldBuilder_ == null) {
        ensureUnionFieldIsMutable();
        unionField_.remove(index);
        onChanged();
      } else {
        unionFieldBuilder_.remove(index);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName.Builder getUnionFieldBuilder(
        int index) {
      return getUnionFieldFieldBuilder().getBuilder(index);
    }
    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public org.oneflow.core.object_msg.ObjectMsgFieldTypeAndNameOrBuilder getUnionFieldOrBuilder(
        int index) {
      if (unionFieldBuilder_ == null) {
        return unionField_.get(index);  } else {
        return unionFieldBuilder_.getMessageOrBuilder(index);
      }
    }
    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public java.util.List<? extends org.oneflow.core.object_msg.ObjectMsgFieldTypeAndNameOrBuilder> 
         getUnionFieldOrBuilderList() {
      if (unionFieldBuilder_ != null) {
        return unionFieldBuilder_.getMessageOrBuilderList();
      } else {
        return java.util.Collections.unmodifiableList(unionField_);
      }
    }
    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName.Builder addUnionFieldBuilder() {
      return getUnionFieldFieldBuilder().addBuilder(
          org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName.getDefaultInstance());
    }
    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName.Builder addUnionFieldBuilder(
        int index) {
      return getUnionFieldFieldBuilder().addBuilder(
          index, org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName.getDefaultInstance());
    }
    /**
     * <code>repeated .oneflow.ObjectMsgFieldTypeAndName union_field = 2;</code>
     */
    public java.util.List<org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName.Builder> 
         getUnionFieldBuilderList() {
      return getUnionFieldFieldBuilder().getBuilderList();
    }
    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName, org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName.Builder, org.oneflow.core.object_msg.ObjectMsgFieldTypeAndNameOrBuilder> 
        getUnionFieldFieldBuilder() {
      if (unionFieldBuilder_ == null) {
        unionFieldBuilder_ = new com.google.protobuf.RepeatedFieldBuilderV3<
            org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName, org.oneflow.core.object_msg.ObjectMsgFieldTypeAndName.Builder, org.oneflow.core.object_msg.ObjectMsgFieldTypeAndNameOrBuilder>(
                unionField_,
                ((bitField0_ & 0x00000002) == 0x00000002),
                getParentForChildren(),
                isClean());
        unionField_ = null;
      }
      return unionFieldBuilder_;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:oneflow.ObjectMsgUnionFieldList)
  }

  // @@protoc_insertion_point(class_scope:oneflow.ObjectMsgUnionFieldList)
  private static final org.oneflow.core.object_msg.ObjectMsgUnionFieldList DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.object_msg.ObjectMsgUnionFieldList();
  }

  public static org.oneflow.core.object_msg.ObjectMsgUnionFieldList getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<ObjectMsgUnionFieldList>
      PARSER = new com.google.protobuf.AbstractParser<ObjectMsgUnionFieldList>() {
    public ObjectMsgUnionFieldList parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new ObjectMsgUnionFieldList(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<ObjectMsgUnionFieldList> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<ObjectMsgUnionFieldList> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.object_msg.ObjectMsgUnionFieldList getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

