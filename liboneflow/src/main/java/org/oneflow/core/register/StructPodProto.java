// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/register/pod.proto

package org.oneflow.core.register;

/**
 * Protobuf type {@code oneflow.StructPodProto}
 */
public  final class StructPodProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.StructPodProto)
    StructPodProtoOrBuilder {
  // Use StructPodProto.newBuilder() to construct.
  private StructPodProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private StructPodProto() {
    field_ = java.util.Collections.emptyList();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private StructPodProto(
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
              field_ = new java.util.ArrayList<org.oneflow.core.register.FieldPodProto>();
              mutable_bitField0_ |= 0x00000001;
            }
            field_.add(
                input.readMessage(org.oneflow.core.register.FieldPodProto.PARSER, extensionRegistry));
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
        field_ = java.util.Collections.unmodifiableList(field_);
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.oneflow.core.register.Pod.internal_static_oneflow_StructPodProto_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.register.Pod.internal_static_oneflow_StructPodProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.register.StructPodProto.class, org.oneflow.core.register.StructPodProto.Builder.class);
  }

  public static final int FIELD_FIELD_NUMBER = 1;
  private java.util.List<org.oneflow.core.register.FieldPodProto> field_;
  /**
   * <code>repeated .oneflow.FieldPodProto field = 1;</code>
   */
  public java.util.List<org.oneflow.core.register.FieldPodProto> getFieldList() {
    return field_;
  }
  /**
   * <code>repeated .oneflow.FieldPodProto field = 1;</code>
   */
  public java.util.List<? extends org.oneflow.core.register.FieldPodProtoOrBuilder> 
      getFieldOrBuilderList() {
    return field_;
  }
  /**
   * <code>repeated .oneflow.FieldPodProto field = 1;</code>
   */
  public int getFieldCount() {
    return field_.size();
  }
  /**
   * <code>repeated .oneflow.FieldPodProto field = 1;</code>
   */
  public org.oneflow.core.register.FieldPodProto getField(int index) {
    return field_.get(index);
  }
  /**
   * <code>repeated .oneflow.FieldPodProto field = 1;</code>
   */
  public org.oneflow.core.register.FieldPodProtoOrBuilder getFieldOrBuilder(
      int index) {
    return field_.get(index);
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    for (int i = 0; i < getFieldCount(); i++) {
      if (!getField(i).isInitialized()) {
        memoizedIsInitialized = 0;
        return false;
      }
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    for (int i = 0; i < field_.size(); i++) {
      output.writeMessage(1, field_.get(i));
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    for (int i = 0; i < field_.size(); i++) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(1, field_.get(i));
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
    if (!(obj instanceof org.oneflow.core.register.StructPodProto)) {
      return super.equals(obj);
    }
    org.oneflow.core.register.StructPodProto other = (org.oneflow.core.register.StructPodProto) obj;

    boolean result = true;
    result = result && getFieldList()
        .equals(other.getFieldList());
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
    if (getFieldCount() > 0) {
      hash = (37 * hash) + FIELD_FIELD_NUMBER;
      hash = (53 * hash) + getFieldList().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.register.StructPodProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.register.StructPodProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.register.StructPodProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.register.StructPodProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.register.StructPodProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.StructPodProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.register.StructPodProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.StructPodProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.register.StructPodProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.StructPodProto parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.register.StructPodProto prototype) {
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
   * Protobuf type {@code oneflow.StructPodProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.StructPodProto)
      org.oneflow.core.register.StructPodProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.register.Pod.internal_static_oneflow_StructPodProto_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.register.Pod.internal_static_oneflow_StructPodProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.register.StructPodProto.class, org.oneflow.core.register.StructPodProto.Builder.class);
    }

    // Construct using org.oneflow.core.register.StructPodProto.newBuilder()
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
        getFieldFieldBuilder();
      }
    }
    public Builder clear() {
      super.clear();
      if (fieldBuilder_ == null) {
        field_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000001);
      } else {
        fieldBuilder_.clear();
      }
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.register.Pod.internal_static_oneflow_StructPodProto_descriptor;
    }

    public org.oneflow.core.register.StructPodProto getDefaultInstanceForType() {
      return org.oneflow.core.register.StructPodProto.getDefaultInstance();
    }

    public org.oneflow.core.register.StructPodProto build() {
      org.oneflow.core.register.StructPodProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.register.StructPodProto buildPartial() {
      org.oneflow.core.register.StructPodProto result = new org.oneflow.core.register.StructPodProto(this);
      int from_bitField0_ = bitField0_;
      if (fieldBuilder_ == null) {
        if (((bitField0_ & 0x00000001) == 0x00000001)) {
          field_ = java.util.Collections.unmodifiableList(field_);
          bitField0_ = (bitField0_ & ~0x00000001);
        }
        result.field_ = field_;
      } else {
        result.field_ = fieldBuilder_.build();
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
      if (other instanceof org.oneflow.core.register.StructPodProto) {
        return mergeFrom((org.oneflow.core.register.StructPodProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.register.StructPodProto other) {
      if (other == org.oneflow.core.register.StructPodProto.getDefaultInstance()) return this;
      if (fieldBuilder_ == null) {
        if (!other.field_.isEmpty()) {
          if (field_.isEmpty()) {
            field_ = other.field_;
            bitField0_ = (bitField0_ & ~0x00000001);
          } else {
            ensureFieldIsMutable();
            field_.addAll(other.field_);
          }
          onChanged();
        }
      } else {
        if (!other.field_.isEmpty()) {
          if (fieldBuilder_.isEmpty()) {
            fieldBuilder_.dispose();
            fieldBuilder_ = null;
            field_ = other.field_;
            bitField0_ = (bitField0_ & ~0x00000001);
            fieldBuilder_ = 
              com.google.protobuf.GeneratedMessageV3.alwaysUseFieldBuilders ?
                 getFieldFieldBuilder() : null;
          } else {
            fieldBuilder_.addAllMessages(other.field_);
          }
        }
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      for (int i = 0; i < getFieldCount(); i++) {
        if (!getField(i).isInitialized()) {
          return false;
        }
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.register.StructPodProto parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.register.StructPodProto) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private java.util.List<org.oneflow.core.register.FieldPodProto> field_ =
      java.util.Collections.emptyList();
    private void ensureFieldIsMutable() {
      if (!((bitField0_ & 0x00000001) == 0x00000001)) {
        field_ = new java.util.ArrayList<org.oneflow.core.register.FieldPodProto>(field_);
        bitField0_ |= 0x00000001;
       }
    }

    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.oneflow.core.register.FieldPodProto, org.oneflow.core.register.FieldPodProto.Builder, org.oneflow.core.register.FieldPodProtoOrBuilder> fieldBuilder_;

    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public java.util.List<org.oneflow.core.register.FieldPodProto> getFieldList() {
      if (fieldBuilder_ == null) {
        return java.util.Collections.unmodifiableList(field_);
      } else {
        return fieldBuilder_.getMessageList();
      }
    }
    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public int getFieldCount() {
      if (fieldBuilder_ == null) {
        return field_.size();
      } else {
        return fieldBuilder_.getCount();
      }
    }
    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public org.oneflow.core.register.FieldPodProto getField(int index) {
      if (fieldBuilder_ == null) {
        return field_.get(index);
      } else {
        return fieldBuilder_.getMessage(index);
      }
    }
    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public Builder setField(
        int index, org.oneflow.core.register.FieldPodProto value) {
      if (fieldBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureFieldIsMutable();
        field_.set(index, value);
        onChanged();
      } else {
        fieldBuilder_.setMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public Builder setField(
        int index, org.oneflow.core.register.FieldPodProto.Builder builderForValue) {
      if (fieldBuilder_ == null) {
        ensureFieldIsMutable();
        field_.set(index, builderForValue.build());
        onChanged();
      } else {
        fieldBuilder_.setMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public Builder addField(org.oneflow.core.register.FieldPodProto value) {
      if (fieldBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureFieldIsMutable();
        field_.add(value);
        onChanged();
      } else {
        fieldBuilder_.addMessage(value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public Builder addField(
        int index, org.oneflow.core.register.FieldPodProto value) {
      if (fieldBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureFieldIsMutable();
        field_.add(index, value);
        onChanged();
      } else {
        fieldBuilder_.addMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public Builder addField(
        org.oneflow.core.register.FieldPodProto.Builder builderForValue) {
      if (fieldBuilder_ == null) {
        ensureFieldIsMutable();
        field_.add(builderForValue.build());
        onChanged();
      } else {
        fieldBuilder_.addMessage(builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public Builder addField(
        int index, org.oneflow.core.register.FieldPodProto.Builder builderForValue) {
      if (fieldBuilder_ == null) {
        ensureFieldIsMutable();
        field_.add(index, builderForValue.build());
        onChanged();
      } else {
        fieldBuilder_.addMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public Builder addAllField(
        java.lang.Iterable<? extends org.oneflow.core.register.FieldPodProto> values) {
      if (fieldBuilder_ == null) {
        ensureFieldIsMutable();
        com.google.protobuf.AbstractMessageLite.Builder.addAll(
            values, field_);
        onChanged();
      } else {
        fieldBuilder_.addAllMessages(values);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public Builder clearField() {
      if (fieldBuilder_ == null) {
        field_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000001);
        onChanged();
      } else {
        fieldBuilder_.clear();
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public Builder removeField(int index) {
      if (fieldBuilder_ == null) {
        ensureFieldIsMutable();
        field_.remove(index);
        onChanged();
      } else {
        fieldBuilder_.remove(index);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public org.oneflow.core.register.FieldPodProto.Builder getFieldBuilder(
        int index) {
      return getFieldFieldBuilder().getBuilder(index);
    }
    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public org.oneflow.core.register.FieldPodProtoOrBuilder getFieldOrBuilder(
        int index) {
      if (fieldBuilder_ == null) {
        return field_.get(index);  } else {
        return fieldBuilder_.getMessageOrBuilder(index);
      }
    }
    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public java.util.List<? extends org.oneflow.core.register.FieldPodProtoOrBuilder> 
         getFieldOrBuilderList() {
      if (fieldBuilder_ != null) {
        return fieldBuilder_.getMessageOrBuilderList();
      } else {
        return java.util.Collections.unmodifiableList(field_);
      }
    }
    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public org.oneflow.core.register.FieldPodProto.Builder addFieldBuilder() {
      return getFieldFieldBuilder().addBuilder(
          org.oneflow.core.register.FieldPodProto.getDefaultInstance());
    }
    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public org.oneflow.core.register.FieldPodProto.Builder addFieldBuilder(
        int index) {
      return getFieldFieldBuilder().addBuilder(
          index, org.oneflow.core.register.FieldPodProto.getDefaultInstance());
    }
    /**
     * <code>repeated .oneflow.FieldPodProto field = 1;</code>
     */
    public java.util.List<org.oneflow.core.register.FieldPodProto.Builder> 
         getFieldBuilderList() {
      return getFieldFieldBuilder().getBuilderList();
    }
    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.oneflow.core.register.FieldPodProto, org.oneflow.core.register.FieldPodProto.Builder, org.oneflow.core.register.FieldPodProtoOrBuilder> 
        getFieldFieldBuilder() {
      if (fieldBuilder_ == null) {
        fieldBuilder_ = new com.google.protobuf.RepeatedFieldBuilderV3<
            org.oneflow.core.register.FieldPodProto, org.oneflow.core.register.FieldPodProto.Builder, org.oneflow.core.register.FieldPodProtoOrBuilder>(
                field_,
                ((bitField0_ & 0x00000001) == 0x00000001),
                getParentForChildren(),
                isClean());
        field_ = null;
      }
      return fieldBuilder_;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:oneflow.StructPodProto)
  }

  // @@protoc_insertion_point(class_scope:oneflow.StructPodProto)
  private static final org.oneflow.core.register.StructPodProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.register.StructPodProto();
  }

  public static org.oneflow.core.register.StructPodProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<StructPodProto>
      PARSER = new com.google.protobuf.AbstractParser<StructPodProto>() {
    public StructPodProto parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new StructPodProto(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<StructPodProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<StructPodProto> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.register.StructPodProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
