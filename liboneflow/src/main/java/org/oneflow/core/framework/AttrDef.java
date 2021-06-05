// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/framework/user_op_attr.proto

package org.oneflow.core.framework;

/**
 * Protobuf type {@code oneflow.AttrDef}
 */
public  final class AttrDef extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.AttrDef)
    AttrDefOrBuilder {
  // Use AttrDef.newBuilder() to construct.
  private AttrDef(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private AttrDef() {
    name_ = "";
    description_ = "";
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private AttrDef(
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
            name_ = bs;
            break;
          }
          case 18: {
            com.google.protobuf.ByteString bs = input.readBytes();
            bitField0_ |= 0x00000002;
            description_ = bs;
            break;
          }
          case 26: {
            org.oneflow.core.framework.AttrValue.Builder subBuilder = null;
            if (((bitField0_ & 0x00000004) == 0x00000004)) {
              subBuilder = defaultVal_.toBuilder();
            }
            defaultVal_ = input.readMessage(org.oneflow.core.framework.AttrValue.PARSER, extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(defaultVal_);
              defaultVal_ = subBuilder.buildPartial();
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
    return org.oneflow.core.framework.UserOpAttr.internal_static_oneflow_AttrDef_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.framework.UserOpAttr.internal_static_oneflow_AttrDef_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.framework.AttrDef.class, org.oneflow.core.framework.AttrDef.Builder.class);
  }

  private int bitField0_;
  public static final int NAME_FIELD_NUMBER = 1;
  private volatile java.lang.Object name_;
  /**
   * <code>required string name = 1;</code>
   */
  public boolean hasName() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>required string name = 1;</code>
   */
  public java.lang.String getName() {
    java.lang.Object ref = name_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      if (bs.isValidUtf8()) {
        name_ = s;
      }
      return s;
    }
  }
  /**
   * <code>required string name = 1;</code>
   */
  public com.google.protobuf.ByteString
      getNameBytes() {
    java.lang.Object ref = name_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      name_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int DESCRIPTION_FIELD_NUMBER = 2;
  private volatile java.lang.Object description_;
  /**
   * <code>required string description = 2;</code>
   */
  public boolean hasDescription() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>required string description = 2;</code>
   */
  public java.lang.String getDescription() {
    java.lang.Object ref = description_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      if (bs.isValidUtf8()) {
        description_ = s;
      }
      return s;
    }
  }
  /**
   * <code>required string description = 2;</code>
   */
  public com.google.protobuf.ByteString
      getDescriptionBytes() {
    java.lang.Object ref = description_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      description_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int DEFAULT_VAL_FIELD_NUMBER = 3;
  private org.oneflow.core.framework.AttrValue defaultVal_;
  /**
   * <code>required .oneflow.AttrValue default_val = 3;</code>
   */
  public boolean hasDefaultVal() {
    return ((bitField0_ & 0x00000004) == 0x00000004);
  }
  /**
   * <code>required .oneflow.AttrValue default_val = 3;</code>
   */
  public org.oneflow.core.framework.AttrValue getDefaultVal() {
    return defaultVal_ == null ? org.oneflow.core.framework.AttrValue.getDefaultInstance() : defaultVal_;
  }
  /**
   * <code>required .oneflow.AttrValue default_val = 3;</code>
   */
  public org.oneflow.core.framework.AttrValueOrBuilder getDefaultValOrBuilder() {
    return defaultVal_ == null ? org.oneflow.core.framework.AttrValue.getDefaultInstance() : defaultVal_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (!hasName()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!hasDescription()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!hasDefaultVal()) {
      memoizedIsInitialized = 0;
      return false;
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, name_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 2, description_);
    }
    if (((bitField0_ & 0x00000004) == 0x00000004)) {
      output.writeMessage(3, getDefaultVal());
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(1, name_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(2, description_);
    }
    if (((bitField0_ & 0x00000004) == 0x00000004)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(3, getDefaultVal());
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
    if (!(obj instanceof org.oneflow.core.framework.AttrDef)) {
      return super.equals(obj);
    }
    org.oneflow.core.framework.AttrDef other = (org.oneflow.core.framework.AttrDef) obj;

    boolean result = true;
    result = result && (hasName() == other.hasName());
    if (hasName()) {
      result = result && getName()
          .equals(other.getName());
    }
    result = result && (hasDescription() == other.hasDescription());
    if (hasDescription()) {
      result = result && getDescription()
          .equals(other.getDescription());
    }
    result = result && (hasDefaultVal() == other.hasDefaultVal());
    if (hasDefaultVal()) {
      result = result && getDefaultVal()
          .equals(other.getDefaultVal());
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
    if (hasName()) {
      hash = (37 * hash) + NAME_FIELD_NUMBER;
      hash = (53 * hash) + getName().hashCode();
    }
    if (hasDescription()) {
      hash = (37 * hash) + DESCRIPTION_FIELD_NUMBER;
      hash = (53 * hash) + getDescription().hashCode();
    }
    if (hasDefaultVal()) {
      hash = (37 * hash) + DEFAULT_VAL_FIELD_NUMBER;
      hash = (53 * hash) + getDefaultVal().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.framework.AttrDef parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.framework.AttrDef parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.framework.AttrDef parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.framework.AttrDef parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.framework.AttrDef parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.framework.AttrDef parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.framework.AttrDef parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.framework.AttrDef parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.framework.AttrDef parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.framework.AttrDef parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.framework.AttrDef prototype) {
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
   * Protobuf type {@code oneflow.AttrDef}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.AttrDef)
      org.oneflow.core.framework.AttrDefOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.framework.UserOpAttr.internal_static_oneflow_AttrDef_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.framework.UserOpAttr.internal_static_oneflow_AttrDef_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.framework.AttrDef.class, org.oneflow.core.framework.AttrDef.Builder.class);
    }

    // Construct using org.oneflow.core.framework.AttrDef.newBuilder()
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
        getDefaultValFieldBuilder();
      }
    }
    public Builder clear() {
      super.clear();
      name_ = "";
      bitField0_ = (bitField0_ & ~0x00000001);
      description_ = "";
      bitField0_ = (bitField0_ & ~0x00000002);
      if (defaultValBuilder_ == null) {
        defaultVal_ = null;
      } else {
        defaultValBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000004);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.framework.UserOpAttr.internal_static_oneflow_AttrDef_descriptor;
    }

    public org.oneflow.core.framework.AttrDef getDefaultInstanceForType() {
      return org.oneflow.core.framework.AttrDef.getDefaultInstance();
    }

    public org.oneflow.core.framework.AttrDef build() {
      org.oneflow.core.framework.AttrDef result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.framework.AttrDef buildPartial() {
      org.oneflow.core.framework.AttrDef result = new org.oneflow.core.framework.AttrDef(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      result.name_ = name_;
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000002;
      }
      result.description_ = description_;
      if (((from_bitField0_ & 0x00000004) == 0x00000004)) {
        to_bitField0_ |= 0x00000004;
      }
      if (defaultValBuilder_ == null) {
        result.defaultVal_ = defaultVal_;
      } else {
        result.defaultVal_ = defaultValBuilder_.build();
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
      if (other instanceof org.oneflow.core.framework.AttrDef) {
        return mergeFrom((org.oneflow.core.framework.AttrDef)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.framework.AttrDef other) {
      if (other == org.oneflow.core.framework.AttrDef.getDefaultInstance()) return this;
      if (other.hasName()) {
        bitField0_ |= 0x00000001;
        name_ = other.name_;
        onChanged();
      }
      if (other.hasDescription()) {
        bitField0_ |= 0x00000002;
        description_ = other.description_;
        onChanged();
      }
      if (other.hasDefaultVal()) {
        mergeDefaultVal(other.getDefaultVal());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (!hasName()) {
        return false;
      }
      if (!hasDescription()) {
        return false;
      }
      if (!hasDefaultVal()) {
        return false;
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.framework.AttrDef parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.framework.AttrDef) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private java.lang.Object name_ = "";
    /**
     * <code>required string name = 1;</code>
     */
    public boolean hasName() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>required string name = 1;</code>
     */
    public java.lang.String getName() {
      java.lang.Object ref = name_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        if (bs.isValidUtf8()) {
          name_ = s;
        }
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>required string name = 1;</code>
     */
    public com.google.protobuf.ByteString
        getNameBytes() {
      java.lang.Object ref = name_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        name_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>required string name = 1;</code>
     */
    public Builder setName(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000001;
      name_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>required string name = 1;</code>
     */
    public Builder clearName() {
      bitField0_ = (bitField0_ & ~0x00000001);
      name_ = getDefaultInstance().getName();
      onChanged();
      return this;
    }
    /**
     * <code>required string name = 1;</code>
     */
    public Builder setNameBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000001;
      name_ = value;
      onChanged();
      return this;
    }

    private java.lang.Object description_ = "";
    /**
     * <code>required string description = 2;</code>
     */
    public boolean hasDescription() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>required string description = 2;</code>
     */
    public java.lang.String getDescription() {
      java.lang.Object ref = description_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        if (bs.isValidUtf8()) {
          description_ = s;
        }
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>required string description = 2;</code>
     */
    public com.google.protobuf.ByteString
        getDescriptionBytes() {
      java.lang.Object ref = description_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        description_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>required string description = 2;</code>
     */
    public Builder setDescription(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000002;
      description_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>required string description = 2;</code>
     */
    public Builder clearDescription() {
      bitField0_ = (bitField0_ & ~0x00000002);
      description_ = getDefaultInstance().getDescription();
      onChanged();
      return this;
    }
    /**
     * <code>required string description = 2;</code>
     */
    public Builder setDescriptionBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000002;
      description_ = value;
      onChanged();
      return this;
    }

    private org.oneflow.core.framework.AttrValue defaultVal_ = null;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.framework.AttrValue, org.oneflow.core.framework.AttrValue.Builder, org.oneflow.core.framework.AttrValueOrBuilder> defaultValBuilder_;
    /**
     * <code>required .oneflow.AttrValue default_val = 3;</code>
     */
    public boolean hasDefaultVal() {
      return ((bitField0_ & 0x00000004) == 0x00000004);
    }
    /**
     * <code>required .oneflow.AttrValue default_val = 3;</code>
     */
    public org.oneflow.core.framework.AttrValue getDefaultVal() {
      if (defaultValBuilder_ == null) {
        return defaultVal_ == null ? org.oneflow.core.framework.AttrValue.getDefaultInstance() : defaultVal_;
      } else {
        return defaultValBuilder_.getMessage();
      }
    }
    /**
     * <code>required .oneflow.AttrValue default_val = 3;</code>
     */
    public Builder setDefaultVal(org.oneflow.core.framework.AttrValue value) {
      if (defaultValBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        defaultVal_ = value;
        onChanged();
      } else {
        defaultValBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000004;
      return this;
    }
    /**
     * <code>required .oneflow.AttrValue default_val = 3;</code>
     */
    public Builder setDefaultVal(
        org.oneflow.core.framework.AttrValue.Builder builderForValue) {
      if (defaultValBuilder_ == null) {
        defaultVal_ = builderForValue.build();
        onChanged();
      } else {
        defaultValBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000004;
      return this;
    }
    /**
     * <code>required .oneflow.AttrValue default_val = 3;</code>
     */
    public Builder mergeDefaultVal(org.oneflow.core.framework.AttrValue value) {
      if (defaultValBuilder_ == null) {
        if (((bitField0_ & 0x00000004) == 0x00000004) &&
            defaultVal_ != null &&
            defaultVal_ != org.oneflow.core.framework.AttrValue.getDefaultInstance()) {
          defaultVal_ =
            org.oneflow.core.framework.AttrValue.newBuilder(defaultVal_).mergeFrom(value).buildPartial();
        } else {
          defaultVal_ = value;
        }
        onChanged();
      } else {
        defaultValBuilder_.mergeFrom(value);
      }
      bitField0_ |= 0x00000004;
      return this;
    }
    /**
     * <code>required .oneflow.AttrValue default_val = 3;</code>
     */
    public Builder clearDefaultVal() {
      if (defaultValBuilder_ == null) {
        defaultVal_ = null;
        onChanged();
      } else {
        defaultValBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000004);
      return this;
    }
    /**
     * <code>required .oneflow.AttrValue default_val = 3;</code>
     */
    public org.oneflow.core.framework.AttrValue.Builder getDefaultValBuilder() {
      bitField0_ |= 0x00000004;
      onChanged();
      return getDefaultValFieldBuilder().getBuilder();
    }
    /**
     * <code>required .oneflow.AttrValue default_val = 3;</code>
     */
    public org.oneflow.core.framework.AttrValueOrBuilder getDefaultValOrBuilder() {
      if (defaultValBuilder_ != null) {
        return defaultValBuilder_.getMessageOrBuilder();
      } else {
        return defaultVal_ == null ?
            org.oneflow.core.framework.AttrValue.getDefaultInstance() : defaultVal_;
      }
    }
    /**
     * <code>required .oneflow.AttrValue default_val = 3;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.framework.AttrValue, org.oneflow.core.framework.AttrValue.Builder, org.oneflow.core.framework.AttrValueOrBuilder> 
        getDefaultValFieldBuilder() {
      if (defaultValBuilder_ == null) {
        defaultValBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.oneflow.core.framework.AttrValue, org.oneflow.core.framework.AttrValue.Builder, org.oneflow.core.framework.AttrValueOrBuilder>(
                getDefaultVal(),
                getParentForChildren(),
                isClean());
        defaultVal_ = null;
      }
      return defaultValBuilder_;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:oneflow.AttrDef)
  }

  // @@protoc_insertion_point(class_scope:oneflow.AttrDef)
  private static final org.oneflow.core.framework.AttrDef DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.framework.AttrDef();
  }

  public static org.oneflow.core.framework.AttrDef getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<AttrDef>
      PARSER = new com.google.protobuf.AbstractParser<AttrDef>() {
    public AttrDef parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new AttrDef(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<AttrDef> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<AttrDef> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.framework.AttrDef getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
