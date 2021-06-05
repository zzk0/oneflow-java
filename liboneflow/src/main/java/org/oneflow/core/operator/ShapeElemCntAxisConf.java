// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/operator/op_conf.proto

package org.oneflow.core.operator;

/**
 * Protobuf type {@code oneflow.ShapeElemCntAxisConf}
 */
public  final class ShapeElemCntAxisConf extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.ShapeElemCntAxisConf)
    ShapeElemCntAxisConfOrBuilder {
  // Use ShapeElemCntAxisConf.newBuilder() to construct.
  private ShapeElemCntAxisConf(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private ShapeElemCntAxisConf() {
    axis_ = java.util.Collections.emptyList();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private ShapeElemCntAxisConf(
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
              axis_ = new java.util.ArrayList<java.lang.Integer>();
              mutable_bitField0_ |= 0x00000001;
            }
            axis_.add(input.readInt32());
            break;
          }
          case 10: {
            int length = input.readRawVarint32();
            int limit = input.pushLimit(length);
            if (!((mutable_bitField0_ & 0x00000001) == 0x00000001) && input.getBytesUntilLimit() > 0) {
              axis_ = new java.util.ArrayList<java.lang.Integer>();
              mutable_bitField0_ |= 0x00000001;
            }
            while (input.getBytesUntilLimit() > 0) {
              axis_.add(input.readInt32());
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
        axis_ = java.util.Collections.unmodifiableList(axis_);
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.oneflow.core.operator.OpConf.internal_static_oneflow_ShapeElemCntAxisConf_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.operator.OpConf.internal_static_oneflow_ShapeElemCntAxisConf_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.operator.ShapeElemCntAxisConf.class, org.oneflow.core.operator.ShapeElemCntAxisConf.Builder.class);
  }

  public static final int AXIS_FIELD_NUMBER = 1;
  private java.util.List<java.lang.Integer> axis_;
  /**
   * <code>repeated int32 axis = 1;</code>
   */
  public java.util.List<java.lang.Integer>
      getAxisList() {
    return axis_;
  }
  /**
   * <code>repeated int32 axis = 1;</code>
   */
  public int getAxisCount() {
    return axis_.size();
  }
  /**
   * <code>repeated int32 axis = 1;</code>
   */
  public int getAxis(int index) {
    return axis_.get(index);
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
    for (int i = 0; i < axis_.size(); i++) {
      output.writeInt32(1, axis_.get(i));
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    {
      int dataSize = 0;
      for (int i = 0; i < axis_.size(); i++) {
        dataSize += com.google.protobuf.CodedOutputStream
          .computeInt32SizeNoTag(axis_.get(i));
      }
      size += dataSize;
      size += 1 * getAxisList().size();
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
    if (!(obj instanceof org.oneflow.core.operator.ShapeElemCntAxisConf)) {
      return super.equals(obj);
    }
    org.oneflow.core.operator.ShapeElemCntAxisConf other = (org.oneflow.core.operator.ShapeElemCntAxisConf) obj;

    boolean result = true;
    result = result && getAxisList()
        .equals(other.getAxisList());
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
    if (getAxisCount() > 0) {
      hash = (37 * hash) + AXIS_FIELD_NUMBER;
      hash = (53 * hash) + getAxisList().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.operator.ShapeElemCntAxisConf parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.operator.ShapeElemCntAxisConf parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.operator.ShapeElemCntAxisConf parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.operator.ShapeElemCntAxisConf parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.operator.ShapeElemCntAxisConf parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.operator.ShapeElemCntAxisConf parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.operator.ShapeElemCntAxisConf parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.operator.ShapeElemCntAxisConf parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.operator.ShapeElemCntAxisConf parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.operator.ShapeElemCntAxisConf parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.operator.ShapeElemCntAxisConf prototype) {
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
   * Protobuf type {@code oneflow.ShapeElemCntAxisConf}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.ShapeElemCntAxisConf)
      org.oneflow.core.operator.ShapeElemCntAxisConfOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.operator.OpConf.internal_static_oneflow_ShapeElemCntAxisConf_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.operator.OpConf.internal_static_oneflow_ShapeElemCntAxisConf_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.operator.ShapeElemCntAxisConf.class, org.oneflow.core.operator.ShapeElemCntAxisConf.Builder.class);
    }

    // Construct using org.oneflow.core.operator.ShapeElemCntAxisConf.newBuilder()
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
      axis_ = java.util.Collections.emptyList();
      bitField0_ = (bitField0_ & ~0x00000001);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.operator.OpConf.internal_static_oneflow_ShapeElemCntAxisConf_descriptor;
    }

    public org.oneflow.core.operator.ShapeElemCntAxisConf getDefaultInstanceForType() {
      return org.oneflow.core.operator.ShapeElemCntAxisConf.getDefaultInstance();
    }

    public org.oneflow.core.operator.ShapeElemCntAxisConf build() {
      org.oneflow.core.operator.ShapeElemCntAxisConf result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.operator.ShapeElemCntAxisConf buildPartial() {
      org.oneflow.core.operator.ShapeElemCntAxisConf result = new org.oneflow.core.operator.ShapeElemCntAxisConf(this);
      int from_bitField0_ = bitField0_;
      if (((bitField0_ & 0x00000001) == 0x00000001)) {
        axis_ = java.util.Collections.unmodifiableList(axis_);
        bitField0_ = (bitField0_ & ~0x00000001);
      }
      result.axis_ = axis_;
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
      if (other instanceof org.oneflow.core.operator.ShapeElemCntAxisConf) {
        return mergeFrom((org.oneflow.core.operator.ShapeElemCntAxisConf)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.operator.ShapeElemCntAxisConf other) {
      if (other == org.oneflow.core.operator.ShapeElemCntAxisConf.getDefaultInstance()) return this;
      if (!other.axis_.isEmpty()) {
        if (axis_.isEmpty()) {
          axis_ = other.axis_;
          bitField0_ = (bitField0_ & ~0x00000001);
        } else {
          ensureAxisIsMutable();
          axis_.addAll(other.axis_);
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
      org.oneflow.core.operator.ShapeElemCntAxisConf parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.operator.ShapeElemCntAxisConf) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private java.util.List<java.lang.Integer> axis_ = java.util.Collections.emptyList();
    private void ensureAxisIsMutable() {
      if (!((bitField0_ & 0x00000001) == 0x00000001)) {
        axis_ = new java.util.ArrayList<java.lang.Integer>(axis_);
        bitField0_ |= 0x00000001;
       }
    }
    /**
     * <code>repeated int32 axis = 1;</code>
     */
    public java.util.List<java.lang.Integer>
        getAxisList() {
      return java.util.Collections.unmodifiableList(axis_);
    }
    /**
     * <code>repeated int32 axis = 1;</code>
     */
    public int getAxisCount() {
      return axis_.size();
    }
    /**
     * <code>repeated int32 axis = 1;</code>
     */
    public int getAxis(int index) {
      return axis_.get(index);
    }
    /**
     * <code>repeated int32 axis = 1;</code>
     */
    public Builder setAxis(
        int index, int value) {
      ensureAxisIsMutable();
      axis_.set(index, value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated int32 axis = 1;</code>
     */
    public Builder addAxis(int value) {
      ensureAxisIsMutable();
      axis_.add(value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated int32 axis = 1;</code>
     */
    public Builder addAllAxis(
        java.lang.Iterable<? extends java.lang.Integer> values) {
      ensureAxisIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, axis_);
      onChanged();
      return this;
    }
    /**
     * <code>repeated int32 axis = 1;</code>
     */
    public Builder clearAxis() {
      axis_ = java.util.Collections.emptyList();
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


    // @@protoc_insertion_point(builder_scope:oneflow.ShapeElemCntAxisConf)
  }

  // @@protoc_insertion_point(class_scope:oneflow.ShapeElemCntAxisConf)
  private static final org.oneflow.core.operator.ShapeElemCntAxisConf DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.operator.ShapeElemCntAxisConf();
  }

  public static org.oneflow.core.operator.ShapeElemCntAxisConf getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<ShapeElemCntAxisConf>
      PARSER = new com.google.protobuf.AbstractParser<ShapeElemCntAxisConf>() {
    public ShapeElemCntAxisConf parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new ShapeElemCntAxisConf(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<ShapeElemCntAxisConf> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<ShapeElemCntAxisConf> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.operator.ShapeElemCntAxisConf getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

