// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/register/pod.proto

package org.oneflow.core.register;

/**
 * Protobuf type {@code oneflow.TensorPodProto}
 */
public  final class TensorPodProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.TensorPodProto)
    TensorPodProtoOrBuilder {
  // Use TensorPodProto.newBuilder() to construct.
  private TensorPodProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private TensorPodProto() {
    dataType_ = 0;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private TensorPodProto(
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
            org.oneflow.core.common.ShapeProto.Builder subBuilder = null;
            if (((bitField0_ & 0x00000001) == 0x00000001)) {
              subBuilder = shape_.toBuilder();
            }
            shape_ = input.readMessage(org.oneflow.core.common.ShapeProto.PARSER, extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(shape_);
              shape_ = subBuilder.buildPartial();
            }
            bitField0_ |= 0x00000001;
            break;
          }
          case 16: {
            int rawValue = input.readEnum();
            org.oneflow.core.common.DataType value = org.oneflow.core.common.DataType.valueOf(rawValue);
            if (value == null) {
              unknownFields.mergeVarintField(2, rawValue);
            } else {
              bitField0_ |= 0x00000002;
              dataType_ = rawValue;
            }
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
    return org.oneflow.core.register.Pod.internal_static_oneflow_TensorPodProto_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.register.Pod.internal_static_oneflow_TensorPodProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.register.TensorPodProto.class, org.oneflow.core.register.TensorPodProto.Builder.class);
  }

  private int bitField0_;
  public static final int SHAPE_FIELD_NUMBER = 1;
  private org.oneflow.core.common.ShapeProto shape_;
  /**
   * <code>required .oneflow.ShapeProto shape = 1;</code>
   */
  public boolean hasShape() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>required .oneflow.ShapeProto shape = 1;</code>
   */
  public org.oneflow.core.common.ShapeProto getShape() {
    return shape_ == null ? org.oneflow.core.common.ShapeProto.getDefaultInstance() : shape_;
  }
  /**
   * <code>required .oneflow.ShapeProto shape = 1;</code>
   */
  public org.oneflow.core.common.ShapeProtoOrBuilder getShapeOrBuilder() {
    return shape_ == null ? org.oneflow.core.common.ShapeProto.getDefaultInstance() : shape_;
  }

  public static final int DATA_TYPE_FIELD_NUMBER = 2;
  private int dataType_;
  /**
   * <code>required .oneflow.DataType data_type = 2;</code>
   */
  public boolean hasDataType() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>required .oneflow.DataType data_type = 2;</code>
   */
  public org.oneflow.core.common.DataType getDataType() {
    org.oneflow.core.common.DataType result = org.oneflow.core.common.DataType.valueOf(dataType_);
    return result == null ? org.oneflow.core.common.DataType.kInvalidDataType : result;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (!hasShape()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!hasDataType()) {
      memoizedIsInitialized = 0;
      return false;
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      output.writeMessage(1, getShape());
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      output.writeEnum(2, dataType_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(1, getShape());
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.CodedOutputStream
        .computeEnumSize(2, dataType_);
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
    if (!(obj instanceof org.oneflow.core.register.TensorPodProto)) {
      return super.equals(obj);
    }
    org.oneflow.core.register.TensorPodProto other = (org.oneflow.core.register.TensorPodProto) obj;

    boolean result = true;
    result = result && (hasShape() == other.hasShape());
    if (hasShape()) {
      result = result && getShape()
          .equals(other.getShape());
    }
    result = result && (hasDataType() == other.hasDataType());
    if (hasDataType()) {
      result = result && dataType_ == other.dataType_;
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
    if (hasShape()) {
      hash = (37 * hash) + SHAPE_FIELD_NUMBER;
      hash = (53 * hash) + getShape().hashCode();
    }
    if (hasDataType()) {
      hash = (37 * hash) + DATA_TYPE_FIELD_NUMBER;
      hash = (53 * hash) + dataType_;
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.register.TensorPodProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.register.TensorPodProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.register.TensorPodProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.register.TensorPodProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.register.TensorPodProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.TensorPodProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.register.TensorPodProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.TensorPodProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.register.TensorPodProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.TensorPodProto parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.register.TensorPodProto prototype) {
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
   * Protobuf type {@code oneflow.TensorPodProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.TensorPodProto)
      org.oneflow.core.register.TensorPodProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.register.Pod.internal_static_oneflow_TensorPodProto_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.register.Pod.internal_static_oneflow_TensorPodProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.register.TensorPodProto.class, org.oneflow.core.register.TensorPodProto.Builder.class);
    }

    // Construct using org.oneflow.core.register.TensorPodProto.newBuilder()
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
        getShapeFieldBuilder();
      }
    }
    public Builder clear() {
      super.clear();
      if (shapeBuilder_ == null) {
        shape_ = null;
      } else {
        shapeBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000001);
      dataType_ = 0;
      bitField0_ = (bitField0_ & ~0x00000002);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.register.Pod.internal_static_oneflow_TensorPodProto_descriptor;
    }

    public org.oneflow.core.register.TensorPodProto getDefaultInstanceForType() {
      return org.oneflow.core.register.TensorPodProto.getDefaultInstance();
    }

    public org.oneflow.core.register.TensorPodProto build() {
      org.oneflow.core.register.TensorPodProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.register.TensorPodProto buildPartial() {
      org.oneflow.core.register.TensorPodProto result = new org.oneflow.core.register.TensorPodProto(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      if (shapeBuilder_ == null) {
        result.shape_ = shape_;
      } else {
        result.shape_ = shapeBuilder_.build();
      }
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000002;
      }
      result.dataType_ = dataType_;
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
      if (other instanceof org.oneflow.core.register.TensorPodProto) {
        return mergeFrom((org.oneflow.core.register.TensorPodProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.register.TensorPodProto other) {
      if (other == org.oneflow.core.register.TensorPodProto.getDefaultInstance()) return this;
      if (other.hasShape()) {
        mergeShape(other.getShape());
      }
      if (other.hasDataType()) {
        setDataType(other.getDataType());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (!hasShape()) {
        return false;
      }
      if (!hasDataType()) {
        return false;
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.register.TensorPodProto parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.register.TensorPodProto) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private org.oneflow.core.common.ShapeProto shape_ = null;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.common.ShapeProto, org.oneflow.core.common.ShapeProto.Builder, org.oneflow.core.common.ShapeProtoOrBuilder> shapeBuilder_;
    /**
     * <code>required .oneflow.ShapeProto shape = 1;</code>
     */
    public boolean hasShape() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>required .oneflow.ShapeProto shape = 1;</code>
     */
    public org.oneflow.core.common.ShapeProto getShape() {
      if (shapeBuilder_ == null) {
        return shape_ == null ? org.oneflow.core.common.ShapeProto.getDefaultInstance() : shape_;
      } else {
        return shapeBuilder_.getMessage();
      }
    }
    /**
     * <code>required .oneflow.ShapeProto shape = 1;</code>
     */
    public Builder setShape(org.oneflow.core.common.ShapeProto value) {
      if (shapeBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        shape_ = value;
        onChanged();
      } else {
        shapeBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>required .oneflow.ShapeProto shape = 1;</code>
     */
    public Builder setShape(
        org.oneflow.core.common.ShapeProto.Builder builderForValue) {
      if (shapeBuilder_ == null) {
        shape_ = builderForValue.build();
        onChanged();
      } else {
        shapeBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>required .oneflow.ShapeProto shape = 1;</code>
     */
    public Builder mergeShape(org.oneflow.core.common.ShapeProto value) {
      if (shapeBuilder_ == null) {
        if (((bitField0_ & 0x00000001) == 0x00000001) &&
            shape_ != null &&
            shape_ != org.oneflow.core.common.ShapeProto.getDefaultInstance()) {
          shape_ =
            org.oneflow.core.common.ShapeProto.newBuilder(shape_).mergeFrom(value).buildPartial();
        } else {
          shape_ = value;
        }
        onChanged();
      } else {
        shapeBuilder_.mergeFrom(value);
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>required .oneflow.ShapeProto shape = 1;</code>
     */
    public Builder clearShape() {
      if (shapeBuilder_ == null) {
        shape_ = null;
        onChanged();
      } else {
        shapeBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000001);
      return this;
    }
    /**
     * <code>required .oneflow.ShapeProto shape = 1;</code>
     */
    public org.oneflow.core.common.ShapeProto.Builder getShapeBuilder() {
      bitField0_ |= 0x00000001;
      onChanged();
      return getShapeFieldBuilder().getBuilder();
    }
    /**
     * <code>required .oneflow.ShapeProto shape = 1;</code>
     */
    public org.oneflow.core.common.ShapeProtoOrBuilder getShapeOrBuilder() {
      if (shapeBuilder_ != null) {
        return shapeBuilder_.getMessageOrBuilder();
      } else {
        return shape_ == null ?
            org.oneflow.core.common.ShapeProto.getDefaultInstance() : shape_;
      }
    }
    /**
     * <code>required .oneflow.ShapeProto shape = 1;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.common.ShapeProto, org.oneflow.core.common.ShapeProto.Builder, org.oneflow.core.common.ShapeProtoOrBuilder> 
        getShapeFieldBuilder() {
      if (shapeBuilder_ == null) {
        shapeBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.oneflow.core.common.ShapeProto, org.oneflow.core.common.ShapeProto.Builder, org.oneflow.core.common.ShapeProtoOrBuilder>(
                getShape(),
                getParentForChildren(),
                isClean());
        shape_ = null;
      }
      return shapeBuilder_;
    }

    private int dataType_ = 0;
    /**
     * <code>required .oneflow.DataType data_type = 2;</code>
     */
    public boolean hasDataType() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>required .oneflow.DataType data_type = 2;</code>
     */
    public org.oneflow.core.common.DataType getDataType() {
      org.oneflow.core.common.DataType result = org.oneflow.core.common.DataType.valueOf(dataType_);
      return result == null ? org.oneflow.core.common.DataType.kInvalidDataType : result;
    }
    /**
     * <code>required .oneflow.DataType data_type = 2;</code>
     */
    public Builder setDataType(org.oneflow.core.common.DataType value) {
      if (value == null) {
        throw new NullPointerException();
      }
      bitField0_ |= 0x00000002;
      dataType_ = value.getNumber();
      onChanged();
      return this;
    }
    /**
     * <code>required .oneflow.DataType data_type = 2;</code>
     */
    public Builder clearDataType() {
      bitField0_ = (bitField0_ & ~0x00000002);
      dataType_ = 0;
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


    // @@protoc_insertion_point(builder_scope:oneflow.TensorPodProto)
  }

  // @@protoc_insertion_point(class_scope:oneflow.TensorPodProto)
  private static final org.oneflow.core.register.TensorPodProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.register.TensorPodProto();
  }

  public static org.oneflow.core.register.TensorPodProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<TensorPodProto>
      PARSER = new com.google.protobuf.AbstractParser<TensorPodProto>() {
    public TensorPodProto parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new TensorPodProto(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<TensorPodProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<TensorPodProto> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.register.TensorPodProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

