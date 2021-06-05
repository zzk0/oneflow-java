// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/operator/op_conf.proto

package org.oneflow.core.operator;

/**
 * Protobuf type {@code oneflow.ConstantLikeOpConf}
 */
public  final class ConstantLikeOpConf extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.ConstantLikeOpConf)
    ConstantLikeOpConfOrBuilder {
  // Use ConstantLikeOpConf.newBuilder() to construct.
  private ConstantLikeOpConf(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private ConstantLikeOpConf() {
    like_ = "";
    out_ = "";
    dataType_ = 0;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private ConstantLikeOpConf(
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
            like_ = bs;
            break;
          }
          case 18: {
            com.google.protobuf.ByteString bs = input.readBytes();
            bitField0_ |= 0x00000002;
            out_ = bs;
            break;
          }
          case 24: {
            int rawValue = input.readEnum();
            org.oneflow.core.common.DataType value = org.oneflow.core.common.DataType.valueOf(rawValue);
            if (value == null) {
              unknownFields.mergeVarintField(3, rawValue);
            } else {
              bitField0_ |= 0x00000004;
              dataType_ = rawValue;
            }
            break;
          }
          case 32: {
            scalarOperandCase_ = 4;
            scalarOperand_ = input.readInt64();
            break;
          }
          case 41: {
            scalarOperandCase_ = 5;
            scalarOperand_ = input.readDouble();
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
    return org.oneflow.core.operator.OpConf.internal_static_oneflow_ConstantLikeOpConf_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.operator.OpConf.internal_static_oneflow_ConstantLikeOpConf_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.operator.ConstantLikeOpConf.class, org.oneflow.core.operator.ConstantLikeOpConf.Builder.class);
  }

  private int bitField0_;
  private int scalarOperandCase_ = 0;
  private java.lang.Object scalarOperand_;
  public enum ScalarOperandCase
      implements com.google.protobuf.Internal.EnumLite {
    INT_OPERAND(4),
    FLOAT_OPERAND(5),
    SCALAROPERAND_NOT_SET(0);
    private final int value;
    private ScalarOperandCase(int value) {
      this.value = value;
    }
    /**
     * @deprecated Use {@link #forNumber(int)} instead.
     */
    @java.lang.Deprecated
    public static ScalarOperandCase valueOf(int value) {
      return forNumber(value);
    }

    public static ScalarOperandCase forNumber(int value) {
      switch (value) {
        case 4: return INT_OPERAND;
        case 5: return FLOAT_OPERAND;
        case 0: return SCALAROPERAND_NOT_SET;
        default: return null;
      }
    }
    public int getNumber() {
      return this.value;
    }
  };

  public ScalarOperandCase
  getScalarOperandCase() {
    return ScalarOperandCase.forNumber(
        scalarOperandCase_);
  }

  public static final int LIKE_FIELD_NUMBER = 1;
  private volatile java.lang.Object like_;
  /**
   * <code>required string like = 1;</code>
   */
  public boolean hasLike() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>required string like = 1;</code>
   */
  public java.lang.String getLike() {
    java.lang.Object ref = like_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      if (bs.isValidUtf8()) {
        like_ = s;
      }
      return s;
    }
  }
  /**
   * <code>required string like = 1;</code>
   */
  public com.google.protobuf.ByteString
      getLikeBytes() {
    java.lang.Object ref = like_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      like_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int OUT_FIELD_NUMBER = 2;
  private volatile java.lang.Object out_;
  /**
   * <code>required string out = 2;</code>
   */
  public boolean hasOut() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>required string out = 2;</code>
   */
  public java.lang.String getOut() {
    java.lang.Object ref = out_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      if (bs.isValidUtf8()) {
        out_ = s;
      }
      return s;
    }
  }
  /**
   * <code>required string out = 2;</code>
   */
  public com.google.protobuf.ByteString
      getOutBytes() {
    java.lang.Object ref = out_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      out_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int DATA_TYPE_FIELD_NUMBER = 3;
  private int dataType_;
  /**
   * <code>optional .oneflow.DataType data_type = 3;</code>
   */
  public boolean hasDataType() {
    return ((bitField0_ & 0x00000004) == 0x00000004);
  }
  /**
   * <code>optional .oneflow.DataType data_type = 3;</code>
   */
  public org.oneflow.core.common.DataType getDataType() {
    org.oneflow.core.common.DataType result = org.oneflow.core.common.DataType.valueOf(dataType_);
    return result == null ? org.oneflow.core.common.DataType.kInvalidDataType : result;
  }

  public static final int INT_OPERAND_FIELD_NUMBER = 4;
  /**
   * <code>optional int64 int_operand = 4;</code>
   */
  public boolean hasIntOperand() {
    return scalarOperandCase_ == 4;
  }
  /**
   * <code>optional int64 int_operand = 4;</code>
   */
  public long getIntOperand() {
    if (scalarOperandCase_ == 4) {
      return (java.lang.Long) scalarOperand_;
    }
    return 0L;
  }

  public static final int FLOAT_OPERAND_FIELD_NUMBER = 5;
  /**
   * <code>optional double float_operand = 5;</code>
   */
  public boolean hasFloatOperand() {
    return scalarOperandCase_ == 5;
  }
  /**
   * <code>optional double float_operand = 5;</code>
   */
  public double getFloatOperand() {
    if (scalarOperandCase_ == 5) {
      return (java.lang.Double) scalarOperand_;
    }
    return 0D;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (!hasLike()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!hasOut()) {
      memoizedIsInitialized = 0;
      return false;
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, like_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 2, out_);
    }
    if (((bitField0_ & 0x00000004) == 0x00000004)) {
      output.writeEnum(3, dataType_);
    }
    if (scalarOperandCase_ == 4) {
      output.writeInt64(
          4, (long)((java.lang.Long) scalarOperand_));
    }
    if (scalarOperandCase_ == 5) {
      output.writeDouble(
          5, (double)((java.lang.Double) scalarOperand_));
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(1, like_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(2, out_);
    }
    if (((bitField0_ & 0x00000004) == 0x00000004)) {
      size += com.google.protobuf.CodedOutputStream
        .computeEnumSize(3, dataType_);
    }
    if (scalarOperandCase_ == 4) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt64Size(
            4, (long)((java.lang.Long) scalarOperand_));
    }
    if (scalarOperandCase_ == 5) {
      size += com.google.protobuf.CodedOutputStream
        .computeDoubleSize(
            5, (double)((java.lang.Double) scalarOperand_));
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
    if (!(obj instanceof org.oneflow.core.operator.ConstantLikeOpConf)) {
      return super.equals(obj);
    }
    org.oneflow.core.operator.ConstantLikeOpConf other = (org.oneflow.core.operator.ConstantLikeOpConf) obj;

    boolean result = true;
    result = result && (hasLike() == other.hasLike());
    if (hasLike()) {
      result = result && getLike()
          .equals(other.getLike());
    }
    result = result && (hasOut() == other.hasOut());
    if (hasOut()) {
      result = result && getOut()
          .equals(other.getOut());
    }
    result = result && (hasDataType() == other.hasDataType());
    if (hasDataType()) {
      result = result && dataType_ == other.dataType_;
    }
    result = result && getScalarOperandCase().equals(
        other.getScalarOperandCase());
    if (!result) return false;
    switch (scalarOperandCase_) {
      case 4:
        result = result && (getIntOperand()
            == other.getIntOperand());
        break;
      case 5:
        result = result && (
            java.lang.Double.doubleToLongBits(getFloatOperand())
            == java.lang.Double.doubleToLongBits(
                other.getFloatOperand()));
        break;
      case 0:
      default:
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
    if (hasLike()) {
      hash = (37 * hash) + LIKE_FIELD_NUMBER;
      hash = (53 * hash) + getLike().hashCode();
    }
    if (hasOut()) {
      hash = (37 * hash) + OUT_FIELD_NUMBER;
      hash = (53 * hash) + getOut().hashCode();
    }
    if (hasDataType()) {
      hash = (37 * hash) + DATA_TYPE_FIELD_NUMBER;
      hash = (53 * hash) + dataType_;
    }
    switch (scalarOperandCase_) {
      case 4:
        hash = (37 * hash) + INT_OPERAND_FIELD_NUMBER;
        hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
            getIntOperand());
        break;
      case 5:
        hash = (37 * hash) + FLOAT_OPERAND_FIELD_NUMBER;
        hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
            java.lang.Double.doubleToLongBits(getFloatOperand()));
        break;
      case 0:
      default:
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.operator.ConstantLikeOpConf parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.operator.ConstantLikeOpConf parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.operator.ConstantLikeOpConf parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.operator.ConstantLikeOpConf parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.operator.ConstantLikeOpConf parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.operator.ConstantLikeOpConf parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.operator.ConstantLikeOpConf parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.operator.ConstantLikeOpConf parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.operator.ConstantLikeOpConf parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.operator.ConstantLikeOpConf parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.operator.ConstantLikeOpConf prototype) {
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
   * Protobuf type {@code oneflow.ConstantLikeOpConf}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.ConstantLikeOpConf)
      org.oneflow.core.operator.ConstantLikeOpConfOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.operator.OpConf.internal_static_oneflow_ConstantLikeOpConf_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.operator.OpConf.internal_static_oneflow_ConstantLikeOpConf_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.operator.ConstantLikeOpConf.class, org.oneflow.core.operator.ConstantLikeOpConf.Builder.class);
    }

    // Construct using org.oneflow.core.operator.ConstantLikeOpConf.newBuilder()
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
      like_ = "";
      bitField0_ = (bitField0_ & ~0x00000001);
      out_ = "";
      bitField0_ = (bitField0_ & ~0x00000002);
      dataType_ = 0;
      bitField0_ = (bitField0_ & ~0x00000004);
      scalarOperandCase_ = 0;
      scalarOperand_ = null;
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.operator.OpConf.internal_static_oneflow_ConstantLikeOpConf_descriptor;
    }

    public org.oneflow.core.operator.ConstantLikeOpConf getDefaultInstanceForType() {
      return org.oneflow.core.operator.ConstantLikeOpConf.getDefaultInstance();
    }

    public org.oneflow.core.operator.ConstantLikeOpConf build() {
      org.oneflow.core.operator.ConstantLikeOpConf result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.operator.ConstantLikeOpConf buildPartial() {
      org.oneflow.core.operator.ConstantLikeOpConf result = new org.oneflow.core.operator.ConstantLikeOpConf(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      result.like_ = like_;
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000002;
      }
      result.out_ = out_;
      if (((from_bitField0_ & 0x00000004) == 0x00000004)) {
        to_bitField0_ |= 0x00000004;
      }
      result.dataType_ = dataType_;
      if (scalarOperandCase_ == 4) {
        result.scalarOperand_ = scalarOperand_;
      }
      if (scalarOperandCase_ == 5) {
        result.scalarOperand_ = scalarOperand_;
      }
      result.bitField0_ = to_bitField0_;
      result.scalarOperandCase_ = scalarOperandCase_;
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
      if (other instanceof org.oneflow.core.operator.ConstantLikeOpConf) {
        return mergeFrom((org.oneflow.core.operator.ConstantLikeOpConf)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.operator.ConstantLikeOpConf other) {
      if (other == org.oneflow.core.operator.ConstantLikeOpConf.getDefaultInstance()) return this;
      if (other.hasLike()) {
        bitField0_ |= 0x00000001;
        like_ = other.like_;
        onChanged();
      }
      if (other.hasOut()) {
        bitField0_ |= 0x00000002;
        out_ = other.out_;
        onChanged();
      }
      if (other.hasDataType()) {
        setDataType(other.getDataType());
      }
      switch (other.getScalarOperandCase()) {
        case INT_OPERAND: {
          setIntOperand(other.getIntOperand());
          break;
        }
        case FLOAT_OPERAND: {
          setFloatOperand(other.getFloatOperand());
          break;
        }
        case SCALAROPERAND_NOT_SET: {
          break;
        }
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (!hasLike()) {
        return false;
      }
      if (!hasOut()) {
        return false;
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.operator.ConstantLikeOpConf parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.operator.ConstantLikeOpConf) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int scalarOperandCase_ = 0;
    private java.lang.Object scalarOperand_;
    public ScalarOperandCase
        getScalarOperandCase() {
      return ScalarOperandCase.forNumber(
          scalarOperandCase_);
    }

    public Builder clearScalarOperand() {
      scalarOperandCase_ = 0;
      scalarOperand_ = null;
      onChanged();
      return this;
    }

    private int bitField0_;

    private java.lang.Object like_ = "";
    /**
     * <code>required string like = 1;</code>
     */
    public boolean hasLike() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>required string like = 1;</code>
     */
    public java.lang.String getLike() {
      java.lang.Object ref = like_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        if (bs.isValidUtf8()) {
          like_ = s;
        }
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>required string like = 1;</code>
     */
    public com.google.protobuf.ByteString
        getLikeBytes() {
      java.lang.Object ref = like_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        like_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>required string like = 1;</code>
     */
    public Builder setLike(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000001;
      like_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>required string like = 1;</code>
     */
    public Builder clearLike() {
      bitField0_ = (bitField0_ & ~0x00000001);
      like_ = getDefaultInstance().getLike();
      onChanged();
      return this;
    }
    /**
     * <code>required string like = 1;</code>
     */
    public Builder setLikeBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000001;
      like_ = value;
      onChanged();
      return this;
    }

    private java.lang.Object out_ = "";
    /**
     * <code>required string out = 2;</code>
     */
    public boolean hasOut() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>required string out = 2;</code>
     */
    public java.lang.String getOut() {
      java.lang.Object ref = out_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        if (bs.isValidUtf8()) {
          out_ = s;
        }
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>required string out = 2;</code>
     */
    public com.google.protobuf.ByteString
        getOutBytes() {
      java.lang.Object ref = out_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        out_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>required string out = 2;</code>
     */
    public Builder setOut(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000002;
      out_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>required string out = 2;</code>
     */
    public Builder clearOut() {
      bitField0_ = (bitField0_ & ~0x00000002);
      out_ = getDefaultInstance().getOut();
      onChanged();
      return this;
    }
    /**
     * <code>required string out = 2;</code>
     */
    public Builder setOutBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000002;
      out_ = value;
      onChanged();
      return this;
    }

    private int dataType_ = 0;
    /**
     * <code>optional .oneflow.DataType data_type = 3;</code>
     */
    public boolean hasDataType() {
      return ((bitField0_ & 0x00000004) == 0x00000004);
    }
    /**
     * <code>optional .oneflow.DataType data_type = 3;</code>
     */
    public org.oneflow.core.common.DataType getDataType() {
      org.oneflow.core.common.DataType result = org.oneflow.core.common.DataType.valueOf(dataType_);
      return result == null ? org.oneflow.core.common.DataType.kInvalidDataType : result;
    }
    /**
     * <code>optional .oneflow.DataType data_type = 3;</code>
     */
    public Builder setDataType(org.oneflow.core.common.DataType value) {
      if (value == null) {
        throw new NullPointerException();
      }
      bitField0_ |= 0x00000004;
      dataType_ = value.getNumber();
      onChanged();
      return this;
    }
    /**
     * <code>optional .oneflow.DataType data_type = 3;</code>
     */
    public Builder clearDataType() {
      bitField0_ = (bitField0_ & ~0x00000004);
      dataType_ = 0;
      onChanged();
      return this;
    }

    /**
     * <code>optional int64 int_operand = 4;</code>
     */
    public boolean hasIntOperand() {
      return scalarOperandCase_ == 4;
    }
    /**
     * <code>optional int64 int_operand = 4;</code>
     */
    public long getIntOperand() {
      if (scalarOperandCase_ == 4) {
        return (java.lang.Long) scalarOperand_;
      }
      return 0L;
    }
    /**
     * <code>optional int64 int_operand = 4;</code>
     */
    public Builder setIntOperand(long value) {
      scalarOperandCase_ = 4;
      scalarOperand_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional int64 int_operand = 4;</code>
     */
    public Builder clearIntOperand() {
      if (scalarOperandCase_ == 4) {
        scalarOperandCase_ = 0;
        scalarOperand_ = null;
        onChanged();
      }
      return this;
    }

    /**
     * <code>optional double float_operand = 5;</code>
     */
    public boolean hasFloatOperand() {
      return scalarOperandCase_ == 5;
    }
    /**
     * <code>optional double float_operand = 5;</code>
     */
    public double getFloatOperand() {
      if (scalarOperandCase_ == 5) {
        return (java.lang.Double) scalarOperand_;
      }
      return 0D;
    }
    /**
     * <code>optional double float_operand = 5;</code>
     */
    public Builder setFloatOperand(double value) {
      scalarOperandCase_ = 5;
      scalarOperand_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional double float_operand = 5;</code>
     */
    public Builder clearFloatOperand() {
      if (scalarOperandCase_ == 5) {
        scalarOperandCase_ = 0;
        scalarOperand_ = null;
        onChanged();
      }
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


    // @@protoc_insertion_point(builder_scope:oneflow.ConstantLikeOpConf)
  }

  // @@protoc_insertion_point(class_scope:oneflow.ConstantLikeOpConf)
  private static final org.oneflow.core.operator.ConstantLikeOpConf DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.operator.ConstantLikeOpConf();
  }

  public static org.oneflow.core.operator.ConstantLikeOpConf getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<ConstantLikeOpConf>
      PARSER = new com.google.protobuf.AbstractParser<ConstantLikeOpConf>() {
    public ConstantLikeOpConf parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new ConstantLikeOpConf(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<ConstantLikeOpConf> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<ConstantLikeOpConf> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.operator.ConstantLikeOpConf getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

