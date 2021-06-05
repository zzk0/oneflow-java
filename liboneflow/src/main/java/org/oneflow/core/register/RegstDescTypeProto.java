// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/register/register_desc.proto

package org.oneflow.core.register;

/**
 * Protobuf type {@code oneflow.RegstDescTypeProto}
 */
public  final class RegstDescTypeProto extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.RegstDescTypeProto)
    RegstDescTypeProtoOrBuilder {
  // Use RegstDescTypeProto.newBuilder() to construct.
  private RegstDescTypeProto(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private RegstDescTypeProto() {
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private RegstDescTypeProto(
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
            org.oneflow.core.register.DataRegstDesc.Builder subBuilder = null;
            if (typeCase_ == 1) {
              subBuilder = ((org.oneflow.core.register.DataRegstDesc) type_).toBuilder();
            }
            type_ =
                input.readMessage(org.oneflow.core.register.DataRegstDesc.PARSER, extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom((org.oneflow.core.register.DataRegstDesc) type_);
              type_ = subBuilder.buildPartial();
            }
            typeCase_ = 1;
            break;
          }
          case 26: {
            org.oneflow.core.register.CtrlRegstDesc.Builder subBuilder = null;
            if (typeCase_ == 3) {
              subBuilder = ((org.oneflow.core.register.CtrlRegstDesc) type_).toBuilder();
            }
            type_ =
                input.readMessage(org.oneflow.core.register.CtrlRegstDesc.PARSER, extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom((org.oneflow.core.register.CtrlRegstDesc) type_);
              type_ = subBuilder.buildPartial();
            }
            typeCase_ = 3;
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
    return org.oneflow.core.register.RegisterDesc.internal_static_oneflow_RegstDescTypeProto_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.register.RegisterDesc.internal_static_oneflow_RegstDescTypeProto_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.register.RegstDescTypeProto.class, org.oneflow.core.register.RegstDescTypeProto.Builder.class);
  }

  private int bitField0_;
  private int typeCase_ = 0;
  private java.lang.Object type_;
  public enum TypeCase
      implements com.google.protobuf.Internal.EnumLite {
    DATA_REGST_DESC(1),
    CTRL_REGST_DESC(3),
    TYPE_NOT_SET(0);
    private final int value;
    private TypeCase(int value) {
      this.value = value;
    }
    /**
     * @deprecated Use {@link #forNumber(int)} instead.
     */
    @java.lang.Deprecated
    public static TypeCase valueOf(int value) {
      return forNumber(value);
    }

    public static TypeCase forNumber(int value) {
      switch (value) {
        case 1: return DATA_REGST_DESC;
        case 3: return CTRL_REGST_DESC;
        case 0: return TYPE_NOT_SET;
        default: return null;
      }
    }
    public int getNumber() {
      return this.value;
    }
  };

  public TypeCase
  getTypeCase() {
    return TypeCase.forNumber(
        typeCase_);
  }

  public static final int DATA_REGST_DESC_FIELD_NUMBER = 1;
  /**
   * <code>optional .oneflow.DataRegstDesc data_regst_desc = 1;</code>
   */
  public boolean hasDataRegstDesc() {
    return typeCase_ == 1;
  }
  /**
   * <code>optional .oneflow.DataRegstDesc data_regst_desc = 1;</code>
   */
  public org.oneflow.core.register.DataRegstDesc getDataRegstDesc() {
    if (typeCase_ == 1) {
       return (org.oneflow.core.register.DataRegstDesc) type_;
    }
    return org.oneflow.core.register.DataRegstDesc.getDefaultInstance();
  }
  /**
   * <code>optional .oneflow.DataRegstDesc data_regst_desc = 1;</code>
   */
  public org.oneflow.core.register.DataRegstDescOrBuilder getDataRegstDescOrBuilder() {
    if (typeCase_ == 1) {
       return (org.oneflow.core.register.DataRegstDesc) type_;
    }
    return org.oneflow.core.register.DataRegstDesc.getDefaultInstance();
  }

  public static final int CTRL_REGST_DESC_FIELD_NUMBER = 3;
  /**
   * <code>optional .oneflow.CtrlRegstDesc ctrl_regst_desc = 3;</code>
   */
  public boolean hasCtrlRegstDesc() {
    return typeCase_ == 3;
  }
  /**
   * <code>optional .oneflow.CtrlRegstDesc ctrl_regst_desc = 3;</code>
   */
  public org.oneflow.core.register.CtrlRegstDesc getCtrlRegstDesc() {
    if (typeCase_ == 3) {
       return (org.oneflow.core.register.CtrlRegstDesc) type_;
    }
    return org.oneflow.core.register.CtrlRegstDesc.getDefaultInstance();
  }
  /**
   * <code>optional .oneflow.CtrlRegstDesc ctrl_regst_desc = 3;</code>
   */
  public org.oneflow.core.register.CtrlRegstDescOrBuilder getCtrlRegstDescOrBuilder() {
    if (typeCase_ == 3) {
       return (org.oneflow.core.register.CtrlRegstDesc) type_;
    }
    return org.oneflow.core.register.CtrlRegstDesc.getDefaultInstance();
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (hasDataRegstDesc()) {
      if (!getDataRegstDesc().isInitialized()) {
        memoizedIsInitialized = 0;
        return false;
      }
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (typeCase_ == 1) {
      output.writeMessage(1, (org.oneflow.core.register.DataRegstDesc) type_);
    }
    if (typeCase_ == 3) {
      output.writeMessage(3, (org.oneflow.core.register.CtrlRegstDesc) type_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (typeCase_ == 1) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(1, (org.oneflow.core.register.DataRegstDesc) type_);
    }
    if (typeCase_ == 3) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(3, (org.oneflow.core.register.CtrlRegstDesc) type_);
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
    if (!(obj instanceof org.oneflow.core.register.RegstDescTypeProto)) {
      return super.equals(obj);
    }
    org.oneflow.core.register.RegstDescTypeProto other = (org.oneflow.core.register.RegstDescTypeProto) obj;

    boolean result = true;
    result = result && getTypeCase().equals(
        other.getTypeCase());
    if (!result) return false;
    switch (typeCase_) {
      case 1:
        result = result && getDataRegstDesc()
            .equals(other.getDataRegstDesc());
        break;
      case 3:
        result = result && getCtrlRegstDesc()
            .equals(other.getCtrlRegstDesc());
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
    switch (typeCase_) {
      case 1:
        hash = (37 * hash) + DATA_REGST_DESC_FIELD_NUMBER;
        hash = (53 * hash) + getDataRegstDesc().hashCode();
        break;
      case 3:
        hash = (37 * hash) + CTRL_REGST_DESC_FIELD_NUMBER;
        hash = (53 * hash) + getCtrlRegstDesc().hashCode();
        break;
      case 0:
      default:
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.register.RegstDescTypeProto parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.register.RegstDescTypeProto parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.register.RegstDescTypeProto parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.register.RegstDescTypeProto parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.register.RegstDescTypeProto parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.RegstDescTypeProto parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.register.RegstDescTypeProto parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.RegstDescTypeProto parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.register.RegstDescTypeProto parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.RegstDescTypeProto parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.register.RegstDescTypeProto prototype) {
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
   * Protobuf type {@code oneflow.RegstDescTypeProto}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.RegstDescTypeProto)
      org.oneflow.core.register.RegstDescTypeProtoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.register.RegisterDesc.internal_static_oneflow_RegstDescTypeProto_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.register.RegisterDesc.internal_static_oneflow_RegstDescTypeProto_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.register.RegstDescTypeProto.class, org.oneflow.core.register.RegstDescTypeProto.Builder.class);
    }

    // Construct using org.oneflow.core.register.RegstDescTypeProto.newBuilder()
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
      typeCase_ = 0;
      type_ = null;
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.register.RegisterDesc.internal_static_oneflow_RegstDescTypeProto_descriptor;
    }

    public org.oneflow.core.register.RegstDescTypeProto getDefaultInstanceForType() {
      return org.oneflow.core.register.RegstDescTypeProto.getDefaultInstance();
    }

    public org.oneflow.core.register.RegstDescTypeProto build() {
      org.oneflow.core.register.RegstDescTypeProto result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.register.RegstDescTypeProto buildPartial() {
      org.oneflow.core.register.RegstDescTypeProto result = new org.oneflow.core.register.RegstDescTypeProto(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (typeCase_ == 1) {
        if (dataRegstDescBuilder_ == null) {
          result.type_ = type_;
        } else {
          result.type_ = dataRegstDescBuilder_.build();
        }
      }
      if (typeCase_ == 3) {
        if (ctrlRegstDescBuilder_ == null) {
          result.type_ = type_;
        } else {
          result.type_ = ctrlRegstDescBuilder_.build();
        }
      }
      result.bitField0_ = to_bitField0_;
      result.typeCase_ = typeCase_;
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
      if (other instanceof org.oneflow.core.register.RegstDescTypeProto) {
        return mergeFrom((org.oneflow.core.register.RegstDescTypeProto)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.register.RegstDescTypeProto other) {
      if (other == org.oneflow.core.register.RegstDescTypeProto.getDefaultInstance()) return this;
      switch (other.getTypeCase()) {
        case DATA_REGST_DESC: {
          mergeDataRegstDesc(other.getDataRegstDesc());
          break;
        }
        case CTRL_REGST_DESC: {
          mergeCtrlRegstDesc(other.getCtrlRegstDesc());
          break;
        }
        case TYPE_NOT_SET: {
          break;
        }
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (hasDataRegstDesc()) {
        if (!getDataRegstDesc().isInitialized()) {
          return false;
        }
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.register.RegstDescTypeProto parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.register.RegstDescTypeProto) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int typeCase_ = 0;
    private java.lang.Object type_;
    public TypeCase
        getTypeCase() {
      return TypeCase.forNumber(
          typeCase_);
    }

    public Builder clearType() {
      typeCase_ = 0;
      type_ = null;
      onChanged();
      return this;
    }

    private int bitField0_;

    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.register.DataRegstDesc, org.oneflow.core.register.DataRegstDesc.Builder, org.oneflow.core.register.DataRegstDescOrBuilder> dataRegstDescBuilder_;
    /**
     * <code>optional .oneflow.DataRegstDesc data_regst_desc = 1;</code>
     */
    public boolean hasDataRegstDesc() {
      return typeCase_ == 1;
    }
    /**
     * <code>optional .oneflow.DataRegstDesc data_regst_desc = 1;</code>
     */
    public org.oneflow.core.register.DataRegstDesc getDataRegstDesc() {
      if (dataRegstDescBuilder_ == null) {
        if (typeCase_ == 1) {
          return (org.oneflow.core.register.DataRegstDesc) type_;
        }
        return org.oneflow.core.register.DataRegstDesc.getDefaultInstance();
      } else {
        if (typeCase_ == 1) {
          return dataRegstDescBuilder_.getMessage();
        }
        return org.oneflow.core.register.DataRegstDesc.getDefaultInstance();
      }
    }
    /**
     * <code>optional .oneflow.DataRegstDesc data_regst_desc = 1;</code>
     */
    public Builder setDataRegstDesc(org.oneflow.core.register.DataRegstDesc value) {
      if (dataRegstDescBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        type_ = value;
        onChanged();
      } else {
        dataRegstDescBuilder_.setMessage(value);
      }
      typeCase_ = 1;
      return this;
    }
    /**
     * <code>optional .oneflow.DataRegstDesc data_regst_desc = 1;</code>
     */
    public Builder setDataRegstDesc(
        org.oneflow.core.register.DataRegstDesc.Builder builderForValue) {
      if (dataRegstDescBuilder_ == null) {
        type_ = builderForValue.build();
        onChanged();
      } else {
        dataRegstDescBuilder_.setMessage(builderForValue.build());
      }
      typeCase_ = 1;
      return this;
    }
    /**
     * <code>optional .oneflow.DataRegstDesc data_regst_desc = 1;</code>
     */
    public Builder mergeDataRegstDesc(org.oneflow.core.register.DataRegstDesc value) {
      if (dataRegstDescBuilder_ == null) {
        if (typeCase_ == 1 &&
            type_ != org.oneflow.core.register.DataRegstDesc.getDefaultInstance()) {
          type_ = org.oneflow.core.register.DataRegstDesc.newBuilder((org.oneflow.core.register.DataRegstDesc) type_)
              .mergeFrom(value).buildPartial();
        } else {
          type_ = value;
        }
        onChanged();
      } else {
        if (typeCase_ == 1) {
          dataRegstDescBuilder_.mergeFrom(value);
        }
        dataRegstDescBuilder_.setMessage(value);
      }
      typeCase_ = 1;
      return this;
    }
    /**
     * <code>optional .oneflow.DataRegstDesc data_regst_desc = 1;</code>
     */
    public Builder clearDataRegstDesc() {
      if (dataRegstDescBuilder_ == null) {
        if (typeCase_ == 1) {
          typeCase_ = 0;
          type_ = null;
          onChanged();
        }
      } else {
        if (typeCase_ == 1) {
          typeCase_ = 0;
          type_ = null;
        }
        dataRegstDescBuilder_.clear();
      }
      return this;
    }
    /**
     * <code>optional .oneflow.DataRegstDesc data_regst_desc = 1;</code>
     */
    public org.oneflow.core.register.DataRegstDesc.Builder getDataRegstDescBuilder() {
      return getDataRegstDescFieldBuilder().getBuilder();
    }
    /**
     * <code>optional .oneflow.DataRegstDesc data_regst_desc = 1;</code>
     */
    public org.oneflow.core.register.DataRegstDescOrBuilder getDataRegstDescOrBuilder() {
      if ((typeCase_ == 1) && (dataRegstDescBuilder_ != null)) {
        return dataRegstDescBuilder_.getMessageOrBuilder();
      } else {
        if (typeCase_ == 1) {
          return (org.oneflow.core.register.DataRegstDesc) type_;
        }
        return org.oneflow.core.register.DataRegstDesc.getDefaultInstance();
      }
    }
    /**
     * <code>optional .oneflow.DataRegstDesc data_regst_desc = 1;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.register.DataRegstDesc, org.oneflow.core.register.DataRegstDesc.Builder, org.oneflow.core.register.DataRegstDescOrBuilder> 
        getDataRegstDescFieldBuilder() {
      if (dataRegstDescBuilder_ == null) {
        if (!(typeCase_ == 1)) {
          type_ = org.oneflow.core.register.DataRegstDesc.getDefaultInstance();
        }
        dataRegstDescBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.oneflow.core.register.DataRegstDesc, org.oneflow.core.register.DataRegstDesc.Builder, org.oneflow.core.register.DataRegstDescOrBuilder>(
                (org.oneflow.core.register.DataRegstDesc) type_,
                getParentForChildren(),
                isClean());
        type_ = null;
      }
      typeCase_ = 1;
      onChanged();;
      return dataRegstDescBuilder_;
    }

    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.register.CtrlRegstDesc, org.oneflow.core.register.CtrlRegstDesc.Builder, org.oneflow.core.register.CtrlRegstDescOrBuilder> ctrlRegstDescBuilder_;
    /**
     * <code>optional .oneflow.CtrlRegstDesc ctrl_regst_desc = 3;</code>
     */
    public boolean hasCtrlRegstDesc() {
      return typeCase_ == 3;
    }
    /**
     * <code>optional .oneflow.CtrlRegstDesc ctrl_regst_desc = 3;</code>
     */
    public org.oneflow.core.register.CtrlRegstDesc getCtrlRegstDesc() {
      if (ctrlRegstDescBuilder_ == null) {
        if (typeCase_ == 3) {
          return (org.oneflow.core.register.CtrlRegstDesc) type_;
        }
        return org.oneflow.core.register.CtrlRegstDesc.getDefaultInstance();
      } else {
        if (typeCase_ == 3) {
          return ctrlRegstDescBuilder_.getMessage();
        }
        return org.oneflow.core.register.CtrlRegstDesc.getDefaultInstance();
      }
    }
    /**
     * <code>optional .oneflow.CtrlRegstDesc ctrl_regst_desc = 3;</code>
     */
    public Builder setCtrlRegstDesc(org.oneflow.core.register.CtrlRegstDesc value) {
      if (ctrlRegstDescBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        type_ = value;
        onChanged();
      } else {
        ctrlRegstDescBuilder_.setMessage(value);
      }
      typeCase_ = 3;
      return this;
    }
    /**
     * <code>optional .oneflow.CtrlRegstDesc ctrl_regst_desc = 3;</code>
     */
    public Builder setCtrlRegstDesc(
        org.oneflow.core.register.CtrlRegstDesc.Builder builderForValue) {
      if (ctrlRegstDescBuilder_ == null) {
        type_ = builderForValue.build();
        onChanged();
      } else {
        ctrlRegstDescBuilder_.setMessage(builderForValue.build());
      }
      typeCase_ = 3;
      return this;
    }
    /**
     * <code>optional .oneflow.CtrlRegstDesc ctrl_regst_desc = 3;</code>
     */
    public Builder mergeCtrlRegstDesc(org.oneflow.core.register.CtrlRegstDesc value) {
      if (ctrlRegstDescBuilder_ == null) {
        if (typeCase_ == 3 &&
            type_ != org.oneflow.core.register.CtrlRegstDesc.getDefaultInstance()) {
          type_ = org.oneflow.core.register.CtrlRegstDesc.newBuilder((org.oneflow.core.register.CtrlRegstDesc) type_)
              .mergeFrom(value).buildPartial();
        } else {
          type_ = value;
        }
        onChanged();
      } else {
        if (typeCase_ == 3) {
          ctrlRegstDescBuilder_.mergeFrom(value);
        }
        ctrlRegstDescBuilder_.setMessage(value);
      }
      typeCase_ = 3;
      return this;
    }
    /**
     * <code>optional .oneflow.CtrlRegstDesc ctrl_regst_desc = 3;</code>
     */
    public Builder clearCtrlRegstDesc() {
      if (ctrlRegstDescBuilder_ == null) {
        if (typeCase_ == 3) {
          typeCase_ = 0;
          type_ = null;
          onChanged();
        }
      } else {
        if (typeCase_ == 3) {
          typeCase_ = 0;
          type_ = null;
        }
        ctrlRegstDescBuilder_.clear();
      }
      return this;
    }
    /**
     * <code>optional .oneflow.CtrlRegstDesc ctrl_regst_desc = 3;</code>
     */
    public org.oneflow.core.register.CtrlRegstDesc.Builder getCtrlRegstDescBuilder() {
      return getCtrlRegstDescFieldBuilder().getBuilder();
    }
    /**
     * <code>optional .oneflow.CtrlRegstDesc ctrl_regst_desc = 3;</code>
     */
    public org.oneflow.core.register.CtrlRegstDescOrBuilder getCtrlRegstDescOrBuilder() {
      if ((typeCase_ == 3) && (ctrlRegstDescBuilder_ != null)) {
        return ctrlRegstDescBuilder_.getMessageOrBuilder();
      } else {
        if (typeCase_ == 3) {
          return (org.oneflow.core.register.CtrlRegstDesc) type_;
        }
        return org.oneflow.core.register.CtrlRegstDesc.getDefaultInstance();
      }
    }
    /**
     * <code>optional .oneflow.CtrlRegstDesc ctrl_regst_desc = 3;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.register.CtrlRegstDesc, org.oneflow.core.register.CtrlRegstDesc.Builder, org.oneflow.core.register.CtrlRegstDescOrBuilder> 
        getCtrlRegstDescFieldBuilder() {
      if (ctrlRegstDescBuilder_ == null) {
        if (!(typeCase_ == 3)) {
          type_ = org.oneflow.core.register.CtrlRegstDesc.getDefaultInstance();
        }
        ctrlRegstDescBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.oneflow.core.register.CtrlRegstDesc, org.oneflow.core.register.CtrlRegstDesc.Builder, org.oneflow.core.register.CtrlRegstDescOrBuilder>(
                (org.oneflow.core.register.CtrlRegstDesc) type_,
                getParentForChildren(),
                isClean());
        type_ = null;
      }
      typeCase_ = 3;
      onChanged();;
      return ctrlRegstDescBuilder_;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:oneflow.RegstDescTypeProto)
  }

  // @@protoc_insertion_point(class_scope:oneflow.RegstDescTypeProto)
  private static final org.oneflow.core.register.RegstDescTypeProto DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.register.RegstDescTypeProto();
  }

  public static org.oneflow.core.register.RegstDescTypeProto getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<RegstDescTypeProto>
      PARSER = new com.google.protobuf.AbstractParser<RegstDescTypeProto>() {
    public RegstDescTypeProto parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new RegstDescTypeProto(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<RegstDescTypeProto> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<RegstDescTypeProto> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.register.RegstDescTypeProto getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

