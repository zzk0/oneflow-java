// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/regularizer_conf.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.RegularizerConf}
 */
public  final class RegularizerConf extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.RegularizerConf)
    RegularizerConfOrBuilder {
  // Use RegularizerConf.newBuilder() to construct.
  private RegularizerConf(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private RegularizerConf() {
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private RegularizerConf(
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
            org.oneflow.core.job.L1L2RegularizerConf.Builder subBuilder = null;
            if (typeCase_ == 1) {
              subBuilder = ((org.oneflow.core.job.L1L2RegularizerConf) type_).toBuilder();
            }
            type_ =
                input.readMessage(org.oneflow.core.job.L1L2RegularizerConf.PARSER, extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom((org.oneflow.core.job.L1L2RegularizerConf) type_);
              type_ = subBuilder.buildPartial();
            }
            typeCase_ = 1;
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
    return org.oneflow.core.job.RegularizerConfOuterClass.internal_static_oneflow_RegularizerConf_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.RegularizerConfOuterClass.internal_static_oneflow_RegularizerConf_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.RegularizerConf.class, org.oneflow.core.job.RegularizerConf.Builder.class);
  }

  private int bitField0_;
  private int typeCase_ = 0;
  private java.lang.Object type_;
  public enum TypeCase
      implements com.google.protobuf.Internal.EnumLite {
    L1_L2_CONF(1),
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
        case 1: return L1_L2_CONF;
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

  public static final int L1_L2_CONF_FIELD_NUMBER = 1;
  /**
   * <code>optional .oneflow.L1L2RegularizerConf l1_l2_conf = 1;</code>
   */
  public boolean hasL1L2Conf() {
    return typeCase_ == 1;
  }
  /**
   * <code>optional .oneflow.L1L2RegularizerConf l1_l2_conf = 1;</code>
   */
  public org.oneflow.core.job.L1L2RegularizerConf getL1L2Conf() {
    if (typeCase_ == 1) {
       return (org.oneflow.core.job.L1L2RegularizerConf) type_;
    }
    return org.oneflow.core.job.L1L2RegularizerConf.getDefaultInstance();
  }
  /**
   * <code>optional .oneflow.L1L2RegularizerConf l1_l2_conf = 1;</code>
   */
  public org.oneflow.core.job.L1L2RegularizerConfOrBuilder getL1L2ConfOrBuilder() {
    if (typeCase_ == 1) {
       return (org.oneflow.core.job.L1L2RegularizerConf) type_;
    }
    return org.oneflow.core.job.L1L2RegularizerConf.getDefaultInstance();
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
    if (typeCase_ == 1) {
      output.writeMessage(1, (org.oneflow.core.job.L1L2RegularizerConf) type_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (typeCase_ == 1) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(1, (org.oneflow.core.job.L1L2RegularizerConf) type_);
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
    if (!(obj instanceof org.oneflow.core.job.RegularizerConf)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.RegularizerConf other = (org.oneflow.core.job.RegularizerConf) obj;

    boolean result = true;
    result = result && getTypeCase().equals(
        other.getTypeCase());
    if (!result) return false;
    switch (typeCase_) {
      case 1:
        result = result && getL1L2Conf()
            .equals(other.getL1L2Conf());
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
        hash = (37 * hash) + L1_L2_CONF_FIELD_NUMBER;
        hash = (53 * hash) + getL1L2Conf().hashCode();
        break;
      case 0:
      default:
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.RegularizerConf parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.RegularizerConf parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.RegularizerConf parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.RegularizerConf parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.RegularizerConf parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.RegularizerConf parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.RegularizerConf parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.RegularizerConf parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.RegularizerConf parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.RegularizerConf parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.RegularizerConf prototype) {
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
   * Protobuf type {@code oneflow.RegularizerConf}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.RegularizerConf)
      org.oneflow.core.job.RegularizerConfOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.RegularizerConfOuterClass.internal_static_oneflow_RegularizerConf_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.RegularizerConfOuterClass.internal_static_oneflow_RegularizerConf_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.RegularizerConf.class, org.oneflow.core.job.RegularizerConf.Builder.class);
    }

    // Construct using org.oneflow.core.job.RegularizerConf.newBuilder()
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
      return org.oneflow.core.job.RegularizerConfOuterClass.internal_static_oneflow_RegularizerConf_descriptor;
    }

    public org.oneflow.core.job.RegularizerConf getDefaultInstanceForType() {
      return org.oneflow.core.job.RegularizerConf.getDefaultInstance();
    }

    public org.oneflow.core.job.RegularizerConf build() {
      org.oneflow.core.job.RegularizerConf result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.RegularizerConf buildPartial() {
      org.oneflow.core.job.RegularizerConf result = new org.oneflow.core.job.RegularizerConf(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (typeCase_ == 1) {
        if (l1L2ConfBuilder_ == null) {
          result.type_ = type_;
        } else {
          result.type_ = l1L2ConfBuilder_.build();
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
      if (other instanceof org.oneflow.core.job.RegularizerConf) {
        return mergeFrom((org.oneflow.core.job.RegularizerConf)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.RegularizerConf other) {
      if (other == org.oneflow.core.job.RegularizerConf.getDefaultInstance()) return this;
      switch (other.getTypeCase()) {
        case L1_L2_CONF: {
          mergeL1L2Conf(other.getL1L2Conf());
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
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.job.RegularizerConf parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.RegularizerConf) e.getUnfinishedMessage();
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
        org.oneflow.core.job.L1L2RegularizerConf, org.oneflow.core.job.L1L2RegularizerConf.Builder, org.oneflow.core.job.L1L2RegularizerConfOrBuilder> l1L2ConfBuilder_;
    /**
     * <code>optional .oneflow.L1L2RegularizerConf l1_l2_conf = 1;</code>
     */
    public boolean hasL1L2Conf() {
      return typeCase_ == 1;
    }
    /**
     * <code>optional .oneflow.L1L2RegularizerConf l1_l2_conf = 1;</code>
     */
    public org.oneflow.core.job.L1L2RegularizerConf getL1L2Conf() {
      if (l1L2ConfBuilder_ == null) {
        if (typeCase_ == 1) {
          return (org.oneflow.core.job.L1L2RegularizerConf) type_;
        }
        return org.oneflow.core.job.L1L2RegularizerConf.getDefaultInstance();
      } else {
        if (typeCase_ == 1) {
          return l1L2ConfBuilder_.getMessage();
        }
        return org.oneflow.core.job.L1L2RegularizerConf.getDefaultInstance();
      }
    }
    /**
     * <code>optional .oneflow.L1L2RegularizerConf l1_l2_conf = 1;</code>
     */
    public Builder setL1L2Conf(org.oneflow.core.job.L1L2RegularizerConf value) {
      if (l1L2ConfBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        type_ = value;
        onChanged();
      } else {
        l1L2ConfBuilder_.setMessage(value);
      }
      typeCase_ = 1;
      return this;
    }
    /**
     * <code>optional .oneflow.L1L2RegularizerConf l1_l2_conf = 1;</code>
     */
    public Builder setL1L2Conf(
        org.oneflow.core.job.L1L2RegularizerConf.Builder builderForValue) {
      if (l1L2ConfBuilder_ == null) {
        type_ = builderForValue.build();
        onChanged();
      } else {
        l1L2ConfBuilder_.setMessage(builderForValue.build());
      }
      typeCase_ = 1;
      return this;
    }
    /**
     * <code>optional .oneflow.L1L2RegularizerConf l1_l2_conf = 1;</code>
     */
    public Builder mergeL1L2Conf(org.oneflow.core.job.L1L2RegularizerConf value) {
      if (l1L2ConfBuilder_ == null) {
        if (typeCase_ == 1 &&
            type_ != org.oneflow.core.job.L1L2RegularizerConf.getDefaultInstance()) {
          type_ = org.oneflow.core.job.L1L2RegularizerConf.newBuilder((org.oneflow.core.job.L1L2RegularizerConf) type_)
              .mergeFrom(value).buildPartial();
        } else {
          type_ = value;
        }
        onChanged();
      } else {
        if (typeCase_ == 1) {
          l1L2ConfBuilder_.mergeFrom(value);
        }
        l1L2ConfBuilder_.setMessage(value);
      }
      typeCase_ = 1;
      return this;
    }
    /**
     * <code>optional .oneflow.L1L2RegularizerConf l1_l2_conf = 1;</code>
     */
    public Builder clearL1L2Conf() {
      if (l1L2ConfBuilder_ == null) {
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
        l1L2ConfBuilder_.clear();
      }
      return this;
    }
    /**
     * <code>optional .oneflow.L1L2RegularizerConf l1_l2_conf = 1;</code>
     */
    public org.oneflow.core.job.L1L2RegularizerConf.Builder getL1L2ConfBuilder() {
      return getL1L2ConfFieldBuilder().getBuilder();
    }
    /**
     * <code>optional .oneflow.L1L2RegularizerConf l1_l2_conf = 1;</code>
     */
    public org.oneflow.core.job.L1L2RegularizerConfOrBuilder getL1L2ConfOrBuilder() {
      if ((typeCase_ == 1) && (l1L2ConfBuilder_ != null)) {
        return l1L2ConfBuilder_.getMessageOrBuilder();
      } else {
        if (typeCase_ == 1) {
          return (org.oneflow.core.job.L1L2RegularizerConf) type_;
        }
        return org.oneflow.core.job.L1L2RegularizerConf.getDefaultInstance();
      }
    }
    /**
     * <code>optional .oneflow.L1L2RegularizerConf l1_l2_conf = 1;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.job.L1L2RegularizerConf, org.oneflow.core.job.L1L2RegularizerConf.Builder, org.oneflow.core.job.L1L2RegularizerConfOrBuilder> 
        getL1L2ConfFieldBuilder() {
      if (l1L2ConfBuilder_ == null) {
        if (!(typeCase_ == 1)) {
          type_ = org.oneflow.core.job.L1L2RegularizerConf.getDefaultInstance();
        }
        l1L2ConfBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.oneflow.core.job.L1L2RegularizerConf, org.oneflow.core.job.L1L2RegularizerConf.Builder, org.oneflow.core.job.L1L2RegularizerConfOrBuilder>(
                (org.oneflow.core.job.L1L2RegularizerConf) type_,
                getParentForChildren(),
                isClean());
        type_ = null;
      }
      typeCase_ = 1;
      onChanged();;
      return l1L2ConfBuilder_;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:oneflow.RegularizerConf)
  }

  // @@protoc_insertion_point(class_scope:oneflow.RegularizerConf)
  private static final org.oneflow.core.job.RegularizerConf DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.RegularizerConf();
  }

  public static org.oneflow.core.job.RegularizerConf getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<RegularizerConf>
      PARSER = new com.google.protobuf.AbstractParser<RegularizerConf>() {
    public RegularizerConf parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new RegularizerConf(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<RegularizerConf> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<RegularizerConf> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.RegularizerConf getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
