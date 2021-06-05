// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/operator/op_conf.proto

package org.oneflow.core.operator;

/**
 * Protobuf type {@code oneflow.DistributeCloneOpConf}
 */
public  final class DistributeCloneOpConf extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.DistributeCloneOpConf)
    DistributeCloneOpConfOrBuilder {
  // Use DistributeCloneOpConf.newBuilder() to construct.
  private DistributeCloneOpConf(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private DistributeCloneOpConf() {
    in_ = "";
    out_ = com.google.protobuf.LazyStringArrayList.EMPTY;
    isVariableRef_ = false;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private DistributeCloneOpConf(
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
            in_ = bs;
            break;
          }
          case 18: {
            com.google.protobuf.ByteString bs = input.readBytes();
            if (!((mutable_bitField0_ & 0x00000002) == 0x00000002)) {
              out_ = new com.google.protobuf.LazyStringArrayList();
              mutable_bitField0_ |= 0x00000002;
            }
            out_.add(bs);
            break;
          }
          case 24: {
            bitField0_ |= 0x00000002;
            isVariableRef_ = input.readBool();
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
        out_ = out_.getUnmodifiableView();
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.oneflow.core.operator.OpConf.internal_static_oneflow_DistributeCloneOpConf_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.operator.OpConf.internal_static_oneflow_DistributeCloneOpConf_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.operator.DistributeCloneOpConf.class, org.oneflow.core.operator.DistributeCloneOpConf.Builder.class);
  }

  private int bitField0_;
  public static final int IN_FIELD_NUMBER = 1;
  private volatile java.lang.Object in_;
  /**
   * <code>required string in = 1;</code>
   */
  public boolean hasIn() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>required string in = 1;</code>
   */
  public java.lang.String getIn() {
    java.lang.Object ref = in_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      if (bs.isValidUtf8()) {
        in_ = s;
      }
      return s;
    }
  }
  /**
   * <code>required string in = 1;</code>
   */
  public com.google.protobuf.ByteString
      getInBytes() {
    java.lang.Object ref = in_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      in_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int OUT_FIELD_NUMBER = 2;
  private com.google.protobuf.LazyStringList out_;
  /**
   * <code>repeated string out = 2;</code>
   */
  public com.google.protobuf.ProtocolStringList
      getOutList() {
    return out_;
  }
  /**
   * <code>repeated string out = 2;</code>
   */
  public int getOutCount() {
    return out_.size();
  }
  /**
   * <code>repeated string out = 2;</code>
   */
  public java.lang.String getOut(int index) {
    return out_.get(index);
  }
  /**
   * <code>repeated string out = 2;</code>
   */
  public com.google.protobuf.ByteString
      getOutBytes(int index) {
    return out_.getByteString(index);
  }

  public static final int IS_VARIABLE_REF_FIELD_NUMBER = 3;
  private boolean isVariableRef_;
  /**
   * <code>optional bool is_variable_ref = 3 [default = false];</code>
   */
  public boolean hasIsVariableRef() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>optional bool is_variable_ref = 3 [default = false];</code>
   */
  public boolean getIsVariableRef() {
    return isVariableRef_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (!hasIn()) {
      memoizedIsInitialized = 0;
      return false;
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, in_);
    }
    for (int i = 0; i < out_.size(); i++) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 2, out_.getRaw(i));
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      output.writeBool(3, isVariableRef_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(1, in_);
    }
    {
      int dataSize = 0;
      for (int i = 0; i < out_.size(); i++) {
        dataSize += computeStringSizeNoTag(out_.getRaw(i));
      }
      size += dataSize;
      size += 1 * getOutList().size();
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.CodedOutputStream
        .computeBoolSize(3, isVariableRef_);
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
    if (!(obj instanceof org.oneflow.core.operator.DistributeCloneOpConf)) {
      return super.equals(obj);
    }
    org.oneflow.core.operator.DistributeCloneOpConf other = (org.oneflow.core.operator.DistributeCloneOpConf) obj;

    boolean result = true;
    result = result && (hasIn() == other.hasIn());
    if (hasIn()) {
      result = result && getIn()
          .equals(other.getIn());
    }
    result = result && getOutList()
        .equals(other.getOutList());
    result = result && (hasIsVariableRef() == other.hasIsVariableRef());
    if (hasIsVariableRef()) {
      result = result && (getIsVariableRef()
          == other.getIsVariableRef());
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
    if (hasIn()) {
      hash = (37 * hash) + IN_FIELD_NUMBER;
      hash = (53 * hash) + getIn().hashCode();
    }
    if (getOutCount() > 0) {
      hash = (37 * hash) + OUT_FIELD_NUMBER;
      hash = (53 * hash) + getOutList().hashCode();
    }
    if (hasIsVariableRef()) {
      hash = (37 * hash) + IS_VARIABLE_REF_FIELD_NUMBER;
      hash = (53 * hash) + com.google.protobuf.Internal.hashBoolean(
          getIsVariableRef());
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.operator.DistributeCloneOpConf parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.operator.DistributeCloneOpConf parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.operator.DistributeCloneOpConf parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.operator.DistributeCloneOpConf parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.operator.DistributeCloneOpConf parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.operator.DistributeCloneOpConf parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.operator.DistributeCloneOpConf parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.operator.DistributeCloneOpConf parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.operator.DistributeCloneOpConf parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.operator.DistributeCloneOpConf parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.operator.DistributeCloneOpConf prototype) {
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
   * Protobuf type {@code oneflow.DistributeCloneOpConf}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.DistributeCloneOpConf)
      org.oneflow.core.operator.DistributeCloneOpConfOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.operator.OpConf.internal_static_oneflow_DistributeCloneOpConf_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.operator.OpConf.internal_static_oneflow_DistributeCloneOpConf_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.operator.DistributeCloneOpConf.class, org.oneflow.core.operator.DistributeCloneOpConf.Builder.class);
    }

    // Construct using org.oneflow.core.operator.DistributeCloneOpConf.newBuilder()
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
      in_ = "";
      bitField0_ = (bitField0_ & ~0x00000001);
      out_ = com.google.protobuf.LazyStringArrayList.EMPTY;
      bitField0_ = (bitField0_ & ~0x00000002);
      isVariableRef_ = false;
      bitField0_ = (bitField0_ & ~0x00000004);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.operator.OpConf.internal_static_oneflow_DistributeCloneOpConf_descriptor;
    }

    public org.oneflow.core.operator.DistributeCloneOpConf getDefaultInstanceForType() {
      return org.oneflow.core.operator.DistributeCloneOpConf.getDefaultInstance();
    }

    public org.oneflow.core.operator.DistributeCloneOpConf build() {
      org.oneflow.core.operator.DistributeCloneOpConf result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.operator.DistributeCloneOpConf buildPartial() {
      org.oneflow.core.operator.DistributeCloneOpConf result = new org.oneflow.core.operator.DistributeCloneOpConf(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      result.in_ = in_;
      if (((bitField0_ & 0x00000002) == 0x00000002)) {
        out_ = out_.getUnmodifiableView();
        bitField0_ = (bitField0_ & ~0x00000002);
      }
      result.out_ = out_;
      if (((from_bitField0_ & 0x00000004) == 0x00000004)) {
        to_bitField0_ |= 0x00000002;
      }
      result.isVariableRef_ = isVariableRef_;
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
      if (other instanceof org.oneflow.core.operator.DistributeCloneOpConf) {
        return mergeFrom((org.oneflow.core.operator.DistributeCloneOpConf)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.operator.DistributeCloneOpConf other) {
      if (other == org.oneflow.core.operator.DistributeCloneOpConf.getDefaultInstance()) return this;
      if (other.hasIn()) {
        bitField0_ |= 0x00000001;
        in_ = other.in_;
        onChanged();
      }
      if (!other.out_.isEmpty()) {
        if (out_.isEmpty()) {
          out_ = other.out_;
          bitField0_ = (bitField0_ & ~0x00000002);
        } else {
          ensureOutIsMutable();
          out_.addAll(other.out_);
        }
        onChanged();
      }
      if (other.hasIsVariableRef()) {
        setIsVariableRef(other.getIsVariableRef());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (!hasIn()) {
        return false;
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.operator.DistributeCloneOpConf parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.operator.DistributeCloneOpConf) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private java.lang.Object in_ = "";
    /**
     * <code>required string in = 1;</code>
     */
    public boolean hasIn() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>required string in = 1;</code>
     */
    public java.lang.String getIn() {
      java.lang.Object ref = in_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        if (bs.isValidUtf8()) {
          in_ = s;
        }
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>required string in = 1;</code>
     */
    public com.google.protobuf.ByteString
        getInBytes() {
      java.lang.Object ref = in_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        in_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>required string in = 1;</code>
     */
    public Builder setIn(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000001;
      in_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>required string in = 1;</code>
     */
    public Builder clearIn() {
      bitField0_ = (bitField0_ & ~0x00000001);
      in_ = getDefaultInstance().getIn();
      onChanged();
      return this;
    }
    /**
     * <code>required string in = 1;</code>
     */
    public Builder setInBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000001;
      in_ = value;
      onChanged();
      return this;
    }

    private com.google.protobuf.LazyStringList out_ = com.google.protobuf.LazyStringArrayList.EMPTY;
    private void ensureOutIsMutable() {
      if (!((bitField0_ & 0x00000002) == 0x00000002)) {
        out_ = new com.google.protobuf.LazyStringArrayList(out_);
        bitField0_ |= 0x00000002;
       }
    }
    /**
     * <code>repeated string out = 2;</code>
     */
    public com.google.protobuf.ProtocolStringList
        getOutList() {
      return out_.getUnmodifiableView();
    }
    /**
     * <code>repeated string out = 2;</code>
     */
    public int getOutCount() {
      return out_.size();
    }
    /**
     * <code>repeated string out = 2;</code>
     */
    public java.lang.String getOut(int index) {
      return out_.get(index);
    }
    /**
     * <code>repeated string out = 2;</code>
     */
    public com.google.protobuf.ByteString
        getOutBytes(int index) {
      return out_.getByteString(index);
    }
    /**
     * <code>repeated string out = 2;</code>
     */
    public Builder setOut(
        int index, java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureOutIsMutable();
      out_.set(index, value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string out = 2;</code>
     */
    public Builder addOut(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureOutIsMutable();
      out_.add(value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string out = 2;</code>
     */
    public Builder addAllOut(
        java.lang.Iterable<java.lang.String> values) {
      ensureOutIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, out_);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string out = 2;</code>
     */
    public Builder clearOut() {
      out_ = com.google.protobuf.LazyStringArrayList.EMPTY;
      bitField0_ = (bitField0_ & ~0x00000002);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string out = 2;</code>
     */
    public Builder addOutBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureOutIsMutable();
      out_.add(value);
      onChanged();
      return this;
    }

    private boolean isVariableRef_ ;
    /**
     * <code>optional bool is_variable_ref = 3 [default = false];</code>
     */
    public boolean hasIsVariableRef() {
      return ((bitField0_ & 0x00000004) == 0x00000004);
    }
    /**
     * <code>optional bool is_variable_ref = 3 [default = false];</code>
     */
    public boolean getIsVariableRef() {
      return isVariableRef_;
    }
    /**
     * <code>optional bool is_variable_ref = 3 [default = false];</code>
     */
    public Builder setIsVariableRef(boolean value) {
      bitField0_ |= 0x00000004;
      isVariableRef_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional bool is_variable_ref = 3 [default = false];</code>
     */
    public Builder clearIsVariableRef() {
      bitField0_ = (bitField0_ & ~0x00000004);
      isVariableRef_ = false;
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


    // @@protoc_insertion_point(builder_scope:oneflow.DistributeCloneOpConf)
  }

  // @@protoc_insertion_point(class_scope:oneflow.DistributeCloneOpConf)
  private static final org.oneflow.core.operator.DistributeCloneOpConf DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.operator.DistributeCloneOpConf();
  }

  public static org.oneflow.core.operator.DistributeCloneOpConf getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<DistributeCloneOpConf>
      PARSER = new com.google.protobuf.AbstractParser<DistributeCloneOpConf>() {
    public DistributeCloneOpConf parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new DistributeCloneOpConf(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<DistributeCloneOpConf> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<DistributeCloneOpConf> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.operator.DistributeCloneOpConf getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
