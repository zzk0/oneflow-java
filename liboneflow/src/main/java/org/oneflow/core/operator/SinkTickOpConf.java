// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/operator/op_conf.proto

package org.oneflow.core.operator;

/**
 * Protobuf type {@code oneflow.SinkTickOpConf}
 */
public  final class SinkTickOpConf extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.SinkTickOpConf)
    SinkTickOpConfOrBuilder {
  // Use SinkTickOpConf.newBuilder() to construct.
  private SinkTickOpConf(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private SinkTickOpConf() {
    tick_ = com.google.protobuf.LazyStringArrayList.EMPTY;
    out_ = "";
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private SinkTickOpConf(
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
            if (!((mutable_bitField0_ & 0x00000001) == 0x00000001)) {
              tick_ = new com.google.protobuf.LazyStringArrayList();
              mutable_bitField0_ |= 0x00000001;
            }
            tick_.add(bs);
            break;
          }
          case 18: {
            com.google.protobuf.ByteString bs = input.readBytes();
            bitField0_ |= 0x00000001;
            out_ = bs;
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
        tick_ = tick_.getUnmodifiableView();
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.oneflow.core.operator.OpConf.internal_static_oneflow_SinkTickOpConf_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.operator.OpConf.internal_static_oneflow_SinkTickOpConf_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.operator.SinkTickOpConf.class, org.oneflow.core.operator.SinkTickOpConf.Builder.class);
  }

  private int bitField0_;
  public static final int TICK_FIELD_NUMBER = 1;
  private com.google.protobuf.LazyStringList tick_;
  /**
   * <code>repeated string tick = 1;</code>
   */
  public com.google.protobuf.ProtocolStringList
      getTickList() {
    return tick_;
  }
  /**
   * <code>repeated string tick = 1;</code>
   */
  public int getTickCount() {
    return tick_.size();
  }
  /**
   * <code>repeated string tick = 1;</code>
   */
  public java.lang.String getTick(int index) {
    return tick_.get(index);
  }
  /**
   * <code>repeated string tick = 1;</code>
   */
  public com.google.protobuf.ByteString
      getTickBytes(int index) {
    return tick_.getByteString(index);
  }

  public static final int OUT_FIELD_NUMBER = 2;
  private volatile java.lang.Object out_;
  /**
   * <code>required string out = 2;</code>
   */
  public boolean hasOut() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
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

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (!hasOut()) {
      memoizedIsInitialized = 0;
      return false;
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    for (int i = 0; i < tick_.size(); i++) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, tick_.getRaw(i));
    }
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 2, out_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    {
      int dataSize = 0;
      for (int i = 0; i < tick_.size(); i++) {
        dataSize += computeStringSizeNoTag(tick_.getRaw(i));
      }
      size += dataSize;
      size += 1 * getTickList().size();
    }
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(2, out_);
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
    if (!(obj instanceof org.oneflow.core.operator.SinkTickOpConf)) {
      return super.equals(obj);
    }
    org.oneflow.core.operator.SinkTickOpConf other = (org.oneflow.core.operator.SinkTickOpConf) obj;

    boolean result = true;
    result = result && getTickList()
        .equals(other.getTickList());
    result = result && (hasOut() == other.hasOut());
    if (hasOut()) {
      result = result && getOut()
          .equals(other.getOut());
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
    if (getTickCount() > 0) {
      hash = (37 * hash) + TICK_FIELD_NUMBER;
      hash = (53 * hash) + getTickList().hashCode();
    }
    if (hasOut()) {
      hash = (37 * hash) + OUT_FIELD_NUMBER;
      hash = (53 * hash) + getOut().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.operator.SinkTickOpConf parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.operator.SinkTickOpConf parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.operator.SinkTickOpConf parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.operator.SinkTickOpConf parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.operator.SinkTickOpConf parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.operator.SinkTickOpConf parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.operator.SinkTickOpConf parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.operator.SinkTickOpConf parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.operator.SinkTickOpConf parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.operator.SinkTickOpConf parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.operator.SinkTickOpConf prototype) {
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
   * Protobuf type {@code oneflow.SinkTickOpConf}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.SinkTickOpConf)
      org.oneflow.core.operator.SinkTickOpConfOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.operator.OpConf.internal_static_oneflow_SinkTickOpConf_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.operator.OpConf.internal_static_oneflow_SinkTickOpConf_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.operator.SinkTickOpConf.class, org.oneflow.core.operator.SinkTickOpConf.Builder.class);
    }

    // Construct using org.oneflow.core.operator.SinkTickOpConf.newBuilder()
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
      tick_ = com.google.protobuf.LazyStringArrayList.EMPTY;
      bitField0_ = (bitField0_ & ~0x00000001);
      out_ = "";
      bitField0_ = (bitField0_ & ~0x00000002);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.operator.OpConf.internal_static_oneflow_SinkTickOpConf_descriptor;
    }

    public org.oneflow.core.operator.SinkTickOpConf getDefaultInstanceForType() {
      return org.oneflow.core.operator.SinkTickOpConf.getDefaultInstance();
    }

    public org.oneflow.core.operator.SinkTickOpConf build() {
      org.oneflow.core.operator.SinkTickOpConf result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.operator.SinkTickOpConf buildPartial() {
      org.oneflow.core.operator.SinkTickOpConf result = new org.oneflow.core.operator.SinkTickOpConf(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((bitField0_ & 0x00000001) == 0x00000001)) {
        tick_ = tick_.getUnmodifiableView();
        bitField0_ = (bitField0_ & ~0x00000001);
      }
      result.tick_ = tick_;
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000001;
      }
      result.out_ = out_;
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
      if (other instanceof org.oneflow.core.operator.SinkTickOpConf) {
        return mergeFrom((org.oneflow.core.operator.SinkTickOpConf)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.operator.SinkTickOpConf other) {
      if (other == org.oneflow.core.operator.SinkTickOpConf.getDefaultInstance()) return this;
      if (!other.tick_.isEmpty()) {
        if (tick_.isEmpty()) {
          tick_ = other.tick_;
          bitField0_ = (bitField0_ & ~0x00000001);
        } else {
          ensureTickIsMutable();
          tick_.addAll(other.tick_);
        }
        onChanged();
      }
      if (other.hasOut()) {
        bitField0_ |= 0x00000002;
        out_ = other.out_;
        onChanged();
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (!hasOut()) {
        return false;
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.operator.SinkTickOpConf parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.operator.SinkTickOpConf) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private com.google.protobuf.LazyStringList tick_ = com.google.protobuf.LazyStringArrayList.EMPTY;
    private void ensureTickIsMutable() {
      if (!((bitField0_ & 0x00000001) == 0x00000001)) {
        tick_ = new com.google.protobuf.LazyStringArrayList(tick_);
        bitField0_ |= 0x00000001;
       }
    }
    /**
     * <code>repeated string tick = 1;</code>
     */
    public com.google.protobuf.ProtocolStringList
        getTickList() {
      return tick_.getUnmodifiableView();
    }
    /**
     * <code>repeated string tick = 1;</code>
     */
    public int getTickCount() {
      return tick_.size();
    }
    /**
     * <code>repeated string tick = 1;</code>
     */
    public java.lang.String getTick(int index) {
      return tick_.get(index);
    }
    /**
     * <code>repeated string tick = 1;</code>
     */
    public com.google.protobuf.ByteString
        getTickBytes(int index) {
      return tick_.getByteString(index);
    }
    /**
     * <code>repeated string tick = 1;</code>
     */
    public Builder setTick(
        int index, java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureTickIsMutable();
      tick_.set(index, value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string tick = 1;</code>
     */
    public Builder addTick(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureTickIsMutable();
      tick_.add(value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string tick = 1;</code>
     */
    public Builder addAllTick(
        java.lang.Iterable<java.lang.String> values) {
      ensureTickIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, tick_);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string tick = 1;</code>
     */
    public Builder clearTick() {
      tick_ = com.google.protobuf.LazyStringArrayList.EMPTY;
      bitField0_ = (bitField0_ & ~0x00000001);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string tick = 1;</code>
     */
    public Builder addTickBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureTickIsMutable();
      tick_.add(value);
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
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:oneflow.SinkTickOpConf)
  }

  // @@protoc_insertion_point(class_scope:oneflow.SinkTickOpConf)
  private static final org.oneflow.core.operator.SinkTickOpConf DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.operator.SinkTickOpConf();
  }

  public static org.oneflow.core.operator.SinkTickOpConf getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<SinkTickOpConf>
      PARSER = new com.google.protobuf.AbstractParser<SinkTickOpConf>() {
    public SinkTickOpConf parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new SinkTickOpConf(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<SinkTickOpConf> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<SinkTickOpConf> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.operator.SinkTickOpConf getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
