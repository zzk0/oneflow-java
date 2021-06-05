// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/control/control.proto

package org.oneflow.core.control;

/**
 * Protobuf type {@code oneflow.LoadServerRequest}
 */
public  final class LoadServerRequest extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.LoadServerRequest)
    LoadServerRequestOrBuilder {
  // Use LoadServerRequest.newBuilder() to construct.
  private LoadServerRequest(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private LoadServerRequest() {
    addr_ = "";
    rank_ = -1L;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private LoadServerRequest(
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
            addr_ = bs;
            break;
          }
          case 16: {
            bitField0_ |= 0x00000002;
            rank_ = input.readInt64();
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
    return org.oneflow.core.control.Control.internal_static_oneflow_LoadServerRequest_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.control.Control.internal_static_oneflow_LoadServerRequest_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.control.LoadServerRequest.class, org.oneflow.core.control.LoadServerRequest.Builder.class);
  }

  private int bitField0_;
  public static final int ADDR_FIELD_NUMBER = 1;
  private volatile java.lang.Object addr_;
  /**
   * <code>required string addr = 1;</code>
   */
  public boolean hasAddr() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>required string addr = 1;</code>
   */
  public java.lang.String getAddr() {
    java.lang.Object ref = addr_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      if (bs.isValidUtf8()) {
        addr_ = s;
      }
      return s;
    }
  }
  /**
   * <code>required string addr = 1;</code>
   */
  public com.google.protobuf.ByteString
      getAddrBytes() {
    java.lang.Object ref = addr_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      addr_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int RANK_FIELD_NUMBER = 2;
  private long rank_;
  /**
   * <code>optional int64 rank = 2 [default = -1];</code>
   */
  public boolean hasRank() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>optional int64 rank = 2 [default = -1];</code>
   */
  public long getRank() {
    return rank_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (!hasAddr()) {
      memoizedIsInitialized = 0;
      return false;
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, addr_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      output.writeInt64(2, rank_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(1, addr_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt64Size(2, rank_);
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
    if (!(obj instanceof org.oneflow.core.control.LoadServerRequest)) {
      return super.equals(obj);
    }
    org.oneflow.core.control.LoadServerRequest other = (org.oneflow.core.control.LoadServerRequest) obj;

    boolean result = true;
    result = result && (hasAddr() == other.hasAddr());
    if (hasAddr()) {
      result = result && getAddr()
          .equals(other.getAddr());
    }
    result = result && (hasRank() == other.hasRank());
    if (hasRank()) {
      result = result && (getRank()
          == other.getRank());
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
    if (hasAddr()) {
      hash = (37 * hash) + ADDR_FIELD_NUMBER;
      hash = (53 * hash) + getAddr().hashCode();
    }
    if (hasRank()) {
      hash = (37 * hash) + RANK_FIELD_NUMBER;
      hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
          getRank());
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.control.LoadServerRequest parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.control.LoadServerRequest parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.control.LoadServerRequest parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.control.LoadServerRequest parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.control.LoadServerRequest parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.control.LoadServerRequest parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.control.LoadServerRequest parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.control.LoadServerRequest parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.control.LoadServerRequest parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.control.LoadServerRequest parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.control.LoadServerRequest prototype) {
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
   * Protobuf type {@code oneflow.LoadServerRequest}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.LoadServerRequest)
      org.oneflow.core.control.LoadServerRequestOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.control.Control.internal_static_oneflow_LoadServerRequest_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.control.Control.internal_static_oneflow_LoadServerRequest_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.control.LoadServerRequest.class, org.oneflow.core.control.LoadServerRequest.Builder.class);
    }

    // Construct using org.oneflow.core.control.LoadServerRequest.newBuilder()
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
      addr_ = "";
      bitField0_ = (bitField0_ & ~0x00000001);
      rank_ = -1L;
      bitField0_ = (bitField0_ & ~0x00000002);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.control.Control.internal_static_oneflow_LoadServerRequest_descriptor;
    }

    public org.oneflow.core.control.LoadServerRequest getDefaultInstanceForType() {
      return org.oneflow.core.control.LoadServerRequest.getDefaultInstance();
    }

    public org.oneflow.core.control.LoadServerRequest build() {
      org.oneflow.core.control.LoadServerRequest result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.control.LoadServerRequest buildPartial() {
      org.oneflow.core.control.LoadServerRequest result = new org.oneflow.core.control.LoadServerRequest(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      result.addr_ = addr_;
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000002;
      }
      result.rank_ = rank_;
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
      if (other instanceof org.oneflow.core.control.LoadServerRequest) {
        return mergeFrom((org.oneflow.core.control.LoadServerRequest)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.control.LoadServerRequest other) {
      if (other == org.oneflow.core.control.LoadServerRequest.getDefaultInstance()) return this;
      if (other.hasAddr()) {
        bitField0_ |= 0x00000001;
        addr_ = other.addr_;
        onChanged();
      }
      if (other.hasRank()) {
        setRank(other.getRank());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (!hasAddr()) {
        return false;
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.control.LoadServerRequest parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.control.LoadServerRequest) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private java.lang.Object addr_ = "";
    /**
     * <code>required string addr = 1;</code>
     */
    public boolean hasAddr() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>required string addr = 1;</code>
     */
    public java.lang.String getAddr() {
      java.lang.Object ref = addr_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        if (bs.isValidUtf8()) {
          addr_ = s;
        }
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>required string addr = 1;</code>
     */
    public com.google.protobuf.ByteString
        getAddrBytes() {
      java.lang.Object ref = addr_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        addr_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>required string addr = 1;</code>
     */
    public Builder setAddr(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000001;
      addr_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>required string addr = 1;</code>
     */
    public Builder clearAddr() {
      bitField0_ = (bitField0_ & ~0x00000001);
      addr_ = getDefaultInstance().getAddr();
      onChanged();
      return this;
    }
    /**
     * <code>required string addr = 1;</code>
     */
    public Builder setAddrBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000001;
      addr_ = value;
      onChanged();
      return this;
    }

    private long rank_ = -1L;
    /**
     * <code>optional int64 rank = 2 [default = -1];</code>
     */
    public boolean hasRank() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>optional int64 rank = 2 [default = -1];</code>
     */
    public long getRank() {
      return rank_;
    }
    /**
     * <code>optional int64 rank = 2 [default = -1];</code>
     */
    public Builder setRank(long value) {
      bitField0_ |= 0x00000002;
      rank_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional int64 rank = 2 [default = -1];</code>
     */
    public Builder clearRank() {
      bitField0_ = (bitField0_ & ~0x00000002);
      rank_ = -1L;
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


    // @@protoc_insertion_point(builder_scope:oneflow.LoadServerRequest)
  }

  // @@protoc_insertion_point(class_scope:oneflow.LoadServerRequest)
  private static final org.oneflow.core.control.LoadServerRequest DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.control.LoadServerRequest();
  }

  public static org.oneflow.core.control.LoadServerRequest getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<LoadServerRequest>
      PARSER = new com.google.protobuf.AbstractParser<LoadServerRequest>() {
    public LoadServerRequest parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new LoadServerRequest(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<LoadServerRequest> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<LoadServerRequest> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.control.LoadServerRequest getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
