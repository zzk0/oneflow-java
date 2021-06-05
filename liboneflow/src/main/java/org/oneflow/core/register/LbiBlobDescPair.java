// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/register/register_desc.proto

package org.oneflow.core.register;

/**
 * Protobuf type {@code oneflow.LbiBlobDescPair}
 */
public  final class LbiBlobDescPair extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.LbiBlobDescPair)
    LbiBlobDescPairOrBuilder {
  // Use LbiBlobDescPair.newBuilder() to construct.
  private LbiBlobDescPair(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private LbiBlobDescPair() {
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private LbiBlobDescPair(
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
            org.oneflow.core.register.LogicalBlobId.Builder subBuilder = null;
            if (((bitField0_ & 0x00000001) == 0x00000001)) {
              subBuilder = lbi_.toBuilder();
            }
            lbi_ = input.readMessage(org.oneflow.core.register.LogicalBlobId.PARSER, extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(lbi_);
              lbi_ = subBuilder.buildPartial();
            }
            bitField0_ |= 0x00000001;
            break;
          }
          case 18: {
            org.oneflow.core.register.BlobDescProto.Builder subBuilder = null;
            if (((bitField0_ & 0x00000002) == 0x00000002)) {
              subBuilder = blobDesc_.toBuilder();
            }
            blobDesc_ = input.readMessage(org.oneflow.core.register.BlobDescProto.PARSER, extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(blobDesc_);
              blobDesc_ = subBuilder.buildPartial();
            }
            bitField0_ |= 0x00000002;
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
    return org.oneflow.core.register.RegisterDesc.internal_static_oneflow_LbiBlobDescPair_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.register.RegisterDesc.internal_static_oneflow_LbiBlobDescPair_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.register.LbiBlobDescPair.class, org.oneflow.core.register.LbiBlobDescPair.Builder.class);
  }

  private int bitField0_;
  public static final int LBI_FIELD_NUMBER = 1;
  private org.oneflow.core.register.LogicalBlobId lbi_;
  /**
   * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
   */
  public boolean hasLbi() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
   */
  public org.oneflow.core.register.LogicalBlobId getLbi() {
    return lbi_ == null ? org.oneflow.core.register.LogicalBlobId.getDefaultInstance() : lbi_;
  }
  /**
   * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
   */
  public org.oneflow.core.register.LogicalBlobIdOrBuilder getLbiOrBuilder() {
    return lbi_ == null ? org.oneflow.core.register.LogicalBlobId.getDefaultInstance() : lbi_;
  }

  public static final int BLOB_DESC_FIELD_NUMBER = 2;
  private org.oneflow.core.register.BlobDescProto blobDesc_;
  /**
   * <code>required .oneflow.BlobDescProto blob_desc = 2;</code>
   */
  public boolean hasBlobDesc() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>required .oneflow.BlobDescProto blob_desc = 2;</code>
   */
  public org.oneflow.core.register.BlobDescProto getBlobDesc() {
    return blobDesc_ == null ? org.oneflow.core.register.BlobDescProto.getDefaultInstance() : blobDesc_;
  }
  /**
   * <code>required .oneflow.BlobDescProto blob_desc = 2;</code>
   */
  public org.oneflow.core.register.BlobDescProtoOrBuilder getBlobDescOrBuilder() {
    return blobDesc_ == null ? org.oneflow.core.register.BlobDescProto.getDefaultInstance() : blobDesc_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (!hasLbi()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!hasBlobDesc()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!getBlobDesc().isInitialized()) {
      memoizedIsInitialized = 0;
      return false;
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      output.writeMessage(1, getLbi());
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      output.writeMessage(2, getBlobDesc());
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(1, getLbi());
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(2, getBlobDesc());
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
    if (!(obj instanceof org.oneflow.core.register.LbiBlobDescPair)) {
      return super.equals(obj);
    }
    org.oneflow.core.register.LbiBlobDescPair other = (org.oneflow.core.register.LbiBlobDescPair) obj;

    boolean result = true;
    result = result && (hasLbi() == other.hasLbi());
    if (hasLbi()) {
      result = result && getLbi()
          .equals(other.getLbi());
    }
    result = result && (hasBlobDesc() == other.hasBlobDesc());
    if (hasBlobDesc()) {
      result = result && getBlobDesc()
          .equals(other.getBlobDesc());
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
    if (hasLbi()) {
      hash = (37 * hash) + LBI_FIELD_NUMBER;
      hash = (53 * hash) + getLbi().hashCode();
    }
    if (hasBlobDesc()) {
      hash = (37 * hash) + BLOB_DESC_FIELD_NUMBER;
      hash = (53 * hash) + getBlobDesc().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.register.LbiBlobDescPair parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.register.LbiBlobDescPair parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.register.LbiBlobDescPair parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.register.LbiBlobDescPair parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.register.LbiBlobDescPair parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.LbiBlobDescPair parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.register.LbiBlobDescPair parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.LbiBlobDescPair parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.register.LbiBlobDescPair parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.LbiBlobDescPair parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.register.LbiBlobDescPair prototype) {
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
   * Protobuf type {@code oneflow.LbiBlobDescPair}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.LbiBlobDescPair)
      org.oneflow.core.register.LbiBlobDescPairOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.register.RegisterDesc.internal_static_oneflow_LbiBlobDescPair_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.register.RegisterDesc.internal_static_oneflow_LbiBlobDescPair_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.register.LbiBlobDescPair.class, org.oneflow.core.register.LbiBlobDescPair.Builder.class);
    }

    // Construct using org.oneflow.core.register.LbiBlobDescPair.newBuilder()
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
        getLbiFieldBuilder();
        getBlobDescFieldBuilder();
      }
    }
    public Builder clear() {
      super.clear();
      if (lbiBuilder_ == null) {
        lbi_ = null;
      } else {
        lbiBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000001);
      if (blobDescBuilder_ == null) {
        blobDesc_ = null;
      } else {
        blobDescBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000002);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.register.RegisterDesc.internal_static_oneflow_LbiBlobDescPair_descriptor;
    }

    public org.oneflow.core.register.LbiBlobDescPair getDefaultInstanceForType() {
      return org.oneflow.core.register.LbiBlobDescPair.getDefaultInstance();
    }

    public org.oneflow.core.register.LbiBlobDescPair build() {
      org.oneflow.core.register.LbiBlobDescPair result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.register.LbiBlobDescPair buildPartial() {
      org.oneflow.core.register.LbiBlobDescPair result = new org.oneflow.core.register.LbiBlobDescPair(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      if (lbiBuilder_ == null) {
        result.lbi_ = lbi_;
      } else {
        result.lbi_ = lbiBuilder_.build();
      }
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000002;
      }
      if (blobDescBuilder_ == null) {
        result.blobDesc_ = blobDesc_;
      } else {
        result.blobDesc_ = blobDescBuilder_.build();
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
      if (other instanceof org.oneflow.core.register.LbiBlobDescPair) {
        return mergeFrom((org.oneflow.core.register.LbiBlobDescPair)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.register.LbiBlobDescPair other) {
      if (other == org.oneflow.core.register.LbiBlobDescPair.getDefaultInstance()) return this;
      if (other.hasLbi()) {
        mergeLbi(other.getLbi());
      }
      if (other.hasBlobDesc()) {
        mergeBlobDesc(other.getBlobDesc());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (!hasLbi()) {
        return false;
      }
      if (!hasBlobDesc()) {
        return false;
      }
      if (!getBlobDesc().isInitialized()) {
        return false;
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.register.LbiBlobDescPair parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.register.LbiBlobDescPair) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private org.oneflow.core.register.LogicalBlobId lbi_ = null;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.register.LogicalBlobId, org.oneflow.core.register.LogicalBlobId.Builder, org.oneflow.core.register.LogicalBlobIdOrBuilder> lbiBuilder_;
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    public boolean hasLbi() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    public org.oneflow.core.register.LogicalBlobId getLbi() {
      if (lbiBuilder_ == null) {
        return lbi_ == null ? org.oneflow.core.register.LogicalBlobId.getDefaultInstance() : lbi_;
      } else {
        return lbiBuilder_.getMessage();
      }
    }
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    public Builder setLbi(org.oneflow.core.register.LogicalBlobId value) {
      if (lbiBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        lbi_ = value;
        onChanged();
      } else {
        lbiBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    public Builder setLbi(
        org.oneflow.core.register.LogicalBlobId.Builder builderForValue) {
      if (lbiBuilder_ == null) {
        lbi_ = builderForValue.build();
        onChanged();
      } else {
        lbiBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    public Builder mergeLbi(org.oneflow.core.register.LogicalBlobId value) {
      if (lbiBuilder_ == null) {
        if (((bitField0_ & 0x00000001) == 0x00000001) &&
            lbi_ != null &&
            lbi_ != org.oneflow.core.register.LogicalBlobId.getDefaultInstance()) {
          lbi_ =
            org.oneflow.core.register.LogicalBlobId.newBuilder(lbi_).mergeFrom(value).buildPartial();
        } else {
          lbi_ = value;
        }
        onChanged();
      } else {
        lbiBuilder_.mergeFrom(value);
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    public Builder clearLbi() {
      if (lbiBuilder_ == null) {
        lbi_ = null;
        onChanged();
      } else {
        lbiBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000001);
      return this;
    }
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    public org.oneflow.core.register.LogicalBlobId.Builder getLbiBuilder() {
      bitField0_ |= 0x00000001;
      onChanged();
      return getLbiFieldBuilder().getBuilder();
    }
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    public org.oneflow.core.register.LogicalBlobIdOrBuilder getLbiOrBuilder() {
      if (lbiBuilder_ != null) {
        return lbiBuilder_.getMessageOrBuilder();
      } else {
        return lbi_ == null ?
            org.oneflow.core.register.LogicalBlobId.getDefaultInstance() : lbi_;
      }
    }
    /**
     * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.register.LogicalBlobId, org.oneflow.core.register.LogicalBlobId.Builder, org.oneflow.core.register.LogicalBlobIdOrBuilder> 
        getLbiFieldBuilder() {
      if (lbiBuilder_ == null) {
        lbiBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.oneflow.core.register.LogicalBlobId, org.oneflow.core.register.LogicalBlobId.Builder, org.oneflow.core.register.LogicalBlobIdOrBuilder>(
                getLbi(),
                getParentForChildren(),
                isClean());
        lbi_ = null;
      }
      return lbiBuilder_;
    }

    private org.oneflow.core.register.BlobDescProto blobDesc_ = null;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.register.BlobDescProto, org.oneflow.core.register.BlobDescProto.Builder, org.oneflow.core.register.BlobDescProtoOrBuilder> blobDescBuilder_;
    /**
     * <code>required .oneflow.BlobDescProto blob_desc = 2;</code>
     */
    public boolean hasBlobDesc() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>required .oneflow.BlobDescProto blob_desc = 2;</code>
     */
    public org.oneflow.core.register.BlobDescProto getBlobDesc() {
      if (blobDescBuilder_ == null) {
        return blobDesc_ == null ? org.oneflow.core.register.BlobDescProto.getDefaultInstance() : blobDesc_;
      } else {
        return blobDescBuilder_.getMessage();
      }
    }
    /**
     * <code>required .oneflow.BlobDescProto blob_desc = 2;</code>
     */
    public Builder setBlobDesc(org.oneflow.core.register.BlobDescProto value) {
      if (blobDescBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        blobDesc_ = value;
        onChanged();
      } else {
        blobDescBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000002;
      return this;
    }
    /**
     * <code>required .oneflow.BlobDescProto blob_desc = 2;</code>
     */
    public Builder setBlobDesc(
        org.oneflow.core.register.BlobDescProto.Builder builderForValue) {
      if (blobDescBuilder_ == null) {
        blobDesc_ = builderForValue.build();
        onChanged();
      } else {
        blobDescBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000002;
      return this;
    }
    /**
     * <code>required .oneflow.BlobDescProto blob_desc = 2;</code>
     */
    public Builder mergeBlobDesc(org.oneflow.core.register.BlobDescProto value) {
      if (blobDescBuilder_ == null) {
        if (((bitField0_ & 0x00000002) == 0x00000002) &&
            blobDesc_ != null &&
            blobDesc_ != org.oneflow.core.register.BlobDescProto.getDefaultInstance()) {
          blobDesc_ =
            org.oneflow.core.register.BlobDescProto.newBuilder(blobDesc_).mergeFrom(value).buildPartial();
        } else {
          blobDesc_ = value;
        }
        onChanged();
      } else {
        blobDescBuilder_.mergeFrom(value);
      }
      bitField0_ |= 0x00000002;
      return this;
    }
    /**
     * <code>required .oneflow.BlobDescProto blob_desc = 2;</code>
     */
    public Builder clearBlobDesc() {
      if (blobDescBuilder_ == null) {
        blobDesc_ = null;
        onChanged();
      } else {
        blobDescBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000002);
      return this;
    }
    /**
     * <code>required .oneflow.BlobDescProto blob_desc = 2;</code>
     */
    public org.oneflow.core.register.BlobDescProto.Builder getBlobDescBuilder() {
      bitField0_ |= 0x00000002;
      onChanged();
      return getBlobDescFieldBuilder().getBuilder();
    }
    /**
     * <code>required .oneflow.BlobDescProto blob_desc = 2;</code>
     */
    public org.oneflow.core.register.BlobDescProtoOrBuilder getBlobDescOrBuilder() {
      if (blobDescBuilder_ != null) {
        return blobDescBuilder_.getMessageOrBuilder();
      } else {
        return blobDesc_ == null ?
            org.oneflow.core.register.BlobDescProto.getDefaultInstance() : blobDesc_;
      }
    }
    /**
     * <code>required .oneflow.BlobDescProto blob_desc = 2;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.register.BlobDescProto, org.oneflow.core.register.BlobDescProto.Builder, org.oneflow.core.register.BlobDescProtoOrBuilder> 
        getBlobDescFieldBuilder() {
      if (blobDescBuilder_ == null) {
        blobDescBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.oneflow.core.register.BlobDescProto, org.oneflow.core.register.BlobDescProto.Builder, org.oneflow.core.register.BlobDescProtoOrBuilder>(
                getBlobDesc(),
                getParentForChildren(),
                isClean());
        blobDesc_ = null;
      }
      return blobDescBuilder_;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:oneflow.LbiBlobDescPair)
  }

  // @@protoc_insertion_point(class_scope:oneflow.LbiBlobDescPair)
  private static final org.oneflow.core.register.LbiBlobDescPair DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.register.LbiBlobDescPair();
  }

  public static org.oneflow.core.register.LbiBlobDescPair getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<LbiBlobDescPair>
      PARSER = new com.google.protobuf.AbstractParser<LbiBlobDescPair>() {
    public LbiBlobDescPair parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new LbiBlobDescPair(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<LbiBlobDescPair> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<LbiBlobDescPair> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.register.LbiBlobDescPair getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

