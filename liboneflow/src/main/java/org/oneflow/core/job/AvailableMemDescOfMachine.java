// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/available_memory_desc.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.AvailableMemDescOfMachine}
 */
public  final class AvailableMemDescOfMachine extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.AvailableMemDescOfMachine)
    AvailableMemDescOfMachineOrBuilder {
  // Use AvailableMemDescOfMachine.newBuilder() to construct.
  private AvailableMemDescOfMachine(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private AvailableMemDescOfMachine() {
    zoneSize_ = java.util.Collections.emptyList();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private AvailableMemDescOfMachine(
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
              zoneSize_ = new java.util.ArrayList<java.lang.Long>();
              mutable_bitField0_ |= 0x00000001;
            }
            zoneSize_.add(input.readUInt64());
            break;
          }
          case 10: {
            int length = input.readRawVarint32();
            int limit = input.pushLimit(length);
            if (!((mutable_bitField0_ & 0x00000001) == 0x00000001) && input.getBytesUntilLimit() > 0) {
              zoneSize_ = new java.util.ArrayList<java.lang.Long>();
              mutable_bitField0_ |= 0x00000001;
            }
            while (input.getBytesUntilLimit() > 0) {
              zoneSize_.add(input.readUInt64());
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
        zoneSize_ = java.util.Collections.unmodifiableList(zoneSize_);
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.oneflow.core.job.AvailableMemoryDesc.internal_static_oneflow_AvailableMemDescOfMachine_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.AvailableMemoryDesc.internal_static_oneflow_AvailableMemDescOfMachine_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.AvailableMemDescOfMachine.class, org.oneflow.core.job.AvailableMemDescOfMachine.Builder.class);
  }

  public static final int ZONE_SIZE_FIELD_NUMBER = 1;
  private java.util.List<java.lang.Long> zoneSize_;
  /**
   * <code>repeated uint64 zone_size = 1;</code>
   */
  public java.util.List<java.lang.Long>
      getZoneSizeList() {
    return zoneSize_;
  }
  /**
   * <code>repeated uint64 zone_size = 1;</code>
   */
  public int getZoneSizeCount() {
    return zoneSize_.size();
  }
  /**
   * <code>repeated uint64 zone_size = 1;</code>
   */
  public long getZoneSize(int index) {
    return zoneSize_.get(index);
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
    for (int i = 0; i < zoneSize_.size(); i++) {
      output.writeUInt64(1, zoneSize_.get(i));
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    {
      int dataSize = 0;
      for (int i = 0; i < zoneSize_.size(); i++) {
        dataSize += com.google.protobuf.CodedOutputStream
          .computeUInt64SizeNoTag(zoneSize_.get(i));
      }
      size += dataSize;
      size += 1 * getZoneSizeList().size();
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
    if (!(obj instanceof org.oneflow.core.job.AvailableMemDescOfMachine)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.AvailableMemDescOfMachine other = (org.oneflow.core.job.AvailableMemDescOfMachine) obj;

    boolean result = true;
    result = result && getZoneSizeList()
        .equals(other.getZoneSizeList());
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
    if (getZoneSizeCount() > 0) {
      hash = (37 * hash) + ZONE_SIZE_FIELD_NUMBER;
      hash = (53 * hash) + getZoneSizeList().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.AvailableMemDescOfMachine parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.AvailableMemDescOfMachine parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.AvailableMemDescOfMachine parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.AvailableMemDescOfMachine parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.AvailableMemDescOfMachine parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.AvailableMemDescOfMachine parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.AvailableMemDescOfMachine parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.AvailableMemDescOfMachine parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.AvailableMemDescOfMachine parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.AvailableMemDescOfMachine parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.AvailableMemDescOfMachine prototype) {
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
   * Protobuf type {@code oneflow.AvailableMemDescOfMachine}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.AvailableMemDescOfMachine)
      org.oneflow.core.job.AvailableMemDescOfMachineOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.AvailableMemoryDesc.internal_static_oneflow_AvailableMemDescOfMachine_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.AvailableMemoryDesc.internal_static_oneflow_AvailableMemDescOfMachine_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.AvailableMemDescOfMachine.class, org.oneflow.core.job.AvailableMemDescOfMachine.Builder.class);
    }

    // Construct using org.oneflow.core.job.AvailableMemDescOfMachine.newBuilder()
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
      zoneSize_ = java.util.Collections.emptyList();
      bitField0_ = (bitField0_ & ~0x00000001);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.AvailableMemoryDesc.internal_static_oneflow_AvailableMemDescOfMachine_descriptor;
    }

    public org.oneflow.core.job.AvailableMemDescOfMachine getDefaultInstanceForType() {
      return org.oneflow.core.job.AvailableMemDescOfMachine.getDefaultInstance();
    }

    public org.oneflow.core.job.AvailableMemDescOfMachine build() {
      org.oneflow.core.job.AvailableMemDescOfMachine result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.AvailableMemDescOfMachine buildPartial() {
      org.oneflow.core.job.AvailableMemDescOfMachine result = new org.oneflow.core.job.AvailableMemDescOfMachine(this);
      int from_bitField0_ = bitField0_;
      if (((bitField0_ & 0x00000001) == 0x00000001)) {
        zoneSize_ = java.util.Collections.unmodifiableList(zoneSize_);
        bitField0_ = (bitField0_ & ~0x00000001);
      }
      result.zoneSize_ = zoneSize_;
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
      if (other instanceof org.oneflow.core.job.AvailableMemDescOfMachine) {
        return mergeFrom((org.oneflow.core.job.AvailableMemDescOfMachine)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.AvailableMemDescOfMachine other) {
      if (other == org.oneflow.core.job.AvailableMemDescOfMachine.getDefaultInstance()) return this;
      if (!other.zoneSize_.isEmpty()) {
        if (zoneSize_.isEmpty()) {
          zoneSize_ = other.zoneSize_;
          bitField0_ = (bitField0_ & ~0x00000001);
        } else {
          ensureZoneSizeIsMutable();
          zoneSize_.addAll(other.zoneSize_);
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
      org.oneflow.core.job.AvailableMemDescOfMachine parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.AvailableMemDescOfMachine) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private java.util.List<java.lang.Long> zoneSize_ = java.util.Collections.emptyList();
    private void ensureZoneSizeIsMutable() {
      if (!((bitField0_ & 0x00000001) == 0x00000001)) {
        zoneSize_ = new java.util.ArrayList<java.lang.Long>(zoneSize_);
        bitField0_ |= 0x00000001;
       }
    }
    /**
     * <code>repeated uint64 zone_size = 1;</code>
     */
    public java.util.List<java.lang.Long>
        getZoneSizeList() {
      return java.util.Collections.unmodifiableList(zoneSize_);
    }
    /**
     * <code>repeated uint64 zone_size = 1;</code>
     */
    public int getZoneSizeCount() {
      return zoneSize_.size();
    }
    /**
     * <code>repeated uint64 zone_size = 1;</code>
     */
    public long getZoneSize(int index) {
      return zoneSize_.get(index);
    }
    /**
     * <code>repeated uint64 zone_size = 1;</code>
     */
    public Builder setZoneSize(
        int index, long value) {
      ensureZoneSizeIsMutable();
      zoneSize_.set(index, value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated uint64 zone_size = 1;</code>
     */
    public Builder addZoneSize(long value) {
      ensureZoneSizeIsMutable();
      zoneSize_.add(value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated uint64 zone_size = 1;</code>
     */
    public Builder addAllZoneSize(
        java.lang.Iterable<? extends java.lang.Long> values) {
      ensureZoneSizeIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, zoneSize_);
      onChanged();
      return this;
    }
    /**
     * <code>repeated uint64 zone_size = 1;</code>
     */
    public Builder clearZoneSize() {
      zoneSize_ = java.util.Collections.emptyList();
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


    // @@protoc_insertion_point(builder_scope:oneflow.AvailableMemDescOfMachine)
  }

  // @@protoc_insertion_point(class_scope:oneflow.AvailableMemDescOfMachine)
  private static final org.oneflow.core.job.AvailableMemDescOfMachine DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.AvailableMemDescOfMachine();
  }

  public static org.oneflow.core.job.AvailableMemDescOfMachine getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<AvailableMemDescOfMachine>
      PARSER = new com.google.protobuf.AbstractParser<AvailableMemDescOfMachine>() {
    public AvailableMemDescOfMachine parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new AvailableMemDescOfMachine(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<AvailableMemDescOfMachine> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<AvailableMemDescOfMachine> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.AvailableMemDescOfMachine getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

