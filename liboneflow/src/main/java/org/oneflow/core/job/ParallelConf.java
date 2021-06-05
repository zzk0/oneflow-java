// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/placement.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.ParallelConf}
 */
public  final class ParallelConf extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.ParallelConf)
    ParallelConfOrBuilder {
  // Use ParallelConf.newBuilder() to construct.
  private ParallelConf(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private ParallelConf() {
    deviceName_ = com.google.protobuf.LazyStringArrayList.EMPTY;
    deviceTag_ = "";
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private ParallelConf(
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
              deviceName_ = new com.google.protobuf.LazyStringArrayList();
              mutable_bitField0_ |= 0x00000001;
            }
            deviceName_.add(bs);
            break;
          }
          case 18: {
            com.google.protobuf.ByteString bs = input.readBytes();
            bitField0_ |= 0x00000001;
            deviceTag_ = bs;
            break;
          }
          case 26: {
            org.oneflow.core.common.ShapeProto.Builder subBuilder = null;
            if (((bitField0_ & 0x00000002) == 0x00000002)) {
              subBuilder = hierarchy_.toBuilder();
            }
            hierarchy_ = input.readMessage(org.oneflow.core.common.ShapeProto.PARSER, extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(hierarchy_);
              hierarchy_ = subBuilder.buildPartial();
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
      if (((mutable_bitField0_ & 0x00000001) == 0x00000001)) {
        deviceName_ = deviceName_.getUnmodifiableView();
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.oneflow.core.job.PlacementOuterClass.internal_static_oneflow_ParallelConf_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.PlacementOuterClass.internal_static_oneflow_ParallelConf_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.ParallelConf.class, org.oneflow.core.job.ParallelConf.Builder.class);
  }

  private int bitField0_;
  public static final int DEVICE_NAME_FIELD_NUMBER = 1;
  private com.google.protobuf.LazyStringList deviceName_;
  /**
   * <code>repeated string device_name = 1;</code>
   */
  public com.google.protobuf.ProtocolStringList
      getDeviceNameList() {
    return deviceName_;
  }
  /**
   * <code>repeated string device_name = 1;</code>
   */
  public int getDeviceNameCount() {
    return deviceName_.size();
  }
  /**
   * <code>repeated string device_name = 1;</code>
   */
  public java.lang.String getDeviceName(int index) {
    return deviceName_.get(index);
  }
  /**
   * <code>repeated string device_name = 1;</code>
   */
  public com.google.protobuf.ByteString
      getDeviceNameBytes(int index) {
    return deviceName_.getByteString(index);
  }

  public static final int DEVICE_TAG_FIELD_NUMBER = 2;
  private volatile java.lang.Object deviceTag_;
  /**
   * <code>required string device_tag = 2;</code>
   */
  public boolean hasDeviceTag() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>required string device_tag = 2;</code>
   */
  public java.lang.String getDeviceTag() {
    java.lang.Object ref = deviceTag_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      if (bs.isValidUtf8()) {
        deviceTag_ = s;
      }
      return s;
    }
  }
  /**
   * <code>required string device_tag = 2;</code>
   */
  public com.google.protobuf.ByteString
      getDeviceTagBytes() {
    java.lang.Object ref = deviceTag_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      deviceTag_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int HIERARCHY_FIELD_NUMBER = 3;
  private org.oneflow.core.common.ShapeProto hierarchy_;
  /**
   * <code>optional .oneflow.ShapeProto hierarchy = 3;</code>
   */
  public boolean hasHierarchy() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>optional .oneflow.ShapeProto hierarchy = 3;</code>
   */
  public org.oneflow.core.common.ShapeProto getHierarchy() {
    return hierarchy_ == null ? org.oneflow.core.common.ShapeProto.getDefaultInstance() : hierarchy_;
  }
  /**
   * <code>optional .oneflow.ShapeProto hierarchy = 3;</code>
   */
  public org.oneflow.core.common.ShapeProtoOrBuilder getHierarchyOrBuilder() {
    return hierarchy_ == null ? org.oneflow.core.common.ShapeProto.getDefaultInstance() : hierarchy_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (!hasDeviceTag()) {
      memoizedIsInitialized = 0;
      return false;
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    for (int i = 0; i < deviceName_.size(); i++) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, deviceName_.getRaw(i));
    }
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 2, deviceTag_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      output.writeMessage(3, getHierarchy());
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    {
      int dataSize = 0;
      for (int i = 0; i < deviceName_.size(); i++) {
        dataSize += computeStringSizeNoTag(deviceName_.getRaw(i));
      }
      size += dataSize;
      size += 1 * getDeviceNameList().size();
    }
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(2, deviceTag_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(3, getHierarchy());
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
    if (!(obj instanceof org.oneflow.core.job.ParallelConf)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.ParallelConf other = (org.oneflow.core.job.ParallelConf) obj;

    boolean result = true;
    result = result && getDeviceNameList()
        .equals(other.getDeviceNameList());
    result = result && (hasDeviceTag() == other.hasDeviceTag());
    if (hasDeviceTag()) {
      result = result && getDeviceTag()
          .equals(other.getDeviceTag());
    }
    result = result && (hasHierarchy() == other.hasHierarchy());
    if (hasHierarchy()) {
      result = result && getHierarchy()
          .equals(other.getHierarchy());
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
    if (getDeviceNameCount() > 0) {
      hash = (37 * hash) + DEVICE_NAME_FIELD_NUMBER;
      hash = (53 * hash) + getDeviceNameList().hashCode();
    }
    if (hasDeviceTag()) {
      hash = (37 * hash) + DEVICE_TAG_FIELD_NUMBER;
      hash = (53 * hash) + getDeviceTag().hashCode();
    }
    if (hasHierarchy()) {
      hash = (37 * hash) + HIERARCHY_FIELD_NUMBER;
      hash = (53 * hash) + getHierarchy().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.ParallelConf parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.ParallelConf parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.ParallelConf parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.ParallelConf parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.ParallelConf parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.ParallelConf parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.ParallelConf parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.ParallelConf parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.ParallelConf parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.ParallelConf parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.ParallelConf prototype) {
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
   * Protobuf type {@code oneflow.ParallelConf}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.ParallelConf)
      org.oneflow.core.job.ParallelConfOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.PlacementOuterClass.internal_static_oneflow_ParallelConf_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.PlacementOuterClass.internal_static_oneflow_ParallelConf_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.ParallelConf.class, org.oneflow.core.job.ParallelConf.Builder.class);
    }

    // Construct using org.oneflow.core.job.ParallelConf.newBuilder()
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
        getHierarchyFieldBuilder();
      }
    }
    public Builder clear() {
      super.clear();
      deviceName_ = com.google.protobuf.LazyStringArrayList.EMPTY;
      bitField0_ = (bitField0_ & ~0x00000001);
      deviceTag_ = "";
      bitField0_ = (bitField0_ & ~0x00000002);
      if (hierarchyBuilder_ == null) {
        hierarchy_ = null;
      } else {
        hierarchyBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000004);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.PlacementOuterClass.internal_static_oneflow_ParallelConf_descriptor;
    }

    public org.oneflow.core.job.ParallelConf getDefaultInstanceForType() {
      return org.oneflow.core.job.ParallelConf.getDefaultInstance();
    }

    public org.oneflow.core.job.ParallelConf build() {
      org.oneflow.core.job.ParallelConf result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.ParallelConf buildPartial() {
      org.oneflow.core.job.ParallelConf result = new org.oneflow.core.job.ParallelConf(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((bitField0_ & 0x00000001) == 0x00000001)) {
        deviceName_ = deviceName_.getUnmodifiableView();
        bitField0_ = (bitField0_ & ~0x00000001);
      }
      result.deviceName_ = deviceName_;
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000001;
      }
      result.deviceTag_ = deviceTag_;
      if (((from_bitField0_ & 0x00000004) == 0x00000004)) {
        to_bitField0_ |= 0x00000002;
      }
      if (hierarchyBuilder_ == null) {
        result.hierarchy_ = hierarchy_;
      } else {
        result.hierarchy_ = hierarchyBuilder_.build();
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
      if (other instanceof org.oneflow.core.job.ParallelConf) {
        return mergeFrom((org.oneflow.core.job.ParallelConf)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.ParallelConf other) {
      if (other == org.oneflow.core.job.ParallelConf.getDefaultInstance()) return this;
      if (!other.deviceName_.isEmpty()) {
        if (deviceName_.isEmpty()) {
          deviceName_ = other.deviceName_;
          bitField0_ = (bitField0_ & ~0x00000001);
        } else {
          ensureDeviceNameIsMutable();
          deviceName_.addAll(other.deviceName_);
        }
        onChanged();
      }
      if (other.hasDeviceTag()) {
        bitField0_ |= 0x00000002;
        deviceTag_ = other.deviceTag_;
        onChanged();
      }
      if (other.hasHierarchy()) {
        mergeHierarchy(other.getHierarchy());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (!hasDeviceTag()) {
        return false;
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.job.ParallelConf parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.ParallelConf) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private com.google.protobuf.LazyStringList deviceName_ = com.google.protobuf.LazyStringArrayList.EMPTY;
    private void ensureDeviceNameIsMutable() {
      if (!((bitField0_ & 0x00000001) == 0x00000001)) {
        deviceName_ = new com.google.protobuf.LazyStringArrayList(deviceName_);
        bitField0_ |= 0x00000001;
       }
    }
    /**
     * <code>repeated string device_name = 1;</code>
     */
    public com.google.protobuf.ProtocolStringList
        getDeviceNameList() {
      return deviceName_.getUnmodifiableView();
    }
    /**
     * <code>repeated string device_name = 1;</code>
     */
    public int getDeviceNameCount() {
      return deviceName_.size();
    }
    /**
     * <code>repeated string device_name = 1;</code>
     */
    public java.lang.String getDeviceName(int index) {
      return deviceName_.get(index);
    }
    /**
     * <code>repeated string device_name = 1;</code>
     */
    public com.google.protobuf.ByteString
        getDeviceNameBytes(int index) {
      return deviceName_.getByteString(index);
    }
    /**
     * <code>repeated string device_name = 1;</code>
     */
    public Builder setDeviceName(
        int index, java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureDeviceNameIsMutable();
      deviceName_.set(index, value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string device_name = 1;</code>
     */
    public Builder addDeviceName(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureDeviceNameIsMutable();
      deviceName_.add(value);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string device_name = 1;</code>
     */
    public Builder addAllDeviceName(
        java.lang.Iterable<java.lang.String> values) {
      ensureDeviceNameIsMutable();
      com.google.protobuf.AbstractMessageLite.Builder.addAll(
          values, deviceName_);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string device_name = 1;</code>
     */
    public Builder clearDeviceName() {
      deviceName_ = com.google.protobuf.LazyStringArrayList.EMPTY;
      bitField0_ = (bitField0_ & ~0x00000001);
      onChanged();
      return this;
    }
    /**
     * <code>repeated string device_name = 1;</code>
     */
    public Builder addDeviceNameBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  ensureDeviceNameIsMutable();
      deviceName_.add(value);
      onChanged();
      return this;
    }

    private java.lang.Object deviceTag_ = "";
    /**
     * <code>required string device_tag = 2;</code>
     */
    public boolean hasDeviceTag() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>required string device_tag = 2;</code>
     */
    public java.lang.String getDeviceTag() {
      java.lang.Object ref = deviceTag_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        if (bs.isValidUtf8()) {
          deviceTag_ = s;
        }
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>required string device_tag = 2;</code>
     */
    public com.google.protobuf.ByteString
        getDeviceTagBytes() {
      java.lang.Object ref = deviceTag_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        deviceTag_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>required string device_tag = 2;</code>
     */
    public Builder setDeviceTag(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000002;
      deviceTag_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>required string device_tag = 2;</code>
     */
    public Builder clearDeviceTag() {
      bitField0_ = (bitField0_ & ~0x00000002);
      deviceTag_ = getDefaultInstance().getDeviceTag();
      onChanged();
      return this;
    }
    /**
     * <code>required string device_tag = 2;</code>
     */
    public Builder setDeviceTagBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000002;
      deviceTag_ = value;
      onChanged();
      return this;
    }

    private org.oneflow.core.common.ShapeProto hierarchy_ = null;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.common.ShapeProto, org.oneflow.core.common.ShapeProto.Builder, org.oneflow.core.common.ShapeProtoOrBuilder> hierarchyBuilder_;
    /**
     * <code>optional .oneflow.ShapeProto hierarchy = 3;</code>
     */
    public boolean hasHierarchy() {
      return ((bitField0_ & 0x00000004) == 0x00000004);
    }
    /**
     * <code>optional .oneflow.ShapeProto hierarchy = 3;</code>
     */
    public org.oneflow.core.common.ShapeProto getHierarchy() {
      if (hierarchyBuilder_ == null) {
        return hierarchy_ == null ? org.oneflow.core.common.ShapeProto.getDefaultInstance() : hierarchy_;
      } else {
        return hierarchyBuilder_.getMessage();
      }
    }
    /**
     * <code>optional .oneflow.ShapeProto hierarchy = 3;</code>
     */
    public Builder setHierarchy(org.oneflow.core.common.ShapeProto value) {
      if (hierarchyBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        hierarchy_ = value;
        onChanged();
      } else {
        hierarchyBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000004;
      return this;
    }
    /**
     * <code>optional .oneflow.ShapeProto hierarchy = 3;</code>
     */
    public Builder setHierarchy(
        org.oneflow.core.common.ShapeProto.Builder builderForValue) {
      if (hierarchyBuilder_ == null) {
        hierarchy_ = builderForValue.build();
        onChanged();
      } else {
        hierarchyBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000004;
      return this;
    }
    /**
     * <code>optional .oneflow.ShapeProto hierarchy = 3;</code>
     */
    public Builder mergeHierarchy(org.oneflow.core.common.ShapeProto value) {
      if (hierarchyBuilder_ == null) {
        if (((bitField0_ & 0x00000004) == 0x00000004) &&
            hierarchy_ != null &&
            hierarchy_ != org.oneflow.core.common.ShapeProto.getDefaultInstance()) {
          hierarchy_ =
            org.oneflow.core.common.ShapeProto.newBuilder(hierarchy_).mergeFrom(value).buildPartial();
        } else {
          hierarchy_ = value;
        }
        onChanged();
      } else {
        hierarchyBuilder_.mergeFrom(value);
      }
      bitField0_ |= 0x00000004;
      return this;
    }
    /**
     * <code>optional .oneflow.ShapeProto hierarchy = 3;</code>
     */
    public Builder clearHierarchy() {
      if (hierarchyBuilder_ == null) {
        hierarchy_ = null;
        onChanged();
      } else {
        hierarchyBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000004);
      return this;
    }
    /**
     * <code>optional .oneflow.ShapeProto hierarchy = 3;</code>
     */
    public org.oneflow.core.common.ShapeProto.Builder getHierarchyBuilder() {
      bitField0_ |= 0x00000004;
      onChanged();
      return getHierarchyFieldBuilder().getBuilder();
    }
    /**
     * <code>optional .oneflow.ShapeProto hierarchy = 3;</code>
     */
    public org.oneflow.core.common.ShapeProtoOrBuilder getHierarchyOrBuilder() {
      if (hierarchyBuilder_ != null) {
        return hierarchyBuilder_.getMessageOrBuilder();
      } else {
        return hierarchy_ == null ?
            org.oneflow.core.common.ShapeProto.getDefaultInstance() : hierarchy_;
      }
    }
    /**
     * <code>optional .oneflow.ShapeProto hierarchy = 3;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.common.ShapeProto, org.oneflow.core.common.ShapeProto.Builder, org.oneflow.core.common.ShapeProtoOrBuilder> 
        getHierarchyFieldBuilder() {
      if (hierarchyBuilder_ == null) {
        hierarchyBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.oneflow.core.common.ShapeProto, org.oneflow.core.common.ShapeProto.Builder, org.oneflow.core.common.ShapeProtoOrBuilder>(
                getHierarchy(),
                getParentForChildren(),
                isClean());
        hierarchy_ = null;
      }
      return hierarchyBuilder_;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:oneflow.ParallelConf)
  }

  // @@protoc_insertion_point(class_scope:oneflow.ParallelConf)
  private static final org.oneflow.core.job.ParallelConf DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.ParallelConf();
  }

  public static org.oneflow.core.job.ParallelConf getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<ParallelConf>
      PARSER = new com.google.protobuf.AbstractParser<ParallelConf>() {
    public ParallelConf parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new ParallelConf(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<ParallelConf> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<ParallelConf> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.ParallelConf getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

