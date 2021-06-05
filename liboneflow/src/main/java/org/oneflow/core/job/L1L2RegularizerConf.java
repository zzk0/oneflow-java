// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/regularizer_conf.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.L1L2RegularizerConf}
 */
public  final class L1L2RegularizerConf extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.L1L2RegularizerConf)
    L1L2RegularizerConfOrBuilder {
  // Use L1L2RegularizerConf.newBuilder() to construct.
  private L1L2RegularizerConf(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private L1L2RegularizerConf() {
    l1_ = 0F;
    l2_ = 0F;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private L1L2RegularizerConf(
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
          case 13: {
            bitField0_ |= 0x00000001;
            l1_ = input.readFloat();
            break;
          }
          case 21: {
            bitField0_ |= 0x00000002;
            l2_ = input.readFloat();
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
    return org.oneflow.core.job.RegularizerConfOuterClass.internal_static_oneflow_L1L2RegularizerConf_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.RegularizerConfOuterClass.internal_static_oneflow_L1L2RegularizerConf_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.L1L2RegularizerConf.class, org.oneflow.core.job.L1L2RegularizerConf.Builder.class);
  }

  private int bitField0_;
  public static final int L1_FIELD_NUMBER = 1;
  private float l1_;
  /**
   * <code>optional float l1 = 1 [default = 0];</code>
   */
  public boolean hasL1() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>optional float l1 = 1 [default = 0];</code>
   */
  public float getL1() {
    return l1_;
  }

  public static final int L2_FIELD_NUMBER = 2;
  private float l2_;
  /**
   * <code>optional float l2 = 2 [default = 0];</code>
   */
  public boolean hasL2() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>optional float l2 = 2 [default = 0];</code>
   */
  public float getL2() {
    return l2_;
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
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      output.writeFloat(1, l1_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      output.writeFloat(2, l2_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.CodedOutputStream
        .computeFloatSize(1, l1_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.CodedOutputStream
        .computeFloatSize(2, l2_);
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
    if (!(obj instanceof org.oneflow.core.job.L1L2RegularizerConf)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.L1L2RegularizerConf other = (org.oneflow.core.job.L1L2RegularizerConf) obj;

    boolean result = true;
    result = result && (hasL1() == other.hasL1());
    if (hasL1()) {
      result = result && (
          java.lang.Float.floatToIntBits(getL1())
          == java.lang.Float.floatToIntBits(
              other.getL1()));
    }
    result = result && (hasL2() == other.hasL2());
    if (hasL2()) {
      result = result && (
          java.lang.Float.floatToIntBits(getL2())
          == java.lang.Float.floatToIntBits(
              other.getL2()));
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
    if (hasL1()) {
      hash = (37 * hash) + L1_FIELD_NUMBER;
      hash = (53 * hash) + java.lang.Float.floatToIntBits(
          getL1());
    }
    if (hasL2()) {
      hash = (37 * hash) + L2_FIELD_NUMBER;
      hash = (53 * hash) + java.lang.Float.floatToIntBits(
          getL2());
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.L1L2RegularizerConf parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.L1L2RegularizerConf parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.L1L2RegularizerConf parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.L1L2RegularizerConf parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.L1L2RegularizerConf parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.L1L2RegularizerConf parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.L1L2RegularizerConf parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.L1L2RegularizerConf parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.L1L2RegularizerConf parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.L1L2RegularizerConf parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.L1L2RegularizerConf prototype) {
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
   * Protobuf type {@code oneflow.L1L2RegularizerConf}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.L1L2RegularizerConf)
      org.oneflow.core.job.L1L2RegularizerConfOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.RegularizerConfOuterClass.internal_static_oneflow_L1L2RegularizerConf_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.RegularizerConfOuterClass.internal_static_oneflow_L1L2RegularizerConf_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.L1L2RegularizerConf.class, org.oneflow.core.job.L1L2RegularizerConf.Builder.class);
    }

    // Construct using org.oneflow.core.job.L1L2RegularizerConf.newBuilder()
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
      l1_ = 0F;
      bitField0_ = (bitField0_ & ~0x00000001);
      l2_ = 0F;
      bitField0_ = (bitField0_ & ~0x00000002);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.RegularizerConfOuterClass.internal_static_oneflow_L1L2RegularizerConf_descriptor;
    }

    public org.oneflow.core.job.L1L2RegularizerConf getDefaultInstanceForType() {
      return org.oneflow.core.job.L1L2RegularizerConf.getDefaultInstance();
    }

    public org.oneflow.core.job.L1L2RegularizerConf build() {
      org.oneflow.core.job.L1L2RegularizerConf result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.L1L2RegularizerConf buildPartial() {
      org.oneflow.core.job.L1L2RegularizerConf result = new org.oneflow.core.job.L1L2RegularizerConf(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      result.l1_ = l1_;
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000002;
      }
      result.l2_ = l2_;
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
      if (other instanceof org.oneflow.core.job.L1L2RegularizerConf) {
        return mergeFrom((org.oneflow.core.job.L1L2RegularizerConf)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.L1L2RegularizerConf other) {
      if (other == org.oneflow.core.job.L1L2RegularizerConf.getDefaultInstance()) return this;
      if (other.hasL1()) {
        setL1(other.getL1());
      }
      if (other.hasL2()) {
        setL2(other.getL2());
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
      org.oneflow.core.job.L1L2RegularizerConf parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.L1L2RegularizerConf) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private float l1_ ;
    /**
     * <code>optional float l1 = 1 [default = 0];</code>
     */
    public boolean hasL1() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>optional float l1 = 1 [default = 0];</code>
     */
    public float getL1() {
      return l1_;
    }
    /**
     * <code>optional float l1 = 1 [default = 0];</code>
     */
    public Builder setL1(float value) {
      bitField0_ |= 0x00000001;
      l1_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional float l1 = 1 [default = 0];</code>
     */
    public Builder clearL1() {
      bitField0_ = (bitField0_ & ~0x00000001);
      l1_ = 0F;
      onChanged();
      return this;
    }

    private float l2_ ;
    /**
     * <code>optional float l2 = 2 [default = 0];</code>
     */
    public boolean hasL2() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>optional float l2 = 2 [default = 0];</code>
     */
    public float getL2() {
      return l2_;
    }
    /**
     * <code>optional float l2 = 2 [default = 0];</code>
     */
    public Builder setL2(float value) {
      bitField0_ |= 0x00000002;
      l2_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional float l2 = 2 [default = 0];</code>
     */
    public Builder clearL2() {
      bitField0_ = (bitField0_ & ~0x00000002);
      l2_ = 0F;
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


    // @@protoc_insertion_point(builder_scope:oneflow.L1L2RegularizerConf)
  }

  // @@protoc_insertion_point(class_scope:oneflow.L1L2RegularizerConf)
  private static final org.oneflow.core.job.L1L2RegularizerConf DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.L1L2RegularizerConf();
  }

  public static org.oneflow.core.job.L1L2RegularizerConf getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<L1L2RegularizerConf>
      PARSER = new com.google.protobuf.AbstractParser<L1L2RegularizerConf>() {
    public L1L2RegularizerConf parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new L1L2RegularizerConf(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<L1L2RegularizerConf> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<L1L2RegularizerConf> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.L1L2RegularizerConf getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

