// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/initializer_conf.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.RandomUniformIntInitializerConf}
 */
public  final class RandomUniformIntInitializerConf extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.RandomUniformIntInitializerConf)
    RandomUniformIntInitializerConfOrBuilder {
  // Use RandomUniformIntInitializerConf.newBuilder() to construct.
  private RandomUniformIntInitializerConf(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private RandomUniformIntInitializerConf() {
    min_ = 0;
    max_ = 1;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private RandomUniformIntInitializerConf(
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
            bitField0_ |= 0x00000001;
            min_ = input.readInt32();
            break;
          }
          case 16: {
            bitField0_ |= 0x00000002;
            max_ = input.readInt32();
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
    return org.oneflow.core.job.InitializerConfOuterClass.internal_static_oneflow_RandomUniformIntInitializerConf_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.InitializerConfOuterClass.internal_static_oneflow_RandomUniformIntInitializerConf_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.RandomUniformIntInitializerConf.class, org.oneflow.core.job.RandomUniformIntInitializerConf.Builder.class);
  }

  private int bitField0_;
  public static final int MIN_FIELD_NUMBER = 1;
  private int min_;
  /**
   * <code>optional int32 min = 1 [default = 0];</code>
   */
  public boolean hasMin() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>optional int32 min = 1 [default = 0];</code>
   */
  public int getMin() {
    return min_;
  }

  public static final int MAX_FIELD_NUMBER = 2;
  private int max_;
  /**
   * <code>optional int32 max = 2 [default = 1];</code>
   */
  public boolean hasMax() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>optional int32 max = 2 [default = 1];</code>
   */
  public int getMax() {
    return max_;
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
      output.writeInt32(1, min_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      output.writeInt32(2, max_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(1, min_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(2, max_);
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
    if (!(obj instanceof org.oneflow.core.job.RandomUniformIntInitializerConf)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.RandomUniformIntInitializerConf other = (org.oneflow.core.job.RandomUniformIntInitializerConf) obj;

    boolean result = true;
    result = result && (hasMin() == other.hasMin());
    if (hasMin()) {
      result = result && (getMin()
          == other.getMin());
    }
    result = result && (hasMax() == other.hasMax());
    if (hasMax()) {
      result = result && (getMax()
          == other.getMax());
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
    if (hasMin()) {
      hash = (37 * hash) + MIN_FIELD_NUMBER;
      hash = (53 * hash) + getMin();
    }
    if (hasMax()) {
      hash = (37 * hash) + MAX_FIELD_NUMBER;
      hash = (53 * hash) + getMax();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.RandomUniformIntInitializerConf parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.RandomUniformIntInitializerConf parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.RandomUniformIntInitializerConf parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.RandomUniformIntInitializerConf parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.RandomUniformIntInitializerConf parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.RandomUniformIntInitializerConf parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.RandomUniformIntInitializerConf parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.RandomUniformIntInitializerConf parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.RandomUniformIntInitializerConf parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.RandomUniformIntInitializerConf parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.RandomUniformIntInitializerConf prototype) {
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
   * Protobuf type {@code oneflow.RandomUniformIntInitializerConf}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.RandomUniformIntInitializerConf)
      org.oneflow.core.job.RandomUniformIntInitializerConfOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.InitializerConfOuterClass.internal_static_oneflow_RandomUniformIntInitializerConf_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.InitializerConfOuterClass.internal_static_oneflow_RandomUniformIntInitializerConf_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.RandomUniformIntInitializerConf.class, org.oneflow.core.job.RandomUniformIntInitializerConf.Builder.class);
    }

    // Construct using org.oneflow.core.job.RandomUniformIntInitializerConf.newBuilder()
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
      min_ = 0;
      bitField0_ = (bitField0_ & ~0x00000001);
      max_ = 1;
      bitField0_ = (bitField0_ & ~0x00000002);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.InitializerConfOuterClass.internal_static_oneflow_RandomUniformIntInitializerConf_descriptor;
    }

    public org.oneflow.core.job.RandomUniformIntInitializerConf getDefaultInstanceForType() {
      return org.oneflow.core.job.RandomUniformIntInitializerConf.getDefaultInstance();
    }

    public org.oneflow.core.job.RandomUniformIntInitializerConf build() {
      org.oneflow.core.job.RandomUniformIntInitializerConf result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.RandomUniformIntInitializerConf buildPartial() {
      org.oneflow.core.job.RandomUniformIntInitializerConf result = new org.oneflow.core.job.RandomUniformIntInitializerConf(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      result.min_ = min_;
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000002;
      }
      result.max_ = max_;
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
      if (other instanceof org.oneflow.core.job.RandomUniformIntInitializerConf) {
        return mergeFrom((org.oneflow.core.job.RandomUniformIntInitializerConf)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.RandomUniformIntInitializerConf other) {
      if (other == org.oneflow.core.job.RandomUniformIntInitializerConf.getDefaultInstance()) return this;
      if (other.hasMin()) {
        setMin(other.getMin());
      }
      if (other.hasMax()) {
        setMax(other.getMax());
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
      org.oneflow.core.job.RandomUniformIntInitializerConf parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.RandomUniformIntInitializerConf) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private int min_ ;
    /**
     * <code>optional int32 min = 1 [default = 0];</code>
     */
    public boolean hasMin() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>optional int32 min = 1 [default = 0];</code>
     */
    public int getMin() {
      return min_;
    }
    /**
     * <code>optional int32 min = 1 [default = 0];</code>
     */
    public Builder setMin(int value) {
      bitField0_ |= 0x00000001;
      min_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional int32 min = 1 [default = 0];</code>
     */
    public Builder clearMin() {
      bitField0_ = (bitField0_ & ~0x00000001);
      min_ = 0;
      onChanged();
      return this;
    }

    private int max_ = 1;
    /**
     * <code>optional int32 max = 2 [default = 1];</code>
     */
    public boolean hasMax() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>optional int32 max = 2 [default = 1];</code>
     */
    public int getMax() {
      return max_;
    }
    /**
     * <code>optional int32 max = 2 [default = 1];</code>
     */
    public Builder setMax(int value) {
      bitField0_ |= 0x00000002;
      max_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional int32 max = 2 [default = 1];</code>
     */
    public Builder clearMax() {
      bitField0_ = (bitField0_ & ~0x00000002);
      max_ = 1;
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


    // @@protoc_insertion_point(builder_scope:oneflow.RandomUniformIntInitializerConf)
  }

  // @@protoc_insertion_point(class_scope:oneflow.RandomUniformIntInitializerConf)
  private static final org.oneflow.core.job.RandomUniformIntInitializerConf DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.RandomUniformIntInitializerConf();
  }

  public static org.oneflow.core.job.RandomUniformIntInitializerConf getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<RandomUniformIntInitializerConf>
      PARSER = new com.google.protobuf.AbstractParser<RandomUniformIntInitializerConf>() {
    public RandomUniformIntInitializerConf parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new RandomUniformIntInitializerConf(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<RandomUniformIntInitializerConf> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<RandomUniformIntInitializerConf> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.RandomUniformIntInitializerConf getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

