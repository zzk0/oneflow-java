// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/kernel/kernel.proto

package org.oneflow.core.kernel;

/**
 * Protobuf type {@code oneflow.DecodeRandomKernelConf}
 */
public  final class DecodeRandomKernelConf extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.DecodeRandomKernelConf)
    DecodeRandomKernelConfOrBuilder {
  // Use DecodeRandomKernelConf.newBuilder() to construct.
  private DecodeRandomKernelConf(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private DecodeRandomKernelConf() {
    randomSeed_ = 0;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private DecodeRandomKernelConf(
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
            randomSeed_ = input.readUInt32();
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
    return org.oneflow.core.kernel.Kernel.internal_static_oneflow_DecodeRandomKernelConf_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.kernel.Kernel.internal_static_oneflow_DecodeRandomKernelConf_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.kernel.DecodeRandomKernelConf.class, org.oneflow.core.kernel.DecodeRandomKernelConf.Builder.class);
  }

  private int bitField0_;
  public static final int RANDOM_SEED_FIELD_NUMBER = 1;
  private int randomSeed_;
  /**
   * <code>required uint32 random_seed = 1;</code>
   */
  public boolean hasRandomSeed() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>required uint32 random_seed = 1;</code>
   */
  public int getRandomSeed() {
    return randomSeed_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (!hasRandomSeed()) {
      memoizedIsInitialized = 0;
      return false;
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      output.writeUInt32(1, randomSeed_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.CodedOutputStream
        .computeUInt32Size(1, randomSeed_);
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
    if (!(obj instanceof org.oneflow.core.kernel.DecodeRandomKernelConf)) {
      return super.equals(obj);
    }
    org.oneflow.core.kernel.DecodeRandomKernelConf other = (org.oneflow.core.kernel.DecodeRandomKernelConf) obj;

    boolean result = true;
    result = result && (hasRandomSeed() == other.hasRandomSeed());
    if (hasRandomSeed()) {
      result = result && (getRandomSeed()
          == other.getRandomSeed());
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
    if (hasRandomSeed()) {
      hash = (37 * hash) + RANDOM_SEED_FIELD_NUMBER;
      hash = (53 * hash) + getRandomSeed();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.kernel.DecodeRandomKernelConf parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.kernel.DecodeRandomKernelConf parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.kernel.DecodeRandomKernelConf parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.kernel.DecodeRandomKernelConf parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.kernel.DecodeRandomKernelConf parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.kernel.DecodeRandomKernelConf parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.kernel.DecodeRandomKernelConf parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.kernel.DecodeRandomKernelConf parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.kernel.DecodeRandomKernelConf parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.kernel.DecodeRandomKernelConf parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.kernel.DecodeRandomKernelConf prototype) {
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
   * Protobuf type {@code oneflow.DecodeRandomKernelConf}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.DecodeRandomKernelConf)
      org.oneflow.core.kernel.DecodeRandomKernelConfOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.kernel.Kernel.internal_static_oneflow_DecodeRandomKernelConf_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.kernel.Kernel.internal_static_oneflow_DecodeRandomKernelConf_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.kernel.DecodeRandomKernelConf.class, org.oneflow.core.kernel.DecodeRandomKernelConf.Builder.class);
    }

    // Construct using org.oneflow.core.kernel.DecodeRandomKernelConf.newBuilder()
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
      randomSeed_ = 0;
      bitField0_ = (bitField0_ & ~0x00000001);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.kernel.Kernel.internal_static_oneflow_DecodeRandomKernelConf_descriptor;
    }

    public org.oneflow.core.kernel.DecodeRandomKernelConf getDefaultInstanceForType() {
      return org.oneflow.core.kernel.DecodeRandomKernelConf.getDefaultInstance();
    }

    public org.oneflow.core.kernel.DecodeRandomKernelConf build() {
      org.oneflow.core.kernel.DecodeRandomKernelConf result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.kernel.DecodeRandomKernelConf buildPartial() {
      org.oneflow.core.kernel.DecodeRandomKernelConf result = new org.oneflow.core.kernel.DecodeRandomKernelConf(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      result.randomSeed_ = randomSeed_;
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
      if (other instanceof org.oneflow.core.kernel.DecodeRandomKernelConf) {
        return mergeFrom((org.oneflow.core.kernel.DecodeRandomKernelConf)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.kernel.DecodeRandomKernelConf other) {
      if (other == org.oneflow.core.kernel.DecodeRandomKernelConf.getDefaultInstance()) return this;
      if (other.hasRandomSeed()) {
        setRandomSeed(other.getRandomSeed());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (!hasRandomSeed()) {
        return false;
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.kernel.DecodeRandomKernelConf parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.kernel.DecodeRandomKernelConf) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private int randomSeed_ ;
    /**
     * <code>required uint32 random_seed = 1;</code>
     */
    public boolean hasRandomSeed() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>required uint32 random_seed = 1;</code>
     */
    public int getRandomSeed() {
      return randomSeed_;
    }
    /**
     * <code>required uint32 random_seed = 1;</code>
     */
    public Builder setRandomSeed(int value) {
      bitField0_ |= 0x00000001;
      randomSeed_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>required uint32 random_seed = 1;</code>
     */
    public Builder clearRandomSeed() {
      bitField0_ = (bitField0_ & ~0x00000001);
      randomSeed_ = 0;
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


    // @@protoc_insertion_point(builder_scope:oneflow.DecodeRandomKernelConf)
  }

  // @@protoc_insertion_point(class_scope:oneflow.DecodeRandomKernelConf)
  private static final org.oneflow.core.kernel.DecodeRandomKernelConf DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.kernel.DecodeRandomKernelConf();
  }

  public static org.oneflow.core.kernel.DecodeRandomKernelConf getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<DecodeRandomKernelConf>
      PARSER = new com.google.protobuf.AbstractParser<DecodeRandomKernelConf>() {
    public DecodeRandomKernelConf parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new DecodeRandomKernelConf(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<DecodeRandomKernelConf> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<DecodeRandomKernelConf> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.kernel.DecodeRandomKernelConf getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
