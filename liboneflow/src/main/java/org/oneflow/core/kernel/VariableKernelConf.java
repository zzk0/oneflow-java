// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/kernel/kernel.proto

package org.oneflow.core.kernel;

/**
 * Protobuf type {@code oneflow.VariableKernelConf}
 */
public  final class VariableKernelConf extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.VariableKernelConf)
    VariableKernelConfOrBuilder {
  // Use VariableKernelConf.newBuilder() to construct.
  private VariableKernelConf(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private VariableKernelConf() {
    isFwInplace_ = false;
    isBwInplace_ = false;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private VariableKernelConf(
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
            isFwInplace_ = input.readBool();
            break;
          }
          case 16: {
            bitField0_ |= 0x00000002;
            isBwInplace_ = input.readBool();
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
    return org.oneflow.core.kernel.Kernel.internal_static_oneflow_VariableKernelConf_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.kernel.Kernel.internal_static_oneflow_VariableKernelConf_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.kernel.VariableKernelConf.class, org.oneflow.core.kernel.VariableKernelConf.Builder.class);
  }

  private int bitField0_;
  public static final int IS_FW_INPLACE_FIELD_NUMBER = 1;
  private boolean isFwInplace_;
  /**
   * <code>required bool is_fw_inplace = 1;</code>
   */
  public boolean hasIsFwInplace() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>required bool is_fw_inplace = 1;</code>
   */
  public boolean getIsFwInplace() {
    return isFwInplace_;
  }

  public static final int IS_BW_INPLACE_FIELD_NUMBER = 2;
  private boolean isBwInplace_;
  /**
   * <code>required bool is_bw_inplace = 2;</code>
   */
  public boolean hasIsBwInplace() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>required bool is_bw_inplace = 2;</code>
   */
  public boolean getIsBwInplace() {
    return isBwInplace_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (!hasIsFwInplace()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!hasIsBwInplace()) {
      memoizedIsInitialized = 0;
      return false;
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      output.writeBool(1, isFwInplace_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      output.writeBool(2, isBwInplace_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.CodedOutputStream
        .computeBoolSize(1, isFwInplace_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.CodedOutputStream
        .computeBoolSize(2, isBwInplace_);
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
    if (!(obj instanceof org.oneflow.core.kernel.VariableKernelConf)) {
      return super.equals(obj);
    }
    org.oneflow.core.kernel.VariableKernelConf other = (org.oneflow.core.kernel.VariableKernelConf) obj;

    boolean result = true;
    result = result && (hasIsFwInplace() == other.hasIsFwInplace());
    if (hasIsFwInplace()) {
      result = result && (getIsFwInplace()
          == other.getIsFwInplace());
    }
    result = result && (hasIsBwInplace() == other.hasIsBwInplace());
    if (hasIsBwInplace()) {
      result = result && (getIsBwInplace()
          == other.getIsBwInplace());
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
    if (hasIsFwInplace()) {
      hash = (37 * hash) + IS_FW_INPLACE_FIELD_NUMBER;
      hash = (53 * hash) + com.google.protobuf.Internal.hashBoolean(
          getIsFwInplace());
    }
    if (hasIsBwInplace()) {
      hash = (37 * hash) + IS_BW_INPLACE_FIELD_NUMBER;
      hash = (53 * hash) + com.google.protobuf.Internal.hashBoolean(
          getIsBwInplace());
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.kernel.VariableKernelConf parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.kernel.VariableKernelConf parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.kernel.VariableKernelConf parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.kernel.VariableKernelConf parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.kernel.VariableKernelConf parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.kernel.VariableKernelConf parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.kernel.VariableKernelConf parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.kernel.VariableKernelConf parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.kernel.VariableKernelConf parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.kernel.VariableKernelConf parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.kernel.VariableKernelConf prototype) {
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
   * Protobuf type {@code oneflow.VariableKernelConf}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.VariableKernelConf)
      org.oneflow.core.kernel.VariableKernelConfOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.kernel.Kernel.internal_static_oneflow_VariableKernelConf_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.kernel.Kernel.internal_static_oneflow_VariableKernelConf_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.kernel.VariableKernelConf.class, org.oneflow.core.kernel.VariableKernelConf.Builder.class);
    }

    // Construct using org.oneflow.core.kernel.VariableKernelConf.newBuilder()
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
      isFwInplace_ = false;
      bitField0_ = (bitField0_ & ~0x00000001);
      isBwInplace_ = false;
      bitField0_ = (bitField0_ & ~0x00000002);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.kernel.Kernel.internal_static_oneflow_VariableKernelConf_descriptor;
    }

    public org.oneflow.core.kernel.VariableKernelConf getDefaultInstanceForType() {
      return org.oneflow.core.kernel.VariableKernelConf.getDefaultInstance();
    }

    public org.oneflow.core.kernel.VariableKernelConf build() {
      org.oneflow.core.kernel.VariableKernelConf result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.kernel.VariableKernelConf buildPartial() {
      org.oneflow.core.kernel.VariableKernelConf result = new org.oneflow.core.kernel.VariableKernelConf(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      result.isFwInplace_ = isFwInplace_;
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000002;
      }
      result.isBwInplace_ = isBwInplace_;
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
      if (other instanceof org.oneflow.core.kernel.VariableKernelConf) {
        return mergeFrom((org.oneflow.core.kernel.VariableKernelConf)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.kernel.VariableKernelConf other) {
      if (other == org.oneflow.core.kernel.VariableKernelConf.getDefaultInstance()) return this;
      if (other.hasIsFwInplace()) {
        setIsFwInplace(other.getIsFwInplace());
      }
      if (other.hasIsBwInplace()) {
        setIsBwInplace(other.getIsBwInplace());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (!hasIsFwInplace()) {
        return false;
      }
      if (!hasIsBwInplace()) {
        return false;
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.kernel.VariableKernelConf parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.kernel.VariableKernelConf) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private boolean isFwInplace_ ;
    /**
     * <code>required bool is_fw_inplace = 1;</code>
     */
    public boolean hasIsFwInplace() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>required bool is_fw_inplace = 1;</code>
     */
    public boolean getIsFwInplace() {
      return isFwInplace_;
    }
    /**
     * <code>required bool is_fw_inplace = 1;</code>
     */
    public Builder setIsFwInplace(boolean value) {
      bitField0_ |= 0x00000001;
      isFwInplace_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>required bool is_fw_inplace = 1;</code>
     */
    public Builder clearIsFwInplace() {
      bitField0_ = (bitField0_ & ~0x00000001);
      isFwInplace_ = false;
      onChanged();
      return this;
    }

    private boolean isBwInplace_ ;
    /**
     * <code>required bool is_bw_inplace = 2;</code>
     */
    public boolean hasIsBwInplace() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>required bool is_bw_inplace = 2;</code>
     */
    public boolean getIsBwInplace() {
      return isBwInplace_;
    }
    /**
     * <code>required bool is_bw_inplace = 2;</code>
     */
    public Builder setIsBwInplace(boolean value) {
      bitField0_ |= 0x00000002;
      isBwInplace_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>required bool is_bw_inplace = 2;</code>
     */
    public Builder clearIsBwInplace() {
      bitField0_ = (bitField0_ & ~0x00000002);
      isBwInplace_ = false;
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


    // @@protoc_insertion_point(builder_scope:oneflow.VariableKernelConf)
  }

  // @@protoc_insertion_point(class_scope:oneflow.VariableKernelConf)
  private static final org.oneflow.core.kernel.VariableKernelConf DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.kernel.VariableKernelConf();
  }

  public static org.oneflow.core.kernel.VariableKernelConf getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<VariableKernelConf>
      PARSER = new com.google.protobuf.AbstractParser<VariableKernelConf>() {
    public VariableKernelConf parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new VariableKernelConf(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<VariableKernelConf> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<VariableKernelConf> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.kernel.VariableKernelConf getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

