// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/env.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.CppLoggingConf}
 */
public  final class CppLoggingConf extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.CppLoggingConf)
    CppLoggingConfOrBuilder {
  // Use CppLoggingConf.newBuilder() to construct.
  private CppLoggingConf(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private CppLoggingConf() {
    logDir_ = "./log";
    logtostderr_ = 0;
    logbuflevel_ = -1;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private CppLoggingConf(
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
            logDir_ = bs;
            break;
          }
          case 16: {
            bitField0_ |= 0x00000002;
            logtostderr_ = input.readInt32();
            break;
          }
          case 24: {
            bitField0_ |= 0x00000004;
            logbuflevel_ = input.readInt32();
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
    return org.oneflow.core.job.Env.internal_static_oneflow_CppLoggingConf_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.Env.internal_static_oneflow_CppLoggingConf_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.CppLoggingConf.class, org.oneflow.core.job.CppLoggingConf.Builder.class);
  }

  private int bitField0_;
  public static final int LOG_DIR_FIELD_NUMBER = 1;
  private volatile java.lang.Object logDir_;
  /**
   * <code>optional string log_dir = 1 [default = "./log"];</code>
   */
  public boolean hasLogDir() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>optional string log_dir = 1 [default = "./log"];</code>
   */
  public java.lang.String getLogDir() {
    java.lang.Object ref = logDir_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      if (bs.isValidUtf8()) {
        logDir_ = s;
      }
      return s;
    }
  }
  /**
   * <code>optional string log_dir = 1 [default = "./log"];</code>
   */
  public com.google.protobuf.ByteString
      getLogDirBytes() {
    java.lang.Object ref = logDir_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      logDir_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  public static final int LOGTOSTDERR_FIELD_NUMBER = 2;
  private int logtostderr_;
  /**
   * <code>optional int32 logtostderr = 2 [default = 0];</code>
   */
  public boolean hasLogtostderr() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>optional int32 logtostderr = 2 [default = 0];</code>
   */
  public int getLogtostderr() {
    return logtostderr_;
  }

  public static final int LOGBUFLEVEL_FIELD_NUMBER = 3;
  private int logbuflevel_;
  /**
   * <code>optional int32 logbuflevel = 3 [default = -1];</code>
   */
  public boolean hasLogbuflevel() {
    return ((bitField0_ & 0x00000004) == 0x00000004);
  }
  /**
   * <code>optional int32 logbuflevel = 3 [default = -1];</code>
   */
  public int getLogbuflevel() {
    return logbuflevel_;
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
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, logDir_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      output.writeInt32(2, logtostderr_);
    }
    if (((bitField0_ & 0x00000004) == 0x00000004)) {
      output.writeInt32(3, logbuflevel_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(1, logDir_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(2, logtostderr_);
    }
    if (((bitField0_ & 0x00000004) == 0x00000004)) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt32Size(3, logbuflevel_);
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
    if (!(obj instanceof org.oneflow.core.job.CppLoggingConf)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.CppLoggingConf other = (org.oneflow.core.job.CppLoggingConf) obj;

    boolean result = true;
    result = result && (hasLogDir() == other.hasLogDir());
    if (hasLogDir()) {
      result = result && getLogDir()
          .equals(other.getLogDir());
    }
    result = result && (hasLogtostderr() == other.hasLogtostderr());
    if (hasLogtostderr()) {
      result = result && (getLogtostderr()
          == other.getLogtostderr());
    }
    result = result && (hasLogbuflevel() == other.hasLogbuflevel());
    if (hasLogbuflevel()) {
      result = result && (getLogbuflevel()
          == other.getLogbuflevel());
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
    if (hasLogDir()) {
      hash = (37 * hash) + LOG_DIR_FIELD_NUMBER;
      hash = (53 * hash) + getLogDir().hashCode();
    }
    if (hasLogtostderr()) {
      hash = (37 * hash) + LOGTOSTDERR_FIELD_NUMBER;
      hash = (53 * hash) + getLogtostderr();
    }
    if (hasLogbuflevel()) {
      hash = (37 * hash) + LOGBUFLEVEL_FIELD_NUMBER;
      hash = (53 * hash) + getLogbuflevel();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.CppLoggingConf parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.CppLoggingConf parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.CppLoggingConf parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.CppLoggingConf parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.CppLoggingConf parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.CppLoggingConf parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.CppLoggingConf parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.CppLoggingConf parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.CppLoggingConf parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.CppLoggingConf parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.CppLoggingConf prototype) {
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
   * Protobuf type {@code oneflow.CppLoggingConf}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.CppLoggingConf)
      org.oneflow.core.job.CppLoggingConfOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.Env.internal_static_oneflow_CppLoggingConf_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.Env.internal_static_oneflow_CppLoggingConf_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.CppLoggingConf.class, org.oneflow.core.job.CppLoggingConf.Builder.class);
    }

    // Construct using org.oneflow.core.job.CppLoggingConf.newBuilder()
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
      logDir_ = "./log";
      bitField0_ = (bitField0_ & ~0x00000001);
      logtostderr_ = 0;
      bitField0_ = (bitField0_ & ~0x00000002);
      logbuflevel_ = -1;
      bitField0_ = (bitField0_ & ~0x00000004);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.Env.internal_static_oneflow_CppLoggingConf_descriptor;
    }

    public org.oneflow.core.job.CppLoggingConf getDefaultInstanceForType() {
      return org.oneflow.core.job.CppLoggingConf.getDefaultInstance();
    }

    public org.oneflow.core.job.CppLoggingConf build() {
      org.oneflow.core.job.CppLoggingConf result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.CppLoggingConf buildPartial() {
      org.oneflow.core.job.CppLoggingConf result = new org.oneflow.core.job.CppLoggingConf(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      result.logDir_ = logDir_;
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000002;
      }
      result.logtostderr_ = logtostderr_;
      if (((from_bitField0_ & 0x00000004) == 0x00000004)) {
        to_bitField0_ |= 0x00000004;
      }
      result.logbuflevel_ = logbuflevel_;
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
      if (other instanceof org.oneflow.core.job.CppLoggingConf) {
        return mergeFrom((org.oneflow.core.job.CppLoggingConf)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.CppLoggingConf other) {
      if (other == org.oneflow.core.job.CppLoggingConf.getDefaultInstance()) return this;
      if (other.hasLogDir()) {
        bitField0_ |= 0x00000001;
        logDir_ = other.logDir_;
        onChanged();
      }
      if (other.hasLogtostderr()) {
        setLogtostderr(other.getLogtostderr());
      }
      if (other.hasLogbuflevel()) {
        setLogbuflevel(other.getLogbuflevel());
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
      org.oneflow.core.job.CppLoggingConf parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.CppLoggingConf) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private java.lang.Object logDir_ = "./log";
    /**
     * <code>optional string log_dir = 1 [default = "./log"];</code>
     */
    public boolean hasLogDir() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>optional string log_dir = 1 [default = "./log"];</code>
     */
    public java.lang.String getLogDir() {
      java.lang.Object ref = logDir_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        if (bs.isValidUtf8()) {
          logDir_ = s;
        }
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <code>optional string log_dir = 1 [default = "./log"];</code>
     */
    public com.google.protobuf.ByteString
        getLogDirBytes() {
      java.lang.Object ref = logDir_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        logDir_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <code>optional string log_dir = 1 [default = "./log"];</code>
     */
    public Builder setLogDir(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000001;
      logDir_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional string log_dir = 1 [default = "./log"];</code>
     */
    public Builder clearLogDir() {
      bitField0_ = (bitField0_ & ~0x00000001);
      logDir_ = getDefaultInstance().getLogDir();
      onChanged();
      return this;
    }
    /**
     * <code>optional string log_dir = 1 [default = "./log"];</code>
     */
    public Builder setLogDirBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000001;
      logDir_ = value;
      onChanged();
      return this;
    }

    private int logtostderr_ ;
    /**
     * <code>optional int32 logtostderr = 2 [default = 0];</code>
     */
    public boolean hasLogtostderr() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>optional int32 logtostderr = 2 [default = 0];</code>
     */
    public int getLogtostderr() {
      return logtostderr_;
    }
    /**
     * <code>optional int32 logtostderr = 2 [default = 0];</code>
     */
    public Builder setLogtostderr(int value) {
      bitField0_ |= 0x00000002;
      logtostderr_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional int32 logtostderr = 2 [default = 0];</code>
     */
    public Builder clearLogtostderr() {
      bitField0_ = (bitField0_ & ~0x00000002);
      logtostderr_ = 0;
      onChanged();
      return this;
    }

    private int logbuflevel_ = -1;
    /**
     * <code>optional int32 logbuflevel = 3 [default = -1];</code>
     */
    public boolean hasLogbuflevel() {
      return ((bitField0_ & 0x00000004) == 0x00000004);
    }
    /**
     * <code>optional int32 logbuflevel = 3 [default = -1];</code>
     */
    public int getLogbuflevel() {
      return logbuflevel_;
    }
    /**
     * <code>optional int32 logbuflevel = 3 [default = -1];</code>
     */
    public Builder setLogbuflevel(int value) {
      bitField0_ |= 0x00000004;
      logbuflevel_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>optional int32 logbuflevel = 3 [default = -1];</code>
     */
    public Builder clearLogbuflevel() {
      bitField0_ = (bitField0_ & ~0x00000004);
      logbuflevel_ = -1;
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


    // @@protoc_insertion_point(builder_scope:oneflow.CppLoggingConf)
  }

  // @@protoc_insertion_point(class_scope:oneflow.CppLoggingConf)
  private static final org.oneflow.core.job.CppLoggingConf DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.CppLoggingConf();
  }

  public static org.oneflow.core.job.CppLoggingConf getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<CppLoggingConf>
      PARSER = new com.google.protobuf.AbstractParser<CppLoggingConf>() {
    public CppLoggingConf parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new CppLoggingConf(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<CppLoggingConf> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<CppLoggingConf> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.CppLoggingConf getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

