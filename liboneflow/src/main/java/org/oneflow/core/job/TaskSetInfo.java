// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/task.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.TaskSetInfo}
 */
public  final class TaskSetInfo extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.TaskSetInfo)
    TaskSetInfoOrBuilder {
  // Use TaskSetInfo.newBuilder() to construct.
  private TaskSetInfo(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private TaskSetInfo() {
    chainId_ = 0L;
    orderInGraph_ = 0L;
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private TaskSetInfo(
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
          case 32: {
            bitField0_ |= 0x00000001;
            chainId_ = input.readInt64();
            break;
          }
          case 40: {
            bitField0_ |= 0x00000002;
            orderInGraph_ = input.readInt64();
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
    return org.oneflow.core.job.Task.internal_static_oneflow_TaskSetInfo_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.Task.internal_static_oneflow_TaskSetInfo_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.TaskSetInfo.class, org.oneflow.core.job.TaskSetInfo.Builder.class);
  }

  private int bitField0_;
  public static final int CHAIN_ID_FIELD_NUMBER = 4;
  private long chainId_;
  /**
   * <code>required int64 chain_id = 4;</code>
   */
  public boolean hasChainId() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>required int64 chain_id = 4;</code>
   */
  public long getChainId() {
    return chainId_;
  }

  public static final int ORDER_IN_GRAPH_FIELD_NUMBER = 5;
  private long orderInGraph_;
  /**
   * <code>required int64 order_in_graph = 5;</code>
   */
  public boolean hasOrderInGraph() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>required int64 order_in_graph = 5;</code>
   */
  public long getOrderInGraph() {
    return orderInGraph_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (!hasChainId()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!hasOrderInGraph()) {
      memoizedIsInitialized = 0;
      return false;
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      output.writeInt64(4, chainId_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      output.writeInt64(5, orderInGraph_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt64Size(4, chainId_);
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.CodedOutputStream
        .computeInt64Size(5, orderInGraph_);
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
    if (!(obj instanceof org.oneflow.core.job.TaskSetInfo)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.TaskSetInfo other = (org.oneflow.core.job.TaskSetInfo) obj;

    boolean result = true;
    result = result && (hasChainId() == other.hasChainId());
    if (hasChainId()) {
      result = result && (getChainId()
          == other.getChainId());
    }
    result = result && (hasOrderInGraph() == other.hasOrderInGraph());
    if (hasOrderInGraph()) {
      result = result && (getOrderInGraph()
          == other.getOrderInGraph());
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
    if (hasChainId()) {
      hash = (37 * hash) + CHAIN_ID_FIELD_NUMBER;
      hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
          getChainId());
    }
    if (hasOrderInGraph()) {
      hash = (37 * hash) + ORDER_IN_GRAPH_FIELD_NUMBER;
      hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
          getOrderInGraph());
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.TaskSetInfo parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.TaskSetInfo parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.TaskSetInfo parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.TaskSetInfo parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.TaskSetInfo parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.TaskSetInfo parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.TaskSetInfo parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.TaskSetInfo parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.TaskSetInfo parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.TaskSetInfo parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.TaskSetInfo prototype) {
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
   * Protobuf type {@code oneflow.TaskSetInfo}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.TaskSetInfo)
      org.oneflow.core.job.TaskSetInfoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.Task.internal_static_oneflow_TaskSetInfo_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.Task.internal_static_oneflow_TaskSetInfo_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.TaskSetInfo.class, org.oneflow.core.job.TaskSetInfo.Builder.class);
    }

    // Construct using org.oneflow.core.job.TaskSetInfo.newBuilder()
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
      chainId_ = 0L;
      bitField0_ = (bitField0_ & ~0x00000001);
      orderInGraph_ = 0L;
      bitField0_ = (bitField0_ & ~0x00000002);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.Task.internal_static_oneflow_TaskSetInfo_descriptor;
    }

    public org.oneflow.core.job.TaskSetInfo getDefaultInstanceForType() {
      return org.oneflow.core.job.TaskSetInfo.getDefaultInstance();
    }

    public org.oneflow.core.job.TaskSetInfo build() {
      org.oneflow.core.job.TaskSetInfo result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.TaskSetInfo buildPartial() {
      org.oneflow.core.job.TaskSetInfo result = new org.oneflow.core.job.TaskSetInfo(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      result.chainId_ = chainId_;
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000002;
      }
      result.orderInGraph_ = orderInGraph_;
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
      if (other instanceof org.oneflow.core.job.TaskSetInfo) {
        return mergeFrom((org.oneflow.core.job.TaskSetInfo)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.TaskSetInfo other) {
      if (other == org.oneflow.core.job.TaskSetInfo.getDefaultInstance()) return this;
      if (other.hasChainId()) {
        setChainId(other.getChainId());
      }
      if (other.hasOrderInGraph()) {
        setOrderInGraph(other.getOrderInGraph());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (!hasChainId()) {
        return false;
      }
      if (!hasOrderInGraph()) {
        return false;
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.job.TaskSetInfo parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.TaskSetInfo) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private long chainId_ ;
    /**
     * <code>required int64 chain_id = 4;</code>
     */
    public boolean hasChainId() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>required int64 chain_id = 4;</code>
     */
    public long getChainId() {
      return chainId_;
    }
    /**
     * <code>required int64 chain_id = 4;</code>
     */
    public Builder setChainId(long value) {
      bitField0_ |= 0x00000001;
      chainId_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>required int64 chain_id = 4;</code>
     */
    public Builder clearChainId() {
      bitField0_ = (bitField0_ & ~0x00000001);
      chainId_ = 0L;
      onChanged();
      return this;
    }

    private long orderInGraph_ ;
    /**
     * <code>required int64 order_in_graph = 5;</code>
     */
    public boolean hasOrderInGraph() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>required int64 order_in_graph = 5;</code>
     */
    public long getOrderInGraph() {
      return orderInGraph_;
    }
    /**
     * <code>required int64 order_in_graph = 5;</code>
     */
    public Builder setOrderInGraph(long value) {
      bitField0_ |= 0x00000002;
      orderInGraph_ = value;
      onChanged();
      return this;
    }
    /**
     * <code>required int64 order_in_graph = 5;</code>
     */
    public Builder clearOrderInGraph() {
      bitField0_ = (bitField0_ & ~0x00000002);
      orderInGraph_ = 0L;
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


    // @@protoc_insertion_point(builder_scope:oneflow.TaskSetInfo)
  }

  // @@protoc_insertion_point(class_scope:oneflow.TaskSetInfo)
  private static final org.oneflow.core.job.TaskSetInfo DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.TaskSetInfo();
  }

  public static org.oneflow.core.job.TaskSetInfo getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<TaskSetInfo>
      PARSER = new com.google.protobuf.AbstractParser<TaskSetInfo>() {
    public TaskSetInfo parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new TaskSetInfo(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<TaskSetInfo> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<TaskSetInfo> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.TaskSetInfo getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
