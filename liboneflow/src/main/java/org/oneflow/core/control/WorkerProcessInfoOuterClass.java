// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/control/worker_process_info.proto

package org.oneflow.core.control;

public final class WorkerProcessInfoOuterClass {
  private WorkerProcessInfoOuterClass() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  public interface WorkerProcessInfoOrBuilder extends
      // @@protoc_insertion_point(interface_extends:oneflow.WorkerProcessInfo)
      com.google.protobuf.MessageOrBuilder {

    /**
     * <code>required int64 rank = 1;</code>
     */
    boolean hasRank();
    /**
     * <code>required int64 rank = 1;</code>
     */
    long getRank();

    /**
     * <code>required int64 port = 2;</code>
     */
    boolean hasPort();
    /**
     * <code>required int64 port = 2;</code>
     */
    long getPort();

    /**
     * <code>optional string host = 3;</code>
     */
    boolean hasHost();
    /**
     * <code>optional string host = 3;</code>
     */
    java.lang.String getHost();
    /**
     * <code>optional string host = 3;</code>
     */
    com.google.protobuf.ByteString
        getHostBytes();
  }
  /**
   * Protobuf type {@code oneflow.WorkerProcessInfo}
   */
  public  static final class WorkerProcessInfo extends
      com.google.protobuf.GeneratedMessageV3 implements
      // @@protoc_insertion_point(message_implements:oneflow.WorkerProcessInfo)
      WorkerProcessInfoOrBuilder {
    // Use WorkerProcessInfo.newBuilder() to construct.
    private WorkerProcessInfo(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
      super(builder);
    }
    private WorkerProcessInfo() {
      rank_ = 0L;
      port_ = 0L;
      host_ = "";
    }

    @java.lang.Override
    public final com.google.protobuf.UnknownFieldSet
    getUnknownFields() {
      return this.unknownFields;
    }
    private WorkerProcessInfo(
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
              rank_ = input.readInt64();
              break;
            }
            case 16: {
              bitField0_ |= 0x00000002;
              port_ = input.readInt64();
              break;
            }
            case 26: {
              com.google.protobuf.ByteString bs = input.readBytes();
              bitField0_ |= 0x00000004;
              host_ = bs;
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
      return org.oneflow.core.control.WorkerProcessInfoOuterClass.internal_static_oneflow_WorkerProcessInfo_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.control.WorkerProcessInfoOuterClass.internal_static_oneflow_WorkerProcessInfo_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo.class, org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo.Builder.class);
    }

    private int bitField0_;
    public static final int RANK_FIELD_NUMBER = 1;
    private long rank_;
    /**
     * <code>required int64 rank = 1;</code>
     */
    public boolean hasRank() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>required int64 rank = 1;</code>
     */
    public long getRank() {
      return rank_;
    }

    public static final int PORT_FIELD_NUMBER = 2;
    private long port_;
    /**
     * <code>required int64 port = 2;</code>
     */
    public boolean hasPort() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>required int64 port = 2;</code>
     */
    public long getPort() {
      return port_;
    }

    public static final int HOST_FIELD_NUMBER = 3;
    private volatile java.lang.Object host_;
    /**
     * <code>optional string host = 3;</code>
     */
    public boolean hasHost() {
      return ((bitField0_ & 0x00000004) == 0x00000004);
    }
    /**
     * <code>optional string host = 3;</code>
     */
    public java.lang.String getHost() {
      java.lang.Object ref = host_;
      if (ref instanceof java.lang.String) {
        return (java.lang.String) ref;
      } else {
        com.google.protobuf.ByteString bs = 
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        if (bs.isValidUtf8()) {
          host_ = s;
        }
        return s;
      }
    }
    /**
     * <code>optional string host = 3;</code>
     */
    public com.google.protobuf.ByteString
        getHostBytes() {
      java.lang.Object ref = host_;
      if (ref instanceof java.lang.String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        host_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }

    private byte memoizedIsInitialized = -1;
    public final boolean isInitialized() {
      byte isInitialized = memoizedIsInitialized;
      if (isInitialized == 1) return true;
      if (isInitialized == 0) return false;

      if (!hasRank()) {
        memoizedIsInitialized = 0;
        return false;
      }
      if (!hasPort()) {
        memoizedIsInitialized = 0;
        return false;
      }
      memoizedIsInitialized = 1;
      return true;
    }

    public void writeTo(com.google.protobuf.CodedOutputStream output)
                        throws java.io.IOException {
      if (((bitField0_ & 0x00000001) == 0x00000001)) {
        output.writeInt64(1, rank_);
      }
      if (((bitField0_ & 0x00000002) == 0x00000002)) {
        output.writeInt64(2, port_);
      }
      if (((bitField0_ & 0x00000004) == 0x00000004)) {
        com.google.protobuf.GeneratedMessageV3.writeString(output, 3, host_);
      }
      unknownFields.writeTo(output);
    }

    public int getSerializedSize() {
      int size = memoizedSize;
      if (size != -1) return size;

      size = 0;
      if (((bitField0_ & 0x00000001) == 0x00000001)) {
        size += com.google.protobuf.CodedOutputStream
          .computeInt64Size(1, rank_);
      }
      if (((bitField0_ & 0x00000002) == 0x00000002)) {
        size += com.google.protobuf.CodedOutputStream
          .computeInt64Size(2, port_);
      }
      if (((bitField0_ & 0x00000004) == 0x00000004)) {
        size += com.google.protobuf.GeneratedMessageV3.computeStringSize(3, host_);
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
      if (!(obj instanceof org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo)) {
        return super.equals(obj);
      }
      org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo other = (org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo) obj;

      boolean result = true;
      result = result && (hasRank() == other.hasRank());
      if (hasRank()) {
        result = result && (getRank()
            == other.getRank());
      }
      result = result && (hasPort() == other.hasPort());
      if (hasPort()) {
        result = result && (getPort()
            == other.getPort());
      }
      result = result && (hasHost() == other.hasHost());
      if (hasHost()) {
        result = result && getHost()
            .equals(other.getHost());
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
      if (hasRank()) {
        hash = (37 * hash) + RANK_FIELD_NUMBER;
        hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
            getRank());
      }
      if (hasPort()) {
        hash = (37 * hash) + PORT_FIELD_NUMBER;
        hash = (53 * hash) + com.google.protobuf.Internal.hashLong(
            getPort());
      }
      if (hasHost()) {
        hash = (37 * hash) + HOST_FIELD_NUMBER;
        hash = (53 * hash) + getHost().hashCode();
      }
      hash = (29 * hash) + unknownFields.hashCode();
      memoizedHashCode = hash;
      return hash;
    }

    public static org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo parseFrom(
        com.google.protobuf.ByteString data)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data);
    }
    public static org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo parseFrom(
        com.google.protobuf.ByteString data,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data, extensionRegistry);
    }
    public static org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo parseFrom(byte[] data)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data);
    }
    public static org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo parseFrom(
        byte[] data,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data, extensionRegistry);
    }
    public static org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo parseFrom(java.io.InputStream input)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseWithIOException(PARSER, input);
    }
    public static org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo parseFrom(
        java.io.InputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseWithIOException(PARSER, input, extensionRegistry);
    }
    public static org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo parseDelimitedFrom(java.io.InputStream input)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseDelimitedWithIOException(PARSER, input);
    }
    public static org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo parseDelimitedFrom(
        java.io.InputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
    }
    public static org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo parseFrom(
        com.google.protobuf.CodedInputStream input)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseWithIOException(PARSER, input);
    }
    public static org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo parseFrom(
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
    public static Builder newBuilder(org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo prototype) {
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
     * Protobuf type {@code oneflow.WorkerProcessInfo}
     */
    public static final class Builder extends
        com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
        // @@protoc_insertion_point(builder_implements:oneflow.WorkerProcessInfo)
        org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfoOrBuilder {
      public static final com.google.protobuf.Descriptors.Descriptor
          getDescriptor() {
        return org.oneflow.core.control.WorkerProcessInfoOuterClass.internal_static_oneflow_WorkerProcessInfo_descriptor;
      }

      protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
          internalGetFieldAccessorTable() {
        return org.oneflow.core.control.WorkerProcessInfoOuterClass.internal_static_oneflow_WorkerProcessInfo_fieldAccessorTable
            .ensureFieldAccessorsInitialized(
                org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo.class, org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo.Builder.class);
      }

      // Construct using org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo.newBuilder()
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
        rank_ = 0L;
        bitField0_ = (bitField0_ & ~0x00000001);
        port_ = 0L;
        bitField0_ = (bitField0_ & ~0x00000002);
        host_ = "";
        bitField0_ = (bitField0_ & ~0x00000004);
        return this;
      }

      public com.google.protobuf.Descriptors.Descriptor
          getDescriptorForType() {
        return org.oneflow.core.control.WorkerProcessInfoOuterClass.internal_static_oneflow_WorkerProcessInfo_descriptor;
      }

      public org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo getDefaultInstanceForType() {
        return org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo.getDefaultInstance();
      }

      public org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo build() {
        org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo result = buildPartial();
        if (!result.isInitialized()) {
          throw newUninitializedMessageException(result);
        }
        return result;
      }

      public org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo buildPartial() {
        org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo result = new org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo(this);
        int from_bitField0_ = bitField0_;
        int to_bitField0_ = 0;
        if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
          to_bitField0_ |= 0x00000001;
        }
        result.rank_ = rank_;
        if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
          to_bitField0_ |= 0x00000002;
        }
        result.port_ = port_;
        if (((from_bitField0_ & 0x00000004) == 0x00000004)) {
          to_bitField0_ |= 0x00000004;
        }
        result.host_ = host_;
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
        if (other instanceof org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo) {
          return mergeFrom((org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo)other);
        } else {
          super.mergeFrom(other);
          return this;
        }
      }

      public Builder mergeFrom(org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo other) {
        if (other == org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo.getDefaultInstance()) return this;
        if (other.hasRank()) {
          setRank(other.getRank());
        }
        if (other.hasPort()) {
          setPort(other.getPort());
        }
        if (other.hasHost()) {
          bitField0_ |= 0x00000004;
          host_ = other.host_;
          onChanged();
        }
        this.mergeUnknownFields(other.unknownFields);
        onChanged();
        return this;
      }

      public final boolean isInitialized() {
        if (!hasRank()) {
          return false;
        }
        if (!hasPort()) {
          return false;
        }
        return true;
      }

      public Builder mergeFrom(
          com.google.protobuf.CodedInputStream input,
          com.google.protobuf.ExtensionRegistryLite extensionRegistry)
          throws java.io.IOException {
        org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo parsedMessage = null;
        try {
          parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
        } catch (com.google.protobuf.InvalidProtocolBufferException e) {
          parsedMessage = (org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo) e.getUnfinishedMessage();
          throw e.unwrapIOException();
        } finally {
          if (parsedMessage != null) {
            mergeFrom(parsedMessage);
          }
        }
        return this;
      }
      private int bitField0_;

      private long rank_ ;
      /**
       * <code>required int64 rank = 1;</code>
       */
      public boolean hasRank() {
        return ((bitField0_ & 0x00000001) == 0x00000001);
      }
      /**
       * <code>required int64 rank = 1;</code>
       */
      public long getRank() {
        return rank_;
      }
      /**
       * <code>required int64 rank = 1;</code>
       */
      public Builder setRank(long value) {
        bitField0_ |= 0x00000001;
        rank_ = value;
        onChanged();
        return this;
      }
      /**
       * <code>required int64 rank = 1;</code>
       */
      public Builder clearRank() {
        bitField0_ = (bitField0_ & ~0x00000001);
        rank_ = 0L;
        onChanged();
        return this;
      }

      private long port_ ;
      /**
       * <code>required int64 port = 2;</code>
       */
      public boolean hasPort() {
        return ((bitField0_ & 0x00000002) == 0x00000002);
      }
      /**
       * <code>required int64 port = 2;</code>
       */
      public long getPort() {
        return port_;
      }
      /**
       * <code>required int64 port = 2;</code>
       */
      public Builder setPort(long value) {
        bitField0_ |= 0x00000002;
        port_ = value;
        onChanged();
        return this;
      }
      /**
       * <code>required int64 port = 2;</code>
       */
      public Builder clearPort() {
        bitField0_ = (bitField0_ & ~0x00000002);
        port_ = 0L;
        onChanged();
        return this;
      }

      private java.lang.Object host_ = "";
      /**
       * <code>optional string host = 3;</code>
       */
      public boolean hasHost() {
        return ((bitField0_ & 0x00000004) == 0x00000004);
      }
      /**
       * <code>optional string host = 3;</code>
       */
      public java.lang.String getHost() {
        java.lang.Object ref = host_;
        if (!(ref instanceof java.lang.String)) {
          com.google.protobuf.ByteString bs =
              (com.google.protobuf.ByteString) ref;
          java.lang.String s = bs.toStringUtf8();
          if (bs.isValidUtf8()) {
            host_ = s;
          }
          return s;
        } else {
          return (java.lang.String) ref;
        }
      }
      /**
       * <code>optional string host = 3;</code>
       */
      public com.google.protobuf.ByteString
          getHostBytes() {
        java.lang.Object ref = host_;
        if (ref instanceof String) {
          com.google.protobuf.ByteString b = 
              com.google.protobuf.ByteString.copyFromUtf8(
                  (java.lang.String) ref);
          host_ = b;
          return b;
        } else {
          return (com.google.protobuf.ByteString) ref;
        }
      }
      /**
       * <code>optional string host = 3;</code>
       */
      public Builder setHost(
          java.lang.String value) {
        if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000004;
        host_ = value;
        onChanged();
        return this;
      }
      /**
       * <code>optional string host = 3;</code>
       */
      public Builder clearHost() {
        bitField0_ = (bitField0_ & ~0x00000004);
        host_ = getDefaultInstance().getHost();
        onChanged();
        return this;
      }
      /**
       * <code>optional string host = 3;</code>
       */
      public Builder setHostBytes(
          com.google.protobuf.ByteString value) {
        if (value == null) {
    throw new NullPointerException();
  }
  bitField0_ |= 0x00000004;
        host_ = value;
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


      // @@protoc_insertion_point(builder_scope:oneflow.WorkerProcessInfo)
    }

    // @@protoc_insertion_point(class_scope:oneflow.WorkerProcessInfo)
    private static final org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo DEFAULT_INSTANCE;
    static {
      DEFAULT_INSTANCE = new org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo();
    }

    public static org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo getDefaultInstance() {
      return DEFAULT_INSTANCE;
    }

    @java.lang.Deprecated public static final com.google.protobuf.Parser<WorkerProcessInfo>
        PARSER = new com.google.protobuf.AbstractParser<WorkerProcessInfo>() {
      public WorkerProcessInfo parsePartialFrom(
          com.google.protobuf.CodedInputStream input,
          com.google.protobuf.ExtensionRegistryLite extensionRegistry)
          throws com.google.protobuf.InvalidProtocolBufferException {
          return new WorkerProcessInfo(input, extensionRegistry);
      }
    };

    public static com.google.protobuf.Parser<WorkerProcessInfo> parser() {
      return PARSER;
    }

    @java.lang.Override
    public com.google.protobuf.Parser<WorkerProcessInfo> getParserForType() {
      return PARSER;
    }

    public org.oneflow.core.control.WorkerProcessInfoOuterClass.WorkerProcessInfo getDefaultInstanceForType() {
      return DEFAULT_INSTANCE;
    }

  }

  private static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_WorkerProcessInfo_descriptor;
  private static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_WorkerProcessInfo_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n.oneflow/core/control/worker_process_in" +
      "fo.proto\022\007oneflow\"=\n\021WorkerProcessInfo\022\014" +
      "\n\004rank\030\001 \002(\003\022\014\n\004port\030\002 \002(\003\022\014\n\004host\030\003 \001(\t" +
      "B\032\n\030org.oneflow.core.control"
    };
    com.google.protobuf.Descriptors.FileDescriptor.InternalDescriptorAssigner assigner =
        new com.google.protobuf.Descriptors.FileDescriptor.    InternalDescriptorAssigner() {
          public com.google.protobuf.ExtensionRegistry assignDescriptors(
              com.google.protobuf.Descriptors.FileDescriptor root) {
            descriptor = root;
            return null;
          }
        };
    com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
        }, assigner);
    internal_static_oneflow_WorkerProcessInfo_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_oneflow_WorkerProcessInfo_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_WorkerProcessInfo_descriptor,
        new java.lang.String[] { "Rank", "Port", "Host", });
  }

  // @@protoc_insertion_point(outer_class_scope)
}
