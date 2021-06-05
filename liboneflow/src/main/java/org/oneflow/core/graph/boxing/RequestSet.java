// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/graph/boxing/collective_boxing.proto

package org.oneflow.core.graph.boxing;

/**
 * Protobuf type {@code oneflow.boxing.collective.RequestSet}
 */
public  final class RequestSet extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.boxing.collective.RequestSet)
    RequestSetOrBuilder {
  // Use RequestSet.newBuilder() to construct.
  private RequestSet(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private RequestSet() {
    request_ = java.util.Collections.emptyList();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private RequestSet(
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
            if (!((mutable_bitField0_ & 0x00000001) == 0x00000001)) {
              request_ = new java.util.ArrayList<org.oneflow.core.graph.boxing.RequestDesc>();
              mutable_bitField0_ |= 0x00000001;
            }
            request_.add(
                input.readMessage(org.oneflow.core.graph.boxing.RequestDesc.PARSER, extensionRegistry));
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
        request_ = java.util.Collections.unmodifiableList(request_);
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.oneflow.core.graph.boxing.CollectiveBoxing.internal_static_oneflow_boxing_collective_RequestSet_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.graph.boxing.CollectiveBoxing.internal_static_oneflow_boxing_collective_RequestSet_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.graph.boxing.RequestSet.class, org.oneflow.core.graph.boxing.RequestSet.Builder.class);
  }

  public static final int REQUEST_FIELD_NUMBER = 1;
  private java.util.List<org.oneflow.core.graph.boxing.RequestDesc> request_;
  /**
   * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
   */
  public java.util.List<org.oneflow.core.graph.boxing.RequestDesc> getRequestList() {
    return request_;
  }
  /**
   * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
   */
  public java.util.List<? extends org.oneflow.core.graph.boxing.RequestDescOrBuilder> 
      getRequestOrBuilderList() {
    return request_;
  }
  /**
   * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
   */
  public int getRequestCount() {
    return request_.size();
  }
  /**
   * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
   */
  public org.oneflow.core.graph.boxing.RequestDesc getRequest(int index) {
    return request_.get(index);
  }
  /**
   * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
   */
  public org.oneflow.core.graph.boxing.RequestDescOrBuilder getRequestOrBuilder(
      int index) {
    return request_.get(index);
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    for (int i = 0; i < getRequestCount(); i++) {
      if (!getRequest(i).isInitialized()) {
        memoizedIsInitialized = 0;
        return false;
      }
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    for (int i = 0; i < request_.size(); i++) {
      output.writeMessage(1, request_.get(i));
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    for (int i = 0; i < request_.size(); i++) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(1, request_.get(i));
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
    if (!(obj instanceof org.oneflow.core.graph.boxing.RequestSet)) {
      return super.equals(obj);
    }
    org.oneflow.core.graph.boxing.RequestSet other = (org.oneflow.core.graph.boxing.RequestSet) obj;

    boolean result = true;
    result = result && getRequestList()
        .equals(other.getRequestList());
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
    if (getRequestCount() > 0) {
      hash = (37 * hash) + REQUEST_FIELD_NUMBER;
      hash = (53 * hash) + getRequestList().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.graph.boxing.RequestSet parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.graph.boxing.RequestSet parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.graph.boxing.RequestSet parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.graph.boxing.RequestSet parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.graph.boxing.RequestSet parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.graph.boxing.RequestSet parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.graph.boxing.RequestSet parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.graph.boxing.RequestSet parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.graph.boxing.RequestSet parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.graph.boxing.RequestSet parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.graph.boxing.RequestSet prototype) {
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
   * Protobuf type {@code oneflow.boxing.collective.RequestSet}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.boxing.collective.RequestSet)
      org.oneflow.core.graph.boxing.RequestSetOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.graph.boxing.CollectiveBoxing.internal_static_oneflow_boxing_collective_RequestSet_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.graph.boxing.CollectiveBoxing.internal_static_oneflow_boxing_collective_RequestSet_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.graph.boxing.RequestSet.class, org.oneflow.core.graph.boxing.RequestSet.Builder.class);
    }

    // Construct using org.oneflow.core.graph.boxing.RequestSet.newBuilder()
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
        getRequestFieldBuilder();
      }
    }
    public Builder clear() {
      super.clear();
      if (requestBuilder_ == null) {
        request_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000001);
      } else {
        requestBuilder_.clear();
      }
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.graph.boxing.CollectiveBoxing.internal_static_oneflow_boxing_collective_RequestSet_descriptor;
    }

    public org.oneflow.core.graph.boxing.RequestSet getDefaultInstanceForType() {
      return org.oneflow.core.graph.boxing.RequestSet.getDefaultInstance();
    }

    public org.oneflow.core.graph.boxing.RequestSet build() {
      org.oneflow.core.graph.boxing.RequestSet result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.graph.boxing.RequestSet buildPartial() {
      org.oneflow.core.graph.boxing.RequestSet result = new org.oneflow.core.graph.boxing.RequestSet(this);
      int from_bitField0_ = bitField0_;
      if (requestBuilder_ == null) {
        if (((bitField0_ & 0x00000001) == 0x00000001)) {
          request_ = java.util.Collections.unmodifiableList(request_);
          bitField0_ = (bitField0_ & ~0x00000001);
        }
        result.request_ = request_;
      } else {
        result.request_ = requestBuilder_.build();
      }
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
      if (other instanceof org.oneflow.core.graph.boxing.RequestSet) {
        return mergeFrom((org.oneflow.core.graph.boxing.RequestSet)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.graph.boxing.RequestSet other) {
      if (other == org.oneflow.core.graph.boxing.RequestSet.getDefaultInstance()) return this;
      if (requestBuilder_ == null) {
        if (!other.request_.isEmpty()) {
          if (request_.isEmpty()) {
            request_ = other.request_;
            bitField0_ = (bitField0_ & ~0x00000001);
          } else {
            ensureRequestIsMutable();
            request_.addAll(other.request_);
          }
          onChanged();
        }
      } else {
        if (!other.request_.isEmpty()) {
          if (requestBuilder_.isEmpty()) {
            requestBuilder_.dispose();
            requestBuilder_ = null;
            request_ = other.request_;
            bitField0_ = (bitField0_ & ~0x00000001);
            requestBuilder_ = 
              com.google.protobuf.GeneratedMessageV3.alwaysUseFieldBuilders ?
                 getRequestFieldBuilder() : null;
          } else {
            requestBuilder_.addAllMessages(other.request_);
          }
        }
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      for (int i = 0; i < getRequestCount(); i++) {
        if (!getRequest(i).isInitialized()) {
          return false;
        }
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.graph.boxing.RequestSet parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.graph.boxing.RequestSet) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private java.util.List<org.oneflow.core.graph.boxing.RequestDesc> request_ =
      java.util.Collections.emptyList();
    private void ensureRequestIsMutable() {
      if (!((bitField0_ & 0x00000001) == 0x00000001)) {
        request_ = new java.util.ArrayList<org.oneflow.core.graph.boxing.RequestDesc>(request_);
        bitField0_ |= 0x00000001;
       }
    }

    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.oneflow.core.graph.boxing.RequestDesc, org.oneflow.core.graph.boxing.RequestDesc.Builder, org.oneflow.core.graph.boxing.RequestDescOrBuilder> requestBuilder_;

    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public java.util.List<org.oneflow.core.graph.boxing.RequestDesc> getRequestList() {
      if (requestBuilder_ == null) {
        return java.util.Collections.unmodifiableList(request_);
      } else {
        return requestBuilder_.getMessageList();
      }
    }
    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public int getRequestCount() {
      if (requestBuilder_ == null) {
        return request_.size();
      } else {
        return requestBuilder_.getCount();
      }
    }
    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public org.oneflow.core.graph.boxing.RequestDesc getRequest(int index) {
      if (requestBuilder_ == null) {
        return request_.get(index);
      } else {
        return requestBuilder_.getMessage(index);
      }
    }
    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public Builder setRequest(
        int index, org.oneflow.core.graph.boxing.RequestDesc value) {
      if (requestBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureRequestIsMutable();
        request_.set(index, value);
        onChanged();
      } else {
        requestBuilder_.setMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public Builder setRequest(
        int index, org.oneflow.core.graph.boxing.RequestDesc.Builder builderForValue) {
      if (requestBuilder_ == null) {
        ensureRequestIsMutable();
        request_.set(index, builderForValue.build());
        onChanged();
      } else {
        requestBuilder_.setMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public Builder addRequest(org.oneflow.core.graph.boxing.RequestDesc value) {
      if (requestBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureRequestIsMutable();
        request_.add(value);
        onChanged();
      } else {
        requestBuilder_.addMessage(value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public Builder addRequest(
        int index, org.oneflow.core.graph.boxing.RequestDesc value) {
      if (requestBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureRequestIsMutable();
        request_.add(index, value);
        onChanged();
      } else {
        requestBuilder_.addMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public Builder addRequest(
        org.oneflow.core.graph.boxing.RequestDesc.Builder builderForValue) {
      if (requestBuilder_ == null) {
        ensureRequestIsMutable();
        request_.add(builderForValue.build());
        onChanged();
      } else {
        requestBuilder_.addMessage(builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public Builder addRequest(
        int index, org.oneflow.core.graph.boxing.RequestDesc.Builder builderForValue) {
      if (requestBuilder_ == null) {
        ensureRequestIsMutable();
        request_.add(index, builderForValue.build());
        onChanged();
      } else {
        requestBuilder_.addMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public Builder addAllRequest(
        java.lang.Iterable<? extends org.oneflow.core.graph.boxing.RequestDesc> values) {
      if (requestBuilder_ == null) {
        ensureRequestIsMutable();
        com.google.protobuf.AbstractMessageLite.Builder.addAll(
            values, request_);
        onChanged();
      } else {
        requestBuilder_.addAllMessages(values);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public Builder clearRequest() {
      if (requestBuilder_ == null) {
        request_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000001);
        onChanged();
      } else {
        requestBuilder_.clear();
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public Builder removeRequest(int index) {
      if (requestBuilder_ == null) {
        ensureRequestIsMutable();
        request_.remove(index);
        onChanged();
      } else {
        requestBuilder_.remove(index);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public org.oneflow.core.graph.boxing.RequestDesc.Builder getRequestBuilder(
        int index) {
      return getRequestFieldBuilder().getBuilder(index);
    }
    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public org.oneflow.core.graph.boxing.RequestDescOrBuilder getRequestOrBuilder(
        int index) {
      if (requestBuilder_ == null) {
        return request_.get(index);  } else {
        return requestBuilder_.getMessageOrBuilder(index);
      }
    }
    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public java.util.List<? extends org.oneflow.core.graph.boxing.RequestDescOrBuilder> 
         getRequestOrBuilderList() {
      if (requestBuilder_ != null) {
        return requestBuilder_.getMessageOrBuilderList();
      } else {
        return java.util.Collections.unmodifiableList(request_);
      }
    }
    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public org.oneflow.core.graph.boxing.RequestDesc.Builder addRequestBuilder() {
      return getRequestFieldBuilder().addBuilder(
          org.oneflow.core.graph.boxing.RequestDesc.getDefaultInstance());
    }
    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public org.oneflow.core.graph.boxing.RequestDesc.Builder addRequestBuilder(
        int index) {
      return getRequestFieldBuilder().addBuilder(
          index, org.oneflow.core.graph.boxing.RequestDesc.getDefaultInstance());
    }
    /**
     * <code>repeated .oneflow.boxing.collective.RequestDesc request = 1;</code>
     */
    public java.util.List<org.oneflow.core.graph.boxing.RequestDesc.Builder> 
         getRequestBuilderList() {
      return getRequestFieldBuilder().getBuilderList();
    }
    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.oneflow.core.graph.boxing.RequestDesc, org.oneflow.core.graph.boxing.RequestDesc.Builder, org.oneflow.core.graph.boxing.RequestDescOrBuilder> 
        getRequestFieldBuilder() {
      if (requestBuilder_ == null) {
        requestBuilder_ = new com.google.protobuf.RepeatedFieldBuilderV3<
            org.oneflow.core.graph.boxing.RequestDesc, org.oneflow.core.graph.boxing.RequestDesc.Builder, org.oneflow.core.graph.boxing.RequestDescOrBuilder>(
                request_,
                ((bitField0_ & 0x00000001) == 0x00000001),
                getParentForChildren(),
                isClean());
        request_ = null;
      }
      return requestBuilder_;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:oneflow.boxing.collective.RequestSet)
  }

  // @@protoc_insertion_point(class_scope:oneflow.boxing.collective.RequestSet)
  private static final org.oneflow.core.graph.boxing.RequestSet DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.graph.boxing.RequestSet();
  }

  public static org.oneflow.core.graph.boxing.RequestSet getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<RequestSet>
      PARSER = new com.google.protobuf.AbstractParser<RequestSet>() {
    public RequestSet parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new RequestSet(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<RequestSet> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<RequestSet> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.graph.boxing.RequestSet getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
