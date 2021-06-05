// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/control/control.proto

package org.oneflow.core.control;

/**
 * Protobuf type {@code oneflow.PushActEventRequest}
 */
public  final class PushActEventRequest extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.PushActEventRequest)
    PushActEventRequestOrBuilder {
  // Use PushActEventRequest.newBuilder() to construct.
  private PushActEventRequest(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private PushActEventRequest() {
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private PushActEventRequest(
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
            oneflow.ActEventOuterClass.ActEvent.Builder subBuilder = null;
            if (((bitField0_ & 0x00000001) == 0x00000001)) {
              subBuilder = actEvent_.toBuilder();
            }
            actEvent_ = input.readMessage(oneflow.ActEventOuterClass.ActEvent.PARSER, extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(actEvent_);
              actEvent_ = subBuilder.buildPartial();
            }
            bitField0_ |= 0x00000001;
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
    return org.oneflow.core.control.Control.internal_static_oneflow_PushActEventRequest_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.control.Control.internal_static_oneflow_PushActEventRequest_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.control.PushActEventRequest.class, org.oneflow.core.control.PushActEventRequest.Builder.class);
  }

  private int bitField0_;
  public static final int ACT_EVENT_FIELD_NUMBER = 1;
  private oneflow.ActEventOuterClass.ActEvent actEvent_;
  /**
   * <code>required .oneflow.ActEvent act_event = 1;</code>
   */
  public boolean hasActEvent() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>required .oneflow.ActEvent act_event = 1;</code>
   */
  public oneflow.ActEventOuterClass.ActEvent getActEvent() {
    return actEvent_ == null ? oneflow.ActEventOuterClass.ActEvent.getDefaultInstance() : actEvent_;
  }
  /**
   * <code>required .oneflow.ActEvent act_event = 1;</code>
   */
  public oneflow.ActEventOuterClass.ActEventOrBuilder getActEventOrBuilder() {
    return actEvent_ == null ? oneflow.ActEventOuterClass.ActEvent.getDefaultInstance() : actEvent_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (!hasActEvent()) {
      memoizedIsInitialized = 0;
      return false;
    }
    if (!getActEvent().isInitialized()) {
      memoizedIsInitialized = 0;
      return false;
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      output.writeMessage(1, getActEvent());
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(1, getActEvent());
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
    if (!(obj instanceof org.oneflow.core.control.PushActEventRequest)) {
      return super.equals(obj);
    }
    org.oneflow.core.control.PushActEventRequest other = (org.oneflow.core.control.PushActEventRequest) obj;

    boolean result = true;
    result = result && (hasActEvent() == other.hasActEvent());
    if (hasActEvent()) {
      result = result && getActEvent()
          .equals(other.getActEvent());
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
    if (hasActEvent()) {
      hash = (37 * hash) + ACT_EVENT_FIELD_NUMBER;
      hash = (53 * hash) + getActEvent().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.control.PushActEventRequest parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.control.PushActEventRequest parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.control.PushActEventRequest parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.control.PushActEventRequest parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.control.PushActEventRequest parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.control.PushActEventRequest parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.control.PushActEventRequest parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.control.PushActEventRequest parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.control.PushActEventRequest parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.control.PushActEventRequest parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.control.PushActEventRequest prototype) {
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
   * Protobuf type {@code oneflow.PushActEventRequest}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.PushActEventRequest)
      org.oneflow.core.control.PushActEventRequestOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.control.Control.internal_static_oneflow_PushActEventRequest_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.control.Control.internal_static_oneflow_PushActEventRequest_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.control.PushActEventRequest.class, org.oneflow.core.control.PushActEventRequest.Builder.class);
    }

    // Construct using org.oneflow.core.control.PushActEventRequest.newBuilder()
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
        getActEventFieldBuilder();
      }
    }
    public Builder clear() {
      super.clear();
      if (actEventBuilder_ == null) {
        actEvent_ = null;
      } else {
        actEventBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000001);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.control.Control.internal_static_oneflow_PushActEventRequest_descriptor;
    }

    public org.oneflow.core.control.PushActEventRequest getDefaultInstanceForType() {
      return org.oneflow.core.control.PushActEventRequest.getDefaultInstance();
    }

    public org.oneflow.core.control.PushActEventRequest build() {
      org.oneflow.core.control.PushActEventRequest result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.control.PushActEventRequest buildPartial() {
      org.oneflow.core.control.PushActEventRequest result = new org.oneflow.core.control.PushActEventRequest(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      if (actEventBuilder_ == null) {
        result.actEvent_ = actEvent_;
      } else {
        result.actEvent_ = actEventBuilder_.build();
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
      if (other instanceof org.oneflow.core.control.PushActEventRequest) {
        return mergeFrom((org.oneflow.core.control.PushActEventRequest)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.control.PushActEventRequest other) {
      if (other == org.oneflow.core.control.PushActEventRequest.getDefaultInstance()) return this;
      if (other.hasActEvent()) {
        mergeActEvent(other.getActEvent());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (!hasActEvent()) {
        return false;
      }
      if (!getActEvent().isInitialized()) {
        return false;
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.control.PushActEventRequest parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.control.PushActEventRequest) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private oneflow.ActEventOuterClass.ActEvent actEvent_ = null;
    private com.google.protobuf.SingleFieldBuilderV3<
        oneflow.ActEventOuterClass.ActEvent, oneflow.ActEventOuterClass.ActEvent.Builder, oneflow.ActEventOuterClass.ActEventOrBuilder> actEventBuilder_;
    /**
     * <code>required .oneflow.ActEvent act_event = 1;</code>
     */
    public boolean hasActEvent() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>required .oneflow.ActEvent act_event = 1;</code>
     */
    public oneflow.ActEventOuterClass.ActEvent getActEvent() {
      if (actEventBuilder_ == null) {
        return actEvent_ == null ? oneflow.ActEventOuterClass.ActEvent.getDefaultInstance() : actEvent_;
      } else {
        return actEventBuilder_.getMessage();
      }
    }
    /**
     * <code>required .oneflow.ActEvent act_event = 1;</code>
     */
    public Builder setActEvent(oneflow.ActEventOuterClass.ActEvent value) {
      if (actEventBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        actEvent_ = value;
        onChanged();
      } else {
        actEventBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>required .oneflow.ActEvent act_event = 1;</code>
     */
    public Builder setActEvent(
        oneflow.ActEventOuterClass.ActEvent.Builder builderForValue) {
      if (actEventBuilder_ == null) {
        actEvent_ = builderForValue.build();
        onChanged();
      } else {
        actEventBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>required .oneflow.ActEvent act_event = 1;</code>
     */
    public Builder mergeActEvent(oneflow.ActEventOuterClass.ActEvent value) {
      if (actEventBuilder_ == null) {
        if (((bitField0_ & 0x00000001) == 0x00000001) &&
            actEvent_ != null &&
            actEvent_ != oneflow.ActEventOuterClass.ActEvent.getDefaultInstance()) {
          actEvent_ =
            oneflow.ActEventOuterClass.ActEvent.newBuilder(actEvent_).mergeFrom(value).buildPartial();
        } else {
          actEvent_ = value;
        }
        onChanged();
      } else {
        actEventBuilder_.mergeFrom(value);
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>required .oneflow.ActEvent act_event = 1;</code>
     */
    public Builder clearActEvent() {
      if (actEventBuilder_ == null) {
        actEvent_ = null;
        onChanged();
      } else {
        actEventBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000001);
      return this;
    }
    /**
     * <code>required .oneflow.ActEvent act_event = 1;</code>
     */
    public oneflow.ActEventOuterClass.ActEvent.Builder getActEventBuilder() {
      bitField0_ |= 0x00000001;
      onChanged();
      return getActEventFieldBuilder().getBuilder();
    }
    /**
     * <code>required .oneflow.ActEvent act_event = 1;</code>
     */
    public oneflow.ActEventOuterClass.ActEventOrBuilder getActEventOrBuilder() {
      if (actEventBuilder_ != null) {
        return actEventBuilder_.getMessageOrBuilder();
      } else {
        return actEvent_ == null ?
            oneflow.ActEventOuterClass.ActEvent.getDefaultInstance() : actEvent_;
      }
    }
    /**
     * <code>required .oneflow.ActEvent act_event = 1;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        oneflow.ActEventOuterClass.ActEvent, oneflow.ActEventOuterClass.ActEvent.Builder, oneflow.ActEventOuterClass.ActEventOrBuilder> 
        getActEventFieldBuilder() {
      if (actEventBuilder_ == null) {
        actEventBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            oneflow.ActEventOuterClass.ActEvent, oneflow.ActEventOuterClass.ActEvent.Builder, oneflow.ActEventOuterClass.ActEventOrBuilder>(
                getActEvent(),
                getParentForChildren(),
                isClean());
        actEvent_ = null;
      }
      return actEventBuilder_;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:oneflow.PushActEventRequest)
  }

  // @@protoc_insertion_point(class_scope:oneflow.PushActEventRequest)
  private static final org.oneflow.core.control.PushActEventRequest DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.control.PushActEventRequest();
  }

  public static org.oneflow.core.control.PushActEventRequest getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<PushActEventRequest>
      PARSER = new com.google.protobuf.AbstractParser<PushActEventRequest>() {
    public PushActEventRequest parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new PushActEventRequest(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<PushActEventRequest> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<PushActEventRequest> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.control.PushActEventRequest getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

