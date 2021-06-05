// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/register/logical_blob_id.proto

package org.oneflow.core.register;

/**
 * Protobuf type {@code oneflow.ArgSignature}
 */
public  final class ArgSignature extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.ArgSignature)
    ArgSignatureOrBuilder {
  // Use ArgSignature.newBuilder() to construct.
  private ArgSignature(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private ArgSignature() {
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private ArgSignature(
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
              bnInOp2Lbi_ = com.google.protobuf.MapField.newMapField(
                  BnInOp2LbiDefaultEntryHolder.defaultEntry);
              mutable_bitField0_ |= 0x00000001;
            }
            com.google.protobuf.MapEntry<java.lang.String, org.oneflow.core.register.LogicalBlobId>
            bnInOp2Lbi = input.readMessage(
                BnInOp2LbiDefaultEntryHolder.defaultEntry.getParserForType(), extensionRegistry);
            bnInOp2Lbi_.getMutableMap().put(bnInOp2Lbi.getKey(), bnInOp2Lbi.getValue());
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
    return org.oneflow.core.register.LogicalBlobIdOuterClass.internal_static_oneflow_ArgSignature_descriptor;
  }

  @SuppressWarnings({"rawtypes"})
  protected com.google.protobuf.MapField internalGetMapField(
      int number) {
    switch (number) {
      case 1:
        return internalGetBnInOp2Lbi();
      default:
        throw new RuntimeException(
            "Invalid map field number: " + number);
    }
  }
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.register.LogicalBlobIdOuterClass.internal_static_oneflow_ArgSignature_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.register.ArgSignature.class, org.oneflow.core.register.ArgSignature.Builder.class);
  }

  public static final int BN_IN_OP2LBI_FIELD_NUMBER = 1;
  private static final class BnInOp2LbiDefaultEntryHolder {
    static final com.google.protobuf.MapEntry<
        java.lang.String, org.oneflow.core.register.LogicalBlobId> defaultEntry =
            com.google.protobuf.MapEntry
            .<java.lang.String, org.oneflow.core.register.LogicalBlobId>newDefaultInstance(
                org.oneflow.core.register.LogicalBlobIdOuterClass.internal_static_oneflow_ArgSignature_BnInOp2lbiEntry_descriptor, 
                com.google.protobuf.WireFormat.FieldType.STRING,
                "",
                com.google.protobuf.WireFormat.FieldType.MESSAGE,
                org.oneflow.core.register.LogicalBlobId.getDefaultInstance());
  }
  private com.google.protobuf.MapField<
      java.lang.String, org.oneflow.core.register.LogicalBlobId> bnInOp2Lbi_;
  private com.google.protobuf.MapField<java.lang.String, org.oneflow.core.register.LogicalBlobId>
  internalGetBnInOp2Lbi() {
    if (bnInOp2Lbi_ == null) {
      return com.google.protobuf.MapField.emptyMapField(
          BnInOp2LbiDefaultEntryHolder.defaultEntry);
    }
    return bnInOp2Lbi_;
  }

  public int getBnInOp2LbiCount() {
    return internalGetBnInOp2Lbi().getMap().size();
  }
  /**
   * <code>map&lt;string, .oneflow.LogicalBlobId&gt; bn_in_op2lbi = 1;</code>
   */

  public boolean containsBnInOp2Lbi(
      java.lang.String key) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    return internalGetBnInOp2Lbi().getMap().containsKey(key);
  }
  /**
   * Use {@link #getBnInOp2LbiMap()} instead.
   */
  @java.lang.Deprecated
  public java.util.Map<java.lang.String, org.oneflow.core.register.LogicalBlobId> getBnInOp2Lbi() {
    return getBnInOp2LbiMap();
  }
  /**
   * <code>map&lt;string, .oneflow.LogicalBlobId&gt; bn_in_op2lbi = 1;</code>
   */

  public java.util.Map<java.lang.String, org.oneflow.core.register.LogicalBlobId> getBnInOp2LbiMap() {
    return internalGetBnInOp2Lbi().getMap();
  }
  /**
   * <code>map&lt;string, .oneflow.LogicalBlobId&gt; bn_in_op2lbi = 1;</code>
   */

  public org.oneflow.core.register.LogicalBlobId getBnInOp2LbiOrDefault(
      java.lang.String key,
      org.oneflow.core.register.LogicalBlobId defaultValue) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    java.util.Map<java.lang.String, org.oneflow.core.register.LogicalBlobId> map =
        internalGetBnInOp2Lbi().getMap();
    return map.containsKey(key) ? map.get(key) : defaultValue;
  }
  /**
   * <code>map&lt;string, .oneflow.LogicalBlobId&gt; bn_in_op2lbi = 1;</code>
   */

  public org.oneflow.core.register.LogicalBlobId getBnInOp2LbiOrThrow(
      java.lang.String key) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    java.util.Map<java.lang.String, org.oneflow.core.register.LogicalBlobId> map =
        internalGetBnInOp2Lbi().getMap();
    if (!map.containsKey(key)) {
      throw new java.lang.IllegalArgumentException();
    }
    return map.get(key);
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
    for (java.util.Map.Entry<java.lang.String, org.oneflow.core.register.LogicalBlobId> entry
         : internalGetBnInOp2Lbi().getMap().entrySet()) {
      com.google.protobuf.MapEntry<java.lang.String, org.oneflow.core.register.LogicalBlobId>
      bnInOp2Lbi = BnInOp2LbiDefaultEntryHolder.defaultEntry.newBuilderForType()
          .setKey(entry.getKey())
          .setValue(entry.getValue())
          .build();
      output.writeMessage(1, bnInOp2Lbi);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    for (java.util.Map.Entry<java.lang.String, org.oneflow.core.register.LogicalBlobId> entry
         : internalGetBnInOp2Lbi().getMap().entrySet()) {
      com.google.protobuf.MapEntry<java.lang.String, org.oneflow.core.register.LogicalBlobId>
      bnInOp2Lbi = BnInOp2LbiDefaultEntryHolder.defaultEntry.newBuilderForType()
          .setKey(entry.getKey())
          .setValue(entry.getValue())
          .build();
      size += com.google.protobuf.CodedOutputStream
          .computeMessageSize(1, bnInOp2Lbi);
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
    if (!(obj instanceof org.oneflow.core.register.ArgSignature)) {
      return super.equals(obj);
    }
    org.oneflow.core.register.ArgSignature other = (org.oneflow.core.register.ArgSignature) obj;

    boolean result = true;
    result = result && internalGetBnInOp2Lbi().equals(
        other.internalGetBnInOp2Lbi());
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
    if (!internalGetBnInOp2Lbi().getMap().isEmpty()) {
      hash = (37 * hash) + BN_IN_OP2LBI_FIELD_NUMBER;
      hash = (53 * hash) + internalGetBnInOp2Lbi().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.register.ArgSignature parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.register.ArgSignature parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.register.ArgSignature parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.register.ArgSignature parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.register.ArgSignature parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.ArgSignature parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.register.ArgSignature parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.ArgSignature parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.register.ArgSignature parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.register.ArgSignature parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.register.ArgSignature prototype) {
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
   * Protobuf type {@code oneflow.ArgSignature}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.ArgSignature)
      org.oneflow.core.register.ArgSignatureOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.register.LogicalBlobIdOuterClass.internal_static_oneflow_ArgSignature_descriptor;
    }

    @SuppressWarnings({"rawtypes"})
    protected com.google.protobuf.MapField internalGetMapField(
        int number) {
      switch (number) {
        case 1:
          return internalGetBnInOp2Lbi();
        default:
          throw new RuntimeException(
              "Invalid map field number: " + number);
      }
    }
    @SuppressWarnings({"rawtypes"})
    protected com.google.protobuf.MapField internalGetMutableMapField(
        int number) {
      switch (number) {
        case 1:
          return internalGetMutableBnInOp2Lbi();
        default:
          throw new RuntimeException(
              "Invalid map field number: " + number);
      }
    }
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.register.LogicalBlobIdOuterClass.internal_static_oneflow_ArgSignature_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.register.ArgSignature.class, org.oneflow.core.register.ArgSignature.Builder.class);
    }

    // Construct using org.oneflow.core.register.ArgSignature.newBuilder()
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
      internalGetMutableBnInOp2Lbi().clear();
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.register.LogicalBlobIdOuterClass.internal_static_oneflow_ArgSignature_descriptor;
    }

    public org.oneflow.core.register.ArgSignature getDefaultInstanceForType() {
      return org.oneflow.core.register.ArgSignature.getDefaultInstance();
    }

    public org.oneflow.core.register.ArgSignature build() {
      org.oneflow.core.register.ArgSignature result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.register.ArgSignature buildPartial() {
      org.oneflow.core.register.ArgSignature result = new org.oneflow.core.register.ArgSignature(this);
      int from_bitField0_ = bitField0_;
      result.bnInOp2Lbi_ = internalGetBnInOp2Lbi();
      result.bnInOp2Lbi_.makeImmutable();
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
      if (other instanceof org.oneflow.core.register.ArgSignature) {
        return mergeFrom((org.oneflow.core.register.ArgSignature)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.register.ArgSignature other) {
      if (other == org.oneflow.core.register.ArgSignature.getDefaultInstance()) return this;
      internalGetMutableBnInOp2Lbi().mergeFrom(
          other.internalGetBnInOp2Lbi());
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
      org.oneflow.core.register.ArgSignature parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.register.ArgSignature) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private com.google.protobuf.MapField<
        java.lang.String, org.oneflow.core.register.LogicalBlobId> bnInOp2Lbi_;
    private com.google.protobuf.MapField<java.lang.String, org.oneflow.core.register.LogicalBlobId>
    internalGetBnInOp2Lbi() {
      if (bnInOp2Lbi_ == null) {
        return com.google.protobuf.MapField.emptyMapField(
            BnInOp2LbiDefaultEntryHolder.defaultEntry);
      }
      return bnInOp2Lbi_;
    }
    private com.google.protobuf.MapField<java.lang.String, org.oneflow.core.register.LogicalBlobId>
    internalGetMutableBnInOp2Lbi() {
      onChanged();;
      if (bnInOp2Lbi_ == null) {
        bnInOp2Lbi_ = com.google.protobuf.MapField.newMapField(
            BnInOp2LbiDefaultEntryHolder.defaultEntry);
      }
      if (!bnInOp2Lbi_.isMutable()) {
        bnInOp2Lbi_ = bnInOp2Lbi_.copy();
      }
      return bnInOp2Lbi_;
    }

    public int getBnInOp2LbiCount() {
      return internalGetBnInOp2Lbi().getMap().size();
    }
    /**
     * <code>map&lt;string, .oneflow.LogicalBlobId&gt; bn_in_op2lbi = 1;</code>
     */

    public boolean containsBnInOp2Lbi(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      return internalGetBnInOp2Lbi().getMap().containsKey(key);
    }
    /**
     * Use {@link #getBnInOp2LbiMap()} instead.
     */
    @java.lang.Deprecated
    public java.util.Map<java.lang.String, org.oneflow.core.register.LogicalBlobId> getBnInOp2Lbi() {
      return getBnInOp2LbiMap();
    }
    /**
     * <code>map&lt;string, .oneflow.LogicalBlobId&gt; bn_in_op2lbi = 1;</code>
     */

    public java.util.Map<java.lang.String, org.oneflow.core.register.LogicalBlobId> getBnInOp2LbiMap() {
      return internalGetBnInOp2Lbi().getMap();
    }
    /**
     * <code>map&lt;string, .oneflow.LogicalBlobId&gt; bn_in_op2lbi = 1;</code>
     */

    public org.oneflow.core.register.LogicalBlobId getBnInOp2LbiOrDefault(
        java.lang.String key,
        org.oneflow.core.register.LogicalBlobId defaultValue) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      java.util.Map<java.lang.String, org.oneflow.core.register.LogicalBlobId> map =
          internalGetBnInOp2Lbi().getMap();
      return map.containsKey(key) ? map.get(key) : defaultValue;
    }
    /**
     * <code>map&lt;string, .oneflow.LogicalBlobId&gt; bn_in_op2lbi = 1;</code>
     */

    public org.oneflow.core.register.LogicalBlobId getBnInOp2LbiOrThrow(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      java.util.Map<java.lang.String, org.oneflow.core.register.LogicalBlobId> map =
          internalGetBnInOp2Lbi().getMap();
      if (!map.containsKey(key)) {
        throw new java.lang.IllegalArgumentException();
      }
      return map.get(key);
    }

    public Builder clearBnInOp2Lbi() {
      getMutableBnInOp2Lbi().clear();
      return this;
    }
    /**
     * <code>map&lt;string, .oneflow.LogicalBlobId&gt; bn_in_op2lbi = 1;</code>
     */

    public Builder removeBnInOp2Lbi(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      getMutableBnInOp2Lbi().remove(key);
      return this;
    }
    /**
     * Use alternate mutation accessors instead.
     */
    @java.lang.Deprecated
    public java.util.Map<java.lang.String, org.oneflow.core.register.LogicalBlobId>
    getMutableBnInOp2Lbi() {
      return internalGetMutableBnInOp2Lbi().getMutableMap();
    }
    /**
     * <code>map&lt;string, .oneflow.LogicalBlobId&gt; bn_in_op2lbi = 1;</code>
     */
    public Builder putBnInOp2Lbi(
        java.lang.String key,
        org.oneflow.core.register.LogicalBlobId value) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      if (value == null) { throw new java.lang.NullPointerException(); }
      getMutableBnInOp2Lbi().put(key, value);
      return this;
    }
    /**
     * <code>map&lt;string, .oneflow.LogicalBlobId&gt; bn_in_op2lbi = 1;</code>
     */

    public Builder putAllBnInOp2Lbi(
        java.util.Map<java.lang.String, org.oneflow.core.register.LogicalBlobId> values) {
      getMutableBnInOp2Lbi().putAll(values);
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


    // @@protoc_insertion_point(builder_scope:oneflow.ArgSignature)
  }

  // @@protoc_insertion_point(class_scope:oneflow.ArgSignature)
  private static final org.oneflow.core.register.ArgSignature DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.register.ArgSignature();
  }

  public static org.oneflow.core.register.ArgSignature getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<ArgSignature>
      PARSER = new com.google.protobuf.AbstractParser<ArgSignature>() {
    public ArgSignature parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new ArgSignature(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<ArgSignature> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<ArgSignature> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.register.ArgSignature getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
