// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/common/dtype_signature.proto

package org.oneflow.core.common;

/**
 * Protobuf type {@code oneflow.DTypeSignature}
 */
public  final class DTypeSignature extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.DTypeSignature)
    DTypeSignatureOrBuilder {
  // Use DTypeSignature.newBuilder() to construct.
  private DTypeSignature(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private DTypeSignature() {
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private DTypeSignature(
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
              name2Dtype_ = com.google.protobuf.MapField.newMapField(
                  Name2DtypeDefaultEntryHolder.defaultEntry);
              mutable_bitField0_ |= 0x00000001;
            }
            com.google.protobuf.ByteString bytes = input.readBytes();
            com.google.protobuf.MapEntry<java.lang.String, java.lang.Integer>
            name2Dtype = Name2DtypeDefaultEntryHolder.defaultEntry.getParserForType().parseFrom(bytes);
            if (org.oneflow.core.common.DataType.forNumber(name2Dtype.getValue()) == null) {
              unknownFields.mergeLengthDelimitedField(1, bytes);
            } else {
              name2Dtype_.getMutableMap().put(name2Dtype.getKey(), name2Dtype.getValue());
            }
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
    return org.oneflow.core.common.DtypeSignature.internal_static_oneflow_DTypeSignature_descriptor;
  }

  @SuppressWarnings({"rawtypes"})
  protected com.google.protobuf.MapField internalGetMapField(
      int number) {
    switch (number) {
      case 1:
        return internalGetName2Dtype();
      default:
        throw new RuntimeException(
            "Invalid map field number: " + number);
    }
  }
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.common.DtypeSignature.internal_static_oneflow_DTypeSignature_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.common.DTypeSignature.class, org.oneflow.core.common.DTypeSignature.Builder.class);
  }

  public static final int NAME2DTYPE_FIELD_NUMBER = 1;
  private static final class Name2DtypeDefaultEntryHolder {
    static final com.google.protobuf.MapEntry<
        java.lang.String, java.lang.Integer> defaultEntry =
            com.google.protobuf.MapEntry
            .<java.lang.String, java.lang.Integer>newDefaultInstance(
                org.oneflow.core.common.DtypeSignature.internal_static_oneflow_DTypeSignature_Name2dtypeEntry_descriptor, 
                com.google.protobuf.WireFormat.FieldType.STRING,
                "",
                com.google.protobuf.WireFormat.FieldType.ENUM,
                org.oneflow.core.common.DataType.kInvalidDataType.getNumber());
  }
  private com.google.protobuf.MapField<
      java.lang.String, java.lang.Integer> name2Dtype_;
  private com.google.protobuf.MapField<java.lang.String, java.lang.Integer>
  internalGetName2Dtype() {
    if (name2Dtype_ == null) {
      return com.google.protobuf.MapField.emptyMapField(
          Name2DtypeDefaultEntryHolder.defaultEntry);
    }
    return name2Dtype_;
  }
  private static final
  com.google.protobuf.Internal.MapAdapter.Converter<
      java.lang.Integer, org.oneflow.core.common.DataType> name2DtypeValueConverter =
          com.google.protobuf.Internal.MapAdapter.newEnumConverter(
              org.oneflow.core.common.DataType.internalGetValueMap(),
              org.oneflow.core.common.DataType.kInvalidDataType);

  public int getName2DtypeCount() {
    return internalGetName2Dtype().getMap().size();
  }
  /**
   * <code>map&lt;string, .oneflow.DataType&gt; name2dtype = 1;</code>
   */

  public boolean containsName2Dtype(
      java.lang.String key) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    return internalGetName2Dtype().getMap().containsKey(key);
  }
  /**
   * Use {@link #getName2DtypeMap()} instead.
   */
  @java.lang.Deprecated
  public java.util.Map<java.lang.String, org.oneflow.core.common.DataType>
  getName2Dtype() {
    return getName2DtypeMap();
  }
  /**
   * <code>map&lt;string, .oneflow.DataType&gt; name2dtype = 1;</code>
   */

  public java.util.Map<java.lang.String, org.oneflow.core.common.DataType>
  getName2DtypeMap() {
    return new com.google.protobuf.Internal.MapAdapter<
        java.lang.String, org.oneflow.core.common.DataType, java.lang.Integer>(
            internalGetName2Dtype().getMap(),
            name2DtypeValueConverter);
  }
  /**
   * <code>map&lt;string, .oneflow.DataType&gt; name2dtype = 1;</code>
   */

  public org.oneflow.core.common.DataType getName2DtypeOrDefault(
      java.lang.String key,
      org.oneflow.core.common.DataType defaultValue) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    java.util.Map<java.lang.String, java.lang.Integer> map =
        internalGetName2Dtype().getMap();
    return map.containsKey(key)
           ? name2DtypeValueConverter.doForward(map.get(key))
           : defaultValue;
  }
  /**
   * <code>map&lt;string, .oneflow.DataType&gt; name2dtype = 1;</code>
   */

  public org.oneflow.core.common.DataType getName2DtypeOrThrow(
      java.lang.String key) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    java.util.Map<java.lang.String, java.lang.Integer> map =
        internalGetName2Dtype().getMap();
    if (!map.containsKey(key)) {
      throw new java.lang.IllegalArgumentException();
    }
    return name2DtypeValueConverter.doForward(map.get(key));
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
    for (java.util.Map.Entry<java.lang.String, java.lang.Integer> entry
         : internalGetName2Dtype().getMap().entrySet()) {
      com.google.protobuf.MapEntry<java.lang.String, java.lang.Integer>
      name2Dtype = Name2DtypeDefaultEntryHolder.defaultEntry.newBuilderForType()
          .setKey(entry.getKey())
          .setValue(entry.getValue())
          .build();
      output.writeMessage(1, name2Dtype);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    for (java.util.Map.Entry<java.lang.String, java.lang.Integer> entry
         : internalGetName2Dtype().getMap().entrySet()) {
      com.google.protobuf.MapEntry<java.lang.String, java.lang.Integer>
      name2Dtype = Name2DtypeDefaultEntryHolder.defaultEntry.newBuilderForType()
          .setKey(entry.getKey())
          .setValue(entry.getValue())
          .build();
      size += com.google.protobuf.CodedOutputStream
          .computeMessageSize(1, name2Dtype);
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
    if (!(obj instanceof org.oneflow.core.common.DTypeSignature)) {
      return super.equals(obj);
    }
    org.oneflow.core.common.DTypeSignature other = (org.oneflow.core.common.DTypeSignature) obj;

    boolean result = true;
    result = result && internalGetName2Dtype().equals(
        other.internalGetName2Dtype());
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
    if (!internalGetName2Dtype().getMap().isEmpty()) {
      hash = (37 * hash) + NAME2DTYPE_FIELD_NUMBER;
      hash = (53 * hash) + internalGetName2Dtype().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.common.DTypeSignature parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.common.DTypeSignature parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.common.DTypeSignature parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.common.DTypeSignature parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.common.DTypeSignature parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.common.DTypeSignature parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.common.DTypeSignature parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.common.DTypeSignature parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.common.DTypeSignature parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.common.DTypeSignature parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.common.DTypeSignature prototype) {
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
   * Protobuf type {@code oneflow.DTypeSignature}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.DTypeSignature)
      org.oneflow.core.common.DTypeSignatureOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.common.DtypeSignature.internal_static_oneflow_DTypeSignature_descriptor;
    }

    @SuppressWarnings({"rawtypes"})
    protected com.google.protobuf.MapField internalGetMapField(
        int number) {
      switch (number) {
        case 1:
          return internalGetName2Dtype();
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
          return internalGetMutableName2Dtype();
        default:
          throw new RuntimeException(
              "Invalid map field number: " + number);
      }
    }
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.common.DtypeSignature.internal_static_oneflow_DTypeSignature_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.common.DTypeSignature.class, org.oneflow.core.common.DTypeSignature.Builder.class);
    }

    // Construct using org.oneflow.core.common.DTypeSignature.newBuilder()
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
      internalGetMutableName2Dtype().clear();
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.common.DtypeSignature.internal_static_oneflow_DTypeSignature_descriptor;
    }

    public org.oneflow.core.common.DTypeSignature getDefaultInstanceForType() {
      return org.oneflow.core.common.DTypeSignature.getDefaultInstance();
    }

    public org.oneflow.core.common.DTypeSignature build() {
      org.oneflow.core.common.DTypeSignature result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.common.DTypeSignature buildPartial() {
      org.oneflow.core.common.DTypeSignature result = new org.oneflow.core.common.DTypeSignature(this);
      int from_bitField0_ = bitField0_;
      result.name2Dtype_ = internalGetName2Dtype();
      result.name2Dtype_.makeImmutable();
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
      if (other instanceof org.oneflow.core.common.DTypeSignature) {
        return mergeFrom((org.oneflow.core.common.DTypeSignature)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.common.DTypeSignature other) {
      if (other == org.oneflow.core.common.DTypeSignature.getDefaultInstance()) return this;
      internalGetMutableName2Dtype().mergeFrom(
          other.internalGetName2Dtype());
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
      org.oneflow.core.common.DTypeSignature parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.common.DTypeSignature) e.getUnfinishedMessage();
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
        java.lang.String, java.lang.Integer> name2Dtype_;
    private com.google.protobuf.MapField<java.lang.String, java.lang.Integer>
    internalGetName2Dtype() {
      if (name2Dtype_ == null) {
        return com.google.protobuf.MapField.emptyMapField(
            Name2DtypeDefaultEntryHolder.defaultEntry);
      }
      return name2Dtype_;
    }
    private com.google.protobuf.MapField<java.lang.String, java.lang.Integer>
    internalGetMutableName2Dtype() {
      onChanged();;
      if (name2Dtype_ == null) {
        name2Dtype_ = com.google.protobuf.MapField.newMapField(
            Name2DtypeDefaultEntryHolder.defaultEntry);
      }
      if (!name2Dtype_.isMutable()) {
        name2Dtype_ = name2Dtype_.copy();
      }
      return name2Dtype_;
    }

    public int getName2DtypeCount() {
      return internalGetName2Dtype().getMap().size();
    }
    /**
     * <code>map&lt;string, .oneflow.DataType&gt; name2dtype = 1;</code>
     */

    public boolean containsName2Dtype(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      return internalGetName2Dtype().getMap().containsKey(key);
    }
    /**
     * Use {@link #getName2DtypeMap()} instead.
     */
    @java.lang.Deprecated
    public java.util.Map<java.lang.String, org.oneflow.core.common.DataType>
    getName2Dtype() {
      return getName2DtypeMap();
    }
    /**
     * <code>map&lt;string, .oneflow.DataType&gt; name2dtype = 1;</code>
     */

    public java.util.Map<java.lang.String, org.oneflow.core.common.DataType>
    getName2DtypeMap() {
      return new com.google.protobuf.Internal.MapAdapter<
          java.lang.String, org.oneflow.core.common.DataType, java.lang.Integer>(
              internalGetName2Dtype().getMap(),
              name2DtypeValueConverter);
    }
    /**
     * <code>map&lt;string, .oneflow.DataType&gt; name2dtype = 1;</code>
     */

    public org.oneflow.core.common.DataType getName2DtypeOrDefault(
        java.lang.String key,
        org.oneflow.core.common.DataType defaultValue) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      java.util.Map<java.lang.String, java.lang.Integer> map =
          internalGetName2Dtype().getMap();
      return map.containsKey(key)
             ? name2DtypeValueConverter.doForward(map.get(key))
             : defaultValue;
    }
    /**
     * <code>map&lt;string, .oneflow.DataType&gt; name2dtype = 1;</code>
     */

    public org.oneflow.core.common.DataType getName2DtypeOrThrow(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      java.util.Map<java.lang.String, java.lang.Integer> map =
          internalGetName2Dtype().getMap();
      if (!map.containsKey(key)) {
        throw new java.lang.IllegalArgumentException();
      }
      return name2DtypeValueConverter.doForward(map.get(key));
    }

    public Builder clearName2Dtype() {
      getMutableName2Dtype().clear();
      return this;
    }
    /**
     * <code>map&lt;string, .oneflow.DataType&gt; name2dtype = 1;</code>
     */

    public Builder removeName2Dtype(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      getMutableName2Dtype().remove(key);
      return this;
    }
    /**
     * Use alternate mutation accessors instead.
     */
    @java.lang.Deprecated
    public java.util.Map<java.lang.String, org.oneflow.core.common.DataType>
    getMutableName2Dtype() {
      return new com.google.protobuf.Internal.MapAdapter<
          java.lang.String, org.oneflow.core.common.DataType, java.lang.Integer>(
              internalGetMutableName2Dtype().getMutableMap(),
              name2DtypeValueConverter);
    }
    /**
     * <code>map&lt;string, .oneflow.DataType&gt; name2dtype = 1;</code>
     */
    public Builder putName2Dtype(
        java.lang.String key,
        org.oneflow.core.common.DataType value) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      if (value == null) { throw new java.lang.NullPointerException(); }
      getMutableName2Dtype().put(key, value);
      return this;
    }
    /**
     * <code>map&lt;string, .oneflow.DataType&gt; name2dtype = 1;</code>
     */
    public Builder putAllName2Dtype(
        java.util.Map<java.lang.String, org.oneflow.core.common.DataType> values) {
      getMutableName2Dtype().putAll(values);
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


    // @@protoc_insertion_point(builder_scope:oneflow.DTypeSignature)
  }

  // @@protoc_insertion_point(class_scope:oneflow.DTypeSignature)
  private static final org.oneflow.core.common.DTypeSignature DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.common.DTypeSignature();
  }

  public static org.oneflow.core.common.DTypeSignature getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<DTypeSignature>
      PARSER = new com.google.protobuf.AbstractParser<DTypeSignature>() {
    public DTypeSignature parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new DTypeSignature(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<DTypeSignature> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<DTypeSignature> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.common.DTypeSignature getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
