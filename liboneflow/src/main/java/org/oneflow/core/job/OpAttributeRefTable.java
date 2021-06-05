// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/plan.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.OpAttributeRefTable}
 */
public  final class OpAttributeRefTable extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.OpAttributeRefTable)
    OpAttributeRefTableOrBuilder {
  // Use OpAttributeRefTable.newBuilder() to construct.
  private OpAttributeRefTable(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private OpAttributeRefTable() {
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private OpAttributeRefTable(
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
              opName2OpAttribute_ = com.google.protobuf.MapField.newMapField(
                  OpName2OpAttributeDefaultEntryHolder.defaultEntry);
              mutable_bitField0_ |= 0x00000001;
            }
            com.google.protobuf.MapEntry<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute>
            opName2OpAttribute = input.readMessage(
                OpName2OpAttributeDefaultEntryHolder.defaultEntry.getParserForType(), extensionRegistry);
            opName2OpAttribute_.getMutableMap().put(opName2OpAttribute.getKey(), opName2OpAttribute.getValue());
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
    return org.oneflow.core.job.PlanOuterClass.internal_static_oneflow_OpAttributeRefTable_descriptor;
  }

  @SuppressWarnings({"rawtypes"})
  protected com.google.protobuf.MapField internalGetMapField(
      int number) {
    switch (number) {
      case 1:
        return internalGetOpName2OpAttribute();
      default:
        throw new RuntimeException(
            "Invalid map field number: " + number);
    }
  }
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.PlanOuterClass.internal_static_oneflow_OpAttributeRefTable_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.OpAttributeRefTable.class, org.oneflow.core.job.OpAttributeRefTable.Builder.class);
  }

  public static final int OP_NAME2OP_ATTRIBUTE_FIELD_NUMBER = 1;
  private static final class OpName2OpAttributeDefaultEntryHolder {
    static final com.google.protobuf.MapEntry<
        java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute> defaultEntry =
            com.google.protobuf.MapEntry
            .<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute>newDefaultInstance(
                org.oneflow.core.job.PlanOuterClass.internal_static_oneflow_OpAttributeRefTable_OpName2opAttributeEntry_descriptor, 
                com.google.protobuf.WireFormat.FieldType.STRING,
                "",
                com.google.protobuf.WireFormat.FieldType.MESSAGE,
                oneflow.OpAttributeOuterClass.OpAttribute.getDefaultInstance());
  }
  private com.google.protobuf.MapField<
      java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute> opName2OpAttribute_;
  private com.google.protobuf.MapField<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute>
  internalGetOpName2OpAttribute() {
    if (opName2OpAttribute_ == null) {
      return com.google.protobuf.MapField.emptyMapField(
          OpName2OpAttributeDefaultEntryHolder.defaultEntry);
    }
    return opName2OpAttribute_;
  }

  public int getOpName2OpAttributeCount() {
    return internalGetOpName2OpAttribute().getMap().size();
  }
  /**
   * <code>map&lt;string, .oneflow.OpAttribute&gt; op_name2op_attribute = 1;</code>
   */

  public boolean containsOpName2OpAttribute(
      java.lang.String key) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    return internalGetOpName2OpAttribute().getMap().containsKey(key);
  }
  /**
   * Use {@link #getOpName2OpAttributeMap()} instead.
   */
  @java.lang.Deprecated
  public java.util.Map<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute> getOpName2OpAttribute() {
    return getOpName2OpAttributeMap();
  }
  /**
   * <code>map&lt;string, .oneflow.OpAttribute&gt; op_name2op_attribute = 1;</code>
   */

  public java.util.Map<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute> getOpName2OpAttributeMap() {
    return internalGetOpName2OpAttribute().getMap();
  }
  /**
   * <code>map&lt;string, .oneflow.OpAttribute&gt; op_name2op_attribute = 1;</code>
   */

  public oneflow.OpAttributeOuterClass.OpAttribute getOpName2OpAttributeOrDefault(
      java.lang.String key,
      oneflow.OpAttributeOuterClass.OpAttribute defaultValue) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    java.util.Map<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute> map =
        internalGetOpName2OpAttribute().getMap();
    return map.containsKey(key) ? map.get(key) : defaultValue;
  }
  /**
   * <code>map&lt;string, .oneflow.OpAttribute&gt; op_name2op_attribute = 1;</code>
   */

  public oneflow.OpAttributeOuterClass.OpAttribute getOpName2OpAttributeOrThrow(
      java.lang.String key) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    java.util.Map<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute> map =
        internalGetOpName2OpAttribute().getMap();
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

    for (oneflow.OpAttributeOuterClass.OpAttribute item : getOpName2OpAttribute().values()) {
      if (!item.isInitialized()) {
        memoizedIsInitialized = 0;
        return false;
      }
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    for (java.util.Map.Entry<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute> entry
         : internalGetOpName2OpAttribute().getMap().entrySet()) {
      com.google.protobuf.MapEntry<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute>
      opName2OpAttribute = OpName2OpAttributeDefaultEntryHolder.defaultEntry.newBuilderForType()
          .setKey(entry.getKey())
          .setValue(entry.getValue())
          .build();
      output.writeMessage(1, opName2OpAttribute);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    for (java.util.Map.Entry<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute> entry
         : internalGetOpName2OpAttribute().getMap().entrySet()) {
      com.google.protobuf.MapEntry<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute>
      opName2OpAttribute = OpName2OpAttributeDefaultEntryHolder.defaultEntry.newBuilderForType()
          .setKey(entry.getKey())
          .setValue(entry.getValue())
          .build();
      size += com.google.protobuf.CodedOutputStream
          .computeMessageSize(1, opName2OpAttribute);
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
    if (!(obj instanceof org.oneflow.core.job.OpAttributeRefTable)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.OpAttributeRefTable other = (org.oneflow.core.job.OpAttributeRefTable) obj;

    boolean result = true;
    result = result && internalGetOpName2OpAttribute().equals(
        other.internalGetOpName2OpAttribute());
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
    if (!internalGetOpName2OpAttribute().getMap().isEmpty()) {
      hash = (37 * hash) + OP_NAME2OP_ATTRIBUTE_FIELD_NUMBER;
      hash = (53 * hash) + internalGetOpName2OpAttribute().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.OpAttributeRefTable parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.OpAttributeRefTable parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.OpAttributeRefTable parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.OpAttributeRefTable parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.OpAttributeRefTable parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.OpAttributeRefTable parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.OpAttributeRefTable parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.OpAttributeRefTable parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.OpAttributeRefTable parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.OpAttributeRefTable parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.OpAttributeRefTable prototype) {
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
   * Protobuf type {@code oneflow.OpAttributeRefTable}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.OpAttributeRefTable)
      org.oneflow.core.job.OpAttributeRefTableOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.PlanOuterClass.internal_static_oneflow_OpAttributeRefTable_descriptor;
    }

    @SuppressWarnings({"rawtypes"})
    protected com.google.protobuf.MapField internalGetMapField(
        int number) {
      switch (number) {
        case 1:
          return internalGetOpName2OpAttribute();
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
          return internalGetMutableOpName2OpAttribute();
        default:
          throw new RuntimeException(
              "Invalid map field number: " + number);
      }
    }
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.PlanOuterClass.internal_static_oneflow_OpAttributeRefTable_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.OpAttributeRefTable.class, org.oneflow.core.job.OpAttributeRefTable.Builder.class);
    }

    // Construct using org.oneflow.core.job.OpAttributeRefTable.newBuilder()
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
      internalGetMutableOpName2OpAttribute().clear();
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.PlanOuterClass.internal_static_oneflow_OpAttributeRefTable_descriptor;
    }

    public org.oneflow.core.job.OpAttributeRefTable getDefaultInstanceForType() {
      return org.oneflow.core.job.OpAttributeRefTable.getDefaultInstance();
    }

    public org.oneflow.core.job.OpAttributeRefTable build() {
      org.oneflow.core.job.OpAttributeRefTable result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.OpAttributeRefTable buildPartial() {
      org.oneflow.core.job.OpAttributeRefTable result = new org.oneflow.core.job.OpAttributeRefTable(this);
      int from_bitField0_ = bitField0_;
      result.opName2OpAttribute_ = internalGetOpName2OpAttribute();
      result.opName2OpAttribute_.makeImmutable();
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
      if (other instanceof org.oneflow.core.job.OpAttributeRefTable) {
        return mergeFrom((org.oneflow.core.job.OpAttributeRefTable)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.OpAttributeRefTable other) {
      if (other == org.oneflow.core.job.OpAttributeRefTable.getDefaultInstance()) return this;
      internalGetMutableOpName2OpAttribute().mergeFrom(
          other.internalGetOpName2OpAttribute());
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      for (oneflow.OpAttributeOuterClass.OpAttribute item : getOpName2OpAttribute().values()) {
        if (!item.isInitialized()) {
          return false;
        }
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.job.OpAttributeRefTable parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.OpAttributeRefTable) e.getUnfinishedMessage();
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
        java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute> opName2OpAttribute_;
    private com.google.protobuf.MapField<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute>
    internalGetOpName2OpAttribute() {
      if (opName2OpAttribute_ == null) {
        return com.google.protobuf.MapField.emptyMapField(
            OpName2OpAttributeDefaultEntryHolder.defaultEntry);
      }
      return opName2OpAttribute_;
    }
    private com.google.protobuf.MapField<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute>
    internalGetMutableOpName2OpAttribute() {
      onChanged();;
      if (opName2OpAttribute_ == null) {
        opName2OpAttribute_ = com.google.protobuf.MapField.newMapField(
            OpName2OpAttributeDefaultEntryHolder.defaultEntry);
      }
      if (!opName2OpAttribute_.isMutable()) {
        opName2OpAttribute_ = opName2OpAttribute_.copy();
      }
      return opName2OpAttribute_;
    }

    public int getOpName2OpAttributeCount() {
      return internalGetOpName2OpAttribute().getMap().size();
    }
    /**
     * <code>map&lt;string, .oneflow.OpAttribute&gt; op_name2op_attribute = 1;</code>
     */

    public boolean containsOpName2OpAttribute(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      return internalGetOpName2OpAttribute().getMap().containsKey(key);
    }
    /**
     * Use {@link #getOpName2OpAttributeMap()} instead.
     */
    @java.lang.Deprecated
    public java.util.Map<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute> getOpName2OpAttribute() {
      return getOpName2OpAttributeMap();
    }
    /**
     * <code>map&lt;string, .oneflow.OpAttribute&gt; op_name2op_attribute = 1;</code>
     */

    public java.util.Map<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute> getOpName2OpAttributeMap() {
      return internalGetOpName2OpAttribute().getMap();
    }
    /**
     * <code>map&lt;string, .oneflow.OpAttribute&gt; op_name2op_attribute = 1;</code>
     */

    public oneflow.OpAttributeOuterClass.OpAttribute getOpName2OpAttributeOrDefault(
        java.lang.String key,
        oneflow.OpAttributeOuterClass.OpAttribute defaultValue) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      java.util.Map<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute> map =
          internalGetOpName2OpAttribute().getMap();
      return map.containsKey(key) ? map.get(key) : defaultValue;
    }
    /**
     * <code>map&lt;string, .oneflow.OpAttribute&gt; op_name2op_attribute = 1;</code>
     */

    public oneflow.OpAttributeOuterClass.OpAttribute getOpName2OpAttributeOrThrow(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      java.util.Map<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute> map =
          internalGetOpName2OpAttribute().getMap();
      if (!map.containsKey(key)) {
        throw new java.lang.IllegalArgumentException();
      }
      return map.get(key);
    }

    public Builder clearOpName2OpAttribute() {
      getMutableOpName2OpAttribute().clear();
      return this;
    }
    /**
     * <code>map&lt;string, .oneflow.OpAttribute&gt; op_name2op_attribute = 1;</code>
     */

    public Builder removeOpName2OpAttribute(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      getMutableOpName2OpAttribute().remove(key);
      return this;
    }
    /**
     * Use alternate mutation accessors instead.
     */
    @java.lang.Deprecated
    public java.util.Map<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute>
    getMutableOpName2OpAttribute() {
      return internalGetMutableOpName2OpAttribute().getMutableMap();
    }
    /**
     * <code>map&lt;string, .oneflow.OpAttribute&gt; op_name2op_attribute = 1;</code>
     */
    public Builder putOpName2OpAttribute(
        java.lang.String key,
        oneflow.OpAttributeOuterClass.OpAttribute value) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      if (value == null) { throw new java.lang.NullPointerException(); }
      getMutableOpName2OpAttribute().put(key, value);
      return this;
    }
    /**
     * <code>map&lt;string, .oneflow.OpAttribute&gt; op_name2op_attribute = 1;</code>
     */

    public Builder putAllOpName2OpAttribute(
        java.util.Map<java.lang.String, oneflow.OpAttributeOuterClass.OpAttribute> values) {
      getMutableOpName2OpAttribute().putAll(values);
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


    // @@protoc_insertion_point(builder_scope:oneflow.OpAttributeRefTable)
  }

  // @@protoc_insertion_point(class_scope:oneflow.OpAttributeRefTable)
  private static final org.oneflow.core.job.OpAttributeRefTable DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.OpAttributeRefTable();
  }

  public static org.oneflow.core.job.OpAttributeRefTable getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<OpAttributeRefTable>
      PARSER = new com.google.protobuf.AbstractParser<OpAttributeRefTable>() {
    public OpAttributeRefTable parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new OpAttributeRefTable(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<OpAttributeRefTable> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<OpAttributeRefTable> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.OpAttributeRefTable getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
