// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/sbp_parallel.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.ParallelDistributionSignature}
 */
public  final class ParallelDistributionSignature extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.ParallelDistributionSignature)
    ParallelDistributionSignatureOrBuilder {
  // Use ParallelDistributionSignature.newBuilder() to construct.
  private ParallelDistributionSignature(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private ParallelDistributionSignature() {
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private ParallelDistributionSignature(
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
              bnInOp2ParallelDistribution_ = com.google.protobuf.MapField.newMapField(
                  BnInOp2ParallelDistributionDefaultEntryHolder.defaultEntry);
              mutable_bitField0_ |= 0x00000001;
            }
            com.google.protobuf.MapEntry<java.lang.String, org.oneflow.core.job.ParallelDistribution>
            bnInOp2ParallelDistribution = input.readMessage(
                BnInOp2ParallelDistributionDefaultEntryHolder.defaultEntry.getParserForType(), extensionRegistry);
            bnInOp2ParallelDistribution_.getMutableMap().put(bnInOp2ParallelDistribution.getKey(), bnInOp2ParallelDistribution.getValue());
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
    return org.oneflow.core.job.SbpParallelOuterClass.internal_static_oneflow_ParallelDistributionSignature_descriptor;
  }

  @SuppressWarnings({"rawtypes"})
  protected com.google.protobuf.MapField internalGetMapField(
      int number) {
    switch (number) {
      case 1:
        return internalGetBnInOp2ParallelDistribution();
      default:
        throw new RuntimeException(
            "Invalid map field number: " + number);
    }
  }
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.SbpParallelOuterClass.internal_static_oneflow_ParallelDistributionSignature_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.ParallelDistributionSignature.class, org.oneflow.core.job.ParallelDistributionSignature.Builder.class);
  }

  public static final int BN_IN_OP2PARALLEL_DISTRIBUTION_FIELD_NUMBER = 1;
  private static final class BnInOp2ParallelDistributionDefaultEntryHolder {
    static final com.google.protobuf.MapEntry<
        java.lang.String, org.oneflow.core.job.ParallelDistribution> defaultEntry =
            com.google.protobuf.MapEntry
            .<java.lang.String, org.oneflow.core.job.ParallelDistribution>newDefaultInstance(
                org.oneflow.core.job.SbpParallelOuterClass.internal_static_oneflow_ParallelDistributionSignature_BnInOp2parallelDistributionEntry_descriptor, 
                com.google.protobuf.WireFormat.FieldType.STRING,
                "",
                com.google.protobuf.WireFormat.FieldType.MESSAGE,
                org.oneflow.core.job.ParallelDistribution.getDefaultInstance());
  }
  private com.google.protobuf.MapField<
      java.lang.String, org.oneflow.core.job.ParallelDistribution> bnInOp2ParallelDistribution_;
  private com.google.protobuf.MapField<java.lang.String, org.oneflow.core.job.ParallelDistribution>
  internalGetBnInOp2ParallelDistribution() {
    if (bnInOp2ParallelDistribution_ == null) {
      return com.google.protobuf.MapField.emptyMapField(
          BnInOp2ParallelDistributionDefaultEntryHolder.defaultEntry);
    }
    return bnInOp2ParallelDistribution_;
  }

  public int getBnInOp2ParallelDistributionCount() {
    return internalGetBnInOp2ParallelDistribution().getMap().size();
  }
  /**
   * <code>map&lt;string, .oneflow.ParallelDistribution&gt; bn_in_op2parallel_distribution = 1;</code>
   */

  public boolean containsBnInOp2ParallelDistribution(
      java.lang.String key) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    return internalGetBnInOp2ParallelDistribution().getMap().containsKey(key);
  }
  /**
   * Use {@link #getBnInOp2ParallelDistributionMap()} instead.
   */
  @java.lang.Deprecated
  public java.util.Map<java.lang.String, org.oneflow.core.job.ParallelDistribution> getBnInOp2ParallelDistribution() {
    return getBnInOp2ParallelDistributionMap();
  }
  /**
   * <code>map&lt;string, .oneflow.ParallelDistribution&gt; bn_in_op2parallel_distribution = 1;</code>
   */

  public java.util.Map<java.lang.String, org.oneflow.core.job.ParallelDistribution> getBnInOp2ParallelDistributionMap() {
    return internalGetBnInOp2ParallelDistribution().getMap();
  }
  /**
   * <code>map&lt;string, .oneflow.ParallelDistribution&gt; bn_in_op2parallel_distribution = 1;</code>
   */

  public org.oneflow.core.job.ParallelDistribution getBnInOp2ParallelDistributionOrDefault(
      java.lang.String key,
      org.oneflow.core.job.ParallelDistribution defaultValue) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    java.util.Map<java.lang.String, org.oneflow.core.job.ParallelDistribution> map =
        internalGetBnInOp2ParallelDistribution().getMap();
    return map.containsKey(key) ? map.get(key) : defaultValue;
  }
  /**
   * <code>map&lt;string, .oneflow.ParallelDistribution&gt; bn_in_op2parallel_distribution = 1;</code>
   */

  public org.oneflow.core.job.ParallelDistribution getBnInOp2ParallelDistributionOrThrow(
      java.lang.String key) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    java.util.Map<java.lang.String, org.oneflow.core.job.ParallelDistribution> map =
        internalGetBnInOp2ParallelDistribution().getMap();
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

    for (org.oneflow.core.job.ParallelDistribution item : getBnInOp2ParallelDistribution().values()) {
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
    for (java.util.Map.Entry<java.lang.String, org.oneflow.core.job.ParallelDistribution> entry
         : internalGetBnInOp2ParallelDistribution().getMap().entrySet()) {
      com.google.protobuf.MapEntry<java.lang.String, org.oneflow.core.job.ParallelDistribution>
      bnInOp2ParallelDistribution = BnInOp2ParallelDistributionDefaultEntryHolder.defaultEntry.newBuilderForType()
          .setKey(entry.getKey())
          .setValue(entry.getValue())
          .build();
      output.writeMessage(1, bnInOp2ParallelDistribution);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    for (java.util.Map.Entry<java.lang.String, org.oneflow.core.job.ParallelDistribution> entry
         : internalGetBnInOp2ParallelDistribution().getMap().entrySet()) {
      com.google.protobuf.MapEntry<java.lang.String, org.oneflow.core.job.ParallelDistribution>
      bnInOp2ParallelDistribution = BnInOp2ParallelDistributionDefaultEntryHolder.defaultEntry.newBuilderForType()
          .setKey(entry.getKey())
          .setValue(entry.getValue())
          .build();
      size += com.google.protobuf.CodedOutputStream
          .computeMessageSize(1, bnInOp2ParallelDistribution);
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
    if (!(obj instanceof org.oneflow.core.job.ParallelDistributionSignature)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.ParallelDistributionSignature other = (org.oneflow.core.job.ParallelDistributionSignature) obj;

    boolean result = true;
    result = result && internalGetBnInOp2ParallelDistribution().equals(
        other.internalGetBnInOp2ParallelDistribution());
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
    if (!internalGetBnInOp2ParallelDistribution().getMap().isEmpty()) {
      hash = (37 * hash) + BN_IN_OP2PARALLEL_DISTRIBUTION_FIELD_NUMBER;
      hash = (53 * hash) + internalGetBnInOp2ParallelDistribution().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.ParallelDistributionSignature parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.ParallelDistributionSignature parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.ParallelDistributionSignature parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.ParallelDistributionSignature parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.ParallelDistributionSignature parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.ParallelDistributionSignature parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.ParallelDistributionSignature parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.ParallelDistributionSignature parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.ParallelDistributionSignature parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.ParallelDistributionSignature parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.ParallelDistributionSignature prototype) {
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
   * Protobuf type {@code oneflow.ParallelDistributionSignature}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.ParallelDistributionSignature)
      org.oneflow.core.job.ParallelDistributionSignatureOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.SbpParallelOuterClass.internal_static_oneflow_ParallelDistributionSignature_descriptor;
    }

    @SuppressWarnings({"rawtypes"})
    protected com.google.protobuf.MapField internalGetMapField(
        int number) {
      switch (number) {
        case 1:
          return internalGetBnInOp2ParallelDistribution();
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
          return internalGetMutableBnInOp2ParallelDistribution();
        default:
          throw new RuntimeException(
              "Invalid map field number: " + number);
      }
    }
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.SbpParallelOuterClass.internal_static_oneflow_ParallelDistributionSignature_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.ParallelDistributionSignature.class, org.oneflow.core.job.ParallelDistributionSignature.Builder.class);
    }

    // Construct using org.oneflow.core.job.ParallelDistributionSignature.newBuilder()
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
      internalGetMutableBnInOp2ParallelDistribution().clear();
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.SbpParallelOuterClass.internal_static_oneflow_ParallelDistributionSignature_descriptor;
    }

    public org.oneflow.core.job.ParallelDistributionSignature getDefaultInstanceForType() {
      return org.oneflow.core.job.ParallelDistributionSignature.getDefaultInstance();
    }

    public org.oneflow.core.job.ParallelDistributionSignature build() {
      org.oneflow.core.job.ParallelDistributionSignature result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.ParallelDistributionSignature buildPartial() {
      org.oneflow.core.job.ParallelDistributionSignature result = new org.oneflow.core.job.ParallelDistributionSignature(this);
      int from_bitField0_ = bitField0_;
      result.bnInOp2ParallelDistribution_ = internalGetBnInOp2ParallelDistribution();
      result.bnInOp2ParallelDistribution_.makeImmutable();
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
      if (other instanceof org.oneflow.core.job.ParallelDistributionSignature) {
        return mergeFrom((org.oneflow.core.job.ParallelDistributionSignature)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.ParallelDistributionSignature other) {
      if (other == org.oneflow.core.job.ParallelDistributionSignature.getDefaultInstance()) return this;
      internalGetMutableBnInOp2ParallelDistribution().mergeFrom(
          other.internalGetBnInOp2ParallelDistribution());
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      for (org.oneflow.core.job.ParallelDistribution item : getBnInOp2ParallelDistribution().values()) {
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
      org.oneflow.core.job.ParallelDistributionSignature parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.ParallelDistributionSignature) e.getUnfinishedMessage();
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
        java.lang.String, org.oneflow.core.job.ParallelDistribution> bnInOp2ParallelDistribution_;
    private com.google.protobuf.MapField<java.lang.String, org.oneflow.core.job.ParallelDistribution>
    internalGetBnInOp2ParallelDistribution() {
      if (bnInOp2ParallelDistribution_ == null) {
        return com.google.protobuf.MapField.emptyMapField(
            BnInOp2ParallelDistributionDefaultEntryHolder.defaultEntry);
      }
      return bnInOp2ParallelDistribution_;
    }
    private com.google.protobuf.MapField<java.lang.String, org.oneflow.core.job.ParallelDistribution>
    internalGetMutableBnInOp2ParallelDistribution() {
      onChanged();;
      if (bnInOp2ParallelDistribution_ == null) {
        bnInOp2ParallelDistribution_ = com.google.protobuf.MapField.newMapField(
            BnInOp2ParallelDistributionDefaultEntryHolder.defaultEntry);
      }
      if (!bnInOp2ParallelDistribution_.isMutable()) {
        bnInOp2ParallelDistribution_ = bnInOp2ParallelDistribution_.copy();
      }
      return bnInOp2ParallelDistribution_;
    }

    public int getBnInOp2ParallelDistributionCount() {
      return internalGetBnInOp2ParallelDistribution().getMap().size();
    }
    /**
     * <code>map&lt;string, .oneflow.ParallelDistribution&gt; bn_in_op2parallel_distribution = 1;</code>
     */

    public boolean containsBnInOp2ParallelDistribution(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      return internalGetBnInOp2ParallelDistribution().getMap().containsKey(key);
    }
    /**
     * Use {@link #getBnInOp2ParallelDistributionMap()} instead.
     */
    @java.lang.Deprecated
    public java.util.Map<java.lang.String, org.oneflow.core.job.ParallelDistribution> getBnInOp2ParallelDistribution() {
      return getBnInOp2ParallelDistributionMap();
    }
    /**
     * <code>map&lt;string, .oneflow.ParallelDistribution&gt; bn_in_op2parallel_distribution = 1;</code>
     */

    public java.util.Map<java.lang.String, org.oneflow.core.job.ParallelDistribution> getBnInOp2ParallelDistributionMap() {
      return internalGetBnInOp2ParallelDistribution().getMap();
    }
    /**
     * <code>map&lt;string, .oneflow.ParallelDistribution&gt; bn_in_op2parallel_distribution = 1;</code>
     */

    public org.oneflow.core.job.ParallelDistribution getBnInOp2ParallelDistributionOrDefault(
        java.lang.String key,
        org.oneflow.core.job.ParallelDistribution defaultValue) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      java.util.Map<java.lang.String, org.oneflow.core.job.ParallelDistribution> map =
          internalGetBnInOp2ParallelDistribution().getMap();
      return map.containsKey(key) ? map.get(key) : defaultValue;
    }
    /**
     * <code>map&lt;string, .oneflow.ParallelDistribution&gt; bn_in_op2parallel_distribution = 1;</code>
     */

    public org.oneflow.core.job.ParallelDistribution getBnInOp2ParallelDistributionOrThrow(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      java.util.Map<java.lang.String, org.oneflow.core.job.ParallelDistribution> map =
          internalGetBnInOp2ParallelDistribution().getMap();
      if (!map.containsKey(key)) {
        throw new java.lang.IllegalArgumentException();
      }
      return map.get(key);
    }

    public Builder clearBnInOp2ParallelDistribution() {
      getMutableBnInOp2ParallelDistribution().clear();
      return this;
    }
    /**
     * <code>map&lt;string, .oneflow.ParallelDistribution&gt; bn_in_op2parallel_distribution = 1;</code>
     */

    public Builder removeBnInOp2ParallelDistribution(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      getMutableBnInOp2ParallelDistribution().remove(key);
      return this;
    }
    /**
     * Use alternate mutation accessors instead.
     */
    @java.lang.Deprecated
    public java.util.Map<java.lang.String, org.oneflow.core.job.ParallelDistribution>
    getMutableBnInOp2ParallelDistribution() {
      return internalGetMutableBnInOp2ParallelDistribution().getMutableMap();
    }
    /**
     * <code>map&lt;string, .oneflow.ParallelDistribution&gt; bn_in_op2parallel_distribution = 1;</code>
     */
    public Builder putBnInOp2ParallelDistribution(
        java.lang.String key,
        org.oneflow.core.job.ParallelDistribution value) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      if (value == null) { throw new java.lang.NullPointerException(); }
      getMutableBnInOp2ParallelDistribution().put(key, value);
      return this;
    }
    /**
     * <code>map&lt;string, .oneflow.ParallelDistribution&gt; bn_in_op2parallel_distribution = 1;</code>
     */

    public Builder putAllBnInOp2ParallelDistribution(
        java.util.Map<java.lang.String, org.oneflow.core.job.ParallelDistribution> values) {
      getMutableBnInOp2ParallelDistribution().putAll(values);
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


    // @@protoc_insertion_point(builder_scope:oneflow.ParallelDistributionSignature)
  }

  // @@protoc_insertion_point(class_scope:oneflow.ParallelDistributionSignature)
  private static final org.oneflow.core.job.ParallelDistributionSignature DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.ParallelDistributionSignature();
  }

  public static org.oneflow.core.job.ParallelDistributionSignature getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<ParallelDistributionSignature>
      PARSER = new com.google.protobuf.AbstractParser<ParallelDistributionSignature>() {
    public ParallelDistributionSignature parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new ParallelDistributionSignature(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<ParallelDistributionSignature> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<ParallelDistributionSignature> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.ParallelDistributionSignature getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
