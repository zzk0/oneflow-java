// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/mirrored_parallel.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.MirroredSignature}
 */
public  final class MirroredSignature extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.MirroredSignature)
    MirroredSignatureOrBuilder {
  // Use MirroredSignature.newBuilder() to construct.
  private MirroredSignature(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private MirroredSignature() {
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private MirroredSignature(
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
              bnInOp2OptMirroredParallel_ = com.google.protobuf.MapField.newMapField(
                  BnInOp2OptMirroredParallelDefaultEntryHolder.defaultEntry);
              mutable_bitField0_ |= 0x00000001;
            }
            com.google.protobuf.MapEntry<java.lang.String, org.oneflow.core.job.OptMirroredParallel>
            bnInOp2OptMirroredParallel = input.readMessage(
                BnInOp2OptMirroredParallelDefaultEntryHolder.defaultEntry.getParserForType(), extensionRegistry);
            bnInOp2OptMirroredParallel_.getMutableMap().put(bnInOp2OptMirroredParallel.getKey(), bnInOp2OptMirroredParallel.getValue());
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
    return org.oneflow.core.job.MirroredParallelOuterClass.internal_static_oneflow_MirroredSignature_descriptor;
  }

  @SuppressWarnings({"rawtypes"})
  protected com.google.protobuf.MapField internalGetMapField(
      int number) {
    switch (number) {
      case 1:
        return internalGetBnInOp2OptMirroredParallel();
      default:
        throw new RuntimeException(
            "Invalid map field number: " + number);
    }
  }
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.MirroredParallelOuterClass.internal_static_oneflow_MirroredSignature_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.MirroredSignature.class, org.oneflow.core.job.MirroredSignature.Builder.class);
  }

  public static final int BN_IN_OP2OPT_MIRRORED_PARALLEL_FIELD_NUMBER = 1;
  private static final class BnInOp2OptMirroredParallelDefaultEntryHolder {
    static final com.google.protobuf.MapEntry<
        java.lang.String, org.oneflow.core.job.OptMirroredParallel> defaultEntry =
            com.google.protobuf.MapEntry
            .<java.lang.String, org.oneflow.core.job.OptMirroredParallel>newDefaultInstance(
                org.oneflow.core.job.MirroredParallelOuterClass.internal_static_oneflow_MirroredSignature_BnInOp2optMirroredParallelEntry_descriptor, 
                com.google.protobuf.WireFormat.FieldType.STRING,
                "",
                com.google.protobuf.WireFormat.FieldType.MESSAGE,
                org.oneflow.core.job.OptMirroredParallel.getDefaultInstance());
  }
  private com.google.protobuf.MapField<
      java.lang.String, org.oneflow.core.job.OptMirroredParallel> bnInOp2OptMirroredParallel_;
  private com.google.protobuf.MapField<java.lang.String, org.oneflow.core.job.OptMirroredParallel>
  internalGetBnInOp2OptMirroredParallel() {
    if (bnInOp2OptMirroredParallel_ == null) {
      return com.google.protobuf.MapField.emptyMapField(
          BnInOp2OptMirroredParallelDefaultEntryHolder.defaultEntry);
    }
    return bnInOp2OptMirroredParallel_;
  }

  public int getBnInOp2OptMirroredParallelCount() {
    return internalGetBnInOp2OptMirroredParallel().getMap().size();
  }
  /**
   * <code>map&lt;string, .oneflow.OptMirroredParallel&gt; bn_in_op2opt_mirrored_parallel = 1;</code>
   */

  public boolean containsBnInOp2OptMirroredParallel(
      java.lang.String key) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    return internalGetBnInOp2OptMirroredParallel().getMap().containsKey(key);
  }
  /**
   * Use {@link #getBnInOp2OptMirroredParallelMap()} instead.
   */
  @java.lang.Deprecated
  public java.util.Map<java.lang.String, org.oneflow.core.job.OptMirroredParallel> getBnInOp2OptMirroredParallel() {
    return getBnInOp2OptMirroredParallelMap();
  }
  /**
   * <code>map&lt;string, .oneflow.OptMirroredParallel&gt; bn_in_op2opt_mirrored_parallel = 1;</code>
   */

  public java.util.Map<java.lang.String, org.oneflow.core.job.OptMirroredParallel> getBnInOp2OptMirroredParallelMap() {
    return internalGetBnInOp2OptMirroredParallel().getMap();
  }
  /**
   * <code>map&lt;string, .oneflow.OptMirroredParallel&gt; bn_in_op2opt_mirrored_parallel = 1;</code>
   */

  public org.oneflow.core.job.OptMirroredParallel getBnInOp2OptMirroredParallelOrDefault(
      java.lang.String key,
      org.oneflow.core.job.OptMirroredParallel defaultValue) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    java.util.Map<java.lang.String, org.oneflow.core.job.OptMirroredParallel> map =
        internalGetBnInOp2OptMirroredParallel().getMap();
    return map.containsKey(key) ? map.get(key) : defaultValue;
  }
  /**
   * <code>map&lt;string, .oneflow.OptMirroredParallel&gt; bn_in_op2opt_mirrored_parallel = 1;</code>
   */

  public org.oneflow.core.job.OptMirroredParallel getBnInOp2OptMirroredParallelOrThrow(
      java.lang.String key) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    java.util.Map<java.lang.String, org.oneflow.core.job.OptMirroredParallel> map =
        internalGetBnInOp2OptMirroredParallel().getMap();
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
    for (java.util.Map.Entry<java.lang.String, org.oneflow.core.job.OptMirroredParallel> entry
         : internalGetBnInOp2OptMirroredParallel().getMap().entrySet()) {
      com.google.protobuf.MapEntry<java.lang.String, org.oneflow.core.job.OptMirroredParallel>
      bnInOp2OptMirroredParallel = BnInOp2OptMirroredParallelDefaultEntryHolder.defaultEntry.newBuilderForType()
          .setKey(entry.getKey())
          .setValue(entry.getValue())
          .build();
      output.writeMessage(1, bnInOp2OptMirroredParallel);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    for (java.util.Map.Entry<java.lang.String, org.oneflow.core.job.OptMirroredParallel> entry
         : internalGetBnInOp2OptMirroredParallel().getMap().entrySet()) {
      com.google.protobuf.MapEntry<java.lang.String, org.oneflow.core.job.OptMirroredParallel>
      bnInOp2OptMirroredParallel = BnInOp2OptMirroredParallelDefaultEntryHolder.defaultEntry.newBuilderForType()
          .setKey(entry.getKey())
          .setValue(entry.getValue())
          .build();
      size += com.google.protobuf.CodedOutputStream
          .computeMessageSize(1, bnInOp2OptMirroredParallel);
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
    if (!(obj instanceof org.oneflow.core.job.MirroredSignature)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.MirroredSignature other = (org.oneflow.core.job.MirroredSignature) obj;

    boolean result = true;
    result = result && internalGetBnInOp2OptMirroredParallel().equals(
        other.internalGetBnInOp2OptMirroredParallel());
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
    if (!internalGetBnInOp2OptMirroredParallel().getMap().isEmpty()) {
      hash = (37 * hash) + BN_IN_OP2OPT_MIRRORED_PARALLEL_FIELD_NUMBER;
      hash = (53 * hash) + internalGetBnInOp2OptMirroredParallel().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.MirroredSignature parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.MirroredSignature parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.MirroredSignature parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.MirroredSignature parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.MirroredSignature parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.MirroredSignature parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.MirroredSignature parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.MirroredSignature parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.MirroredSignature parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.MirroredSignature parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.MirroredSignature prototype) {
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
   * Protobuf type {@code oneflow.MirroredSignature}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.MirroredSignature)
      org.oneflow.core.job.MirroredSignatureOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.MirroredParallelOuterClass.internal_static_oneflow_MirroredSignature_descriptor;
    }

    @SuppressWarnings({"rawtypes"})
    protected com.google.protobuf.MapField internalGetMapField(
        int number) {
      switch (number) {
        case 1:
          return internalGetBnInOp2OptMirroredParallel();
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
          return internalGetMutableBnInOp2OptMirroredParallel();
        default:
          throw new RuntimeException(
              "Invalid map field number: " + number);
      }
    }
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.MirroredParallelOuterClass.internal_static_oneflow_MirroredSignature_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.MirroredSignature.class, org.oneflow.core.job.MirroredSignature.Builder.class);
    }

    // Construct using org.oneflow.core.job.MirroredSignature.newBuilder()
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
      internalGetMutableBnInOp2OptMirroredParallel().clear();
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.MirroredParallelOuterClass.internal_static_oneflow_MirroredSignature_descriptor;
    }

    public org.oneflow.core.job.MirroredSignature getDefaultInstanceForType() {
      return org.oneflow.core.job.MirroredSignature.getDefaultInstance();
    }

    public org.oneflow.core.job.MirroredSignature build() {
      org.oneflow.core.job.MirroredSignature result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.MirroredSignature buildPartial() {
      org.oneflow.core.job.MirroredSignature result = new org.oneflow.core.job.MirroredSignature(this);
      int from_bitField0_ = bitField0_;
      result.bnInOp2OptMirroredParallel_ = internalGetBnInOp2OptMirroredParallel();
      result.bnInOp2OptMirroredParallel_.makeImmutable();
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
      if (other instanceof org.oneflow.core.job.MirroredSignature) {
        return mergeFrom((org.oneflow.core.job.MirroredSignature)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.MirroredSignature other) {
      if (other == org.oneflow.core.job.MirroredSignature.getDefaultInstance()) return this;
      internalGetMutableBnInOp2OptMirroredParallel().mergeFrom(
          other.internalGetBnInOp2OptMirroredParallel());
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
      org.oneflow.core.job.MirroredSignature parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.MirroredSignature) e.getUnfinishedMessage();
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
        java.lang.String, org.oneflow.core.job.OptMirroredParallel> bnInOp2OptMirroredParallel_;
    private com.google.protobuf.MapField<java.lang.String, org.oneflow.core.job.OptMirroredParallel>
    internalGetBnInOp2OptMirroredParallel() {
      if (bnInOp2OptMirroredParallel_ == null) {
        return com.google.protobuf.MapField.emptyMapField(
            BnInOp2OptMirroredParallelDefaultEntryHolder.defaultEntry);
      }
      return bnInOp2OptMirroredParallel_;
    }
    private com.google.protobuf.MapField<java.lang.String, org.oneflow.core.job.OptMirroredParallel>
    internalGetMutableBnInOp2OptMirroredParallel() {
      onChanged();;
      if (bnInOp2OptMirroredParallel_ == null) {
        bnInOp2OptMirroredParallel_ = com.google.protobuf.MapField.newMapField(
            BnInOp2OptMirroredParallelDefaultEntryHolder.defaultEntry);
      }
      if (!bnInOp2OptMirroredParallel_.isMutable()) {
        bnInOp2OptMirroredParallel_ = bnInOp2OptMirroredParallel_.copy();
      }
      return bnInOp2OptMirroredParallel_;
    }

    public int getBnInOp2OptMirroredParallelCount() {
      return internalGetBnInOp2OptMirroredParallel().getMap().size();
    }
    /**
     * <code>map&lt;string, .oneflow.OptMirroredParallel&gt; bn_in_op2opt_mirrored_parallel = 1;</code>
     */

    public boolean containsBnInOp2OptMirroredParallel(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      return internalGetBnInOp2OptMirroredParallel().getMap().containsKey(key);
    }
    /**
     * Use {@link #getBnInOp2OptMirroredParallelMap()} instead.
     */
    @java.lang.Deprecated
    public java.util.Map<java.lang.String, org.oneflow.core.job.OptMirroredParallel> getBnInOp2OptMirroredParallel() {
      return getBnInOp2OptMirroredParallelMap();
    }
    /**
     * <code>map&lt;string, .oneflow.OptMirroredParallel&gt; bn_in_op2opt_mirrored_parallel = 1;</code>
     */

    public java.util.Map<java.lang.String, org.oneflow.core.job.OptMirroredParallel> getBnInOp2OptMirroredParallelMap() {
      return internalGetBnInOp2OptMirroredParallel().getMap();
    }
    /**
     * <code>map&lt;string, .oneflow.OptMirroredParallel&gt; bn_in_op2opt_mirrored_parallel = 1;</code>
     */

    public org.oneflow.core.job.OptMirroredParallel getBnInOp2OptMirroredParallelOrDefault(
        java.lang.String key,
        org.oneflow.core.job.OptMirroredParallel defaultValue) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      java.util.Map<java.lang.String, org.oneflow.core.job.OptMirroredParallel> map =
          internalGetBnInOp2OptMirroredParallel().getMap();
      return map.containsKey(key) ? map.get(key) : defaultValue;
    }
    /**
     * <code>map&lt;string, .oneflow.OptMirroredParallel&gt; bn_in_op2opt_mirrored_parallel = 1;</code>
     */

    public org.oneflow.core.job.OptMirroredParallel getBnInOp2OptMirroredParallelOrThrow(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      java.util.Map<java.lang.String, org.oneflow.core.job.OptMirroredParallel> map =
          internalGetBnInOp2OptMirroredParallel().getMap();
      if (!map.containsKey(key)) {
        throw new java.lang.IllegalArgumentException();
      }
      return map.get(key);
    }

    public Builder clearBnInOp2OptMirroredParallel() {
      getMutableBnInOp2OptMirroredParallel().clear();
      return this;
    }
    /**
     * <code>map&lt;string, .oneflow.OptMirroredParallel&gt; bn_in_op2opt_mirrored_parallel = 1;</code>
     */

    public Builder removeBnInOp2OptMirroredParallel(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      getMutableBnInOp2OptMirroredParallel().remove(key);
      return this;
    }
    /**
     * Use alternate mutation accessors instead.
     */
    @java.lang.Deprecated
    public java.util.Map<java.lang.String, org.oneflow.core.job.OptMirroredParallel>
    getMutableBnInOp2OptMirroredParallel() {
      return internalGetMutableBnInOp2OptMirroredParallel().getMutableMap();
    }
    /**
     * <code>map&lt;string, .oneflow.OptMirroredParallel&gt; bn_in_op2opt_mirrored_parallel = 1;</code>
     */
    public Builder putBnInOp2OptMirroredParallel(
        java.lang.String key,
        org.oneflow.core.job.OptMirroredParallel value) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      if (value == null) { throw new java.lang.NullPointerException(); }
      getMutableBnInOp2OptMirroredParallel().put(key, value);
      return this;
    }
    /**
     * <code>map&lt;string, .oneflow.OptMirroredParallel&gt; bn_in_op2opt_mirrored_parallel = 1;</code>
     */

    public Builder putAllBnInOp2OptMirroredParallel(
        java.util.Map<java.lang.String, org.oneflow.core.job.OptMirroredParallel> values) {
      getMutableBnInOp2OptMirroredParallel().putAll(values);
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


    // @@protoc_insertion_point(builder_scope:oneflow.MirroredSignature)
  }

  // @@protoc_insertion_point(class_scope:oneflow.MirroredSignature)
  private static final org.oneflow.core.job.MirroredSignature DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.MirroredSignature();
  }

  public static org.oneflow.core.job.MirroredSignature getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<MirroredSignature>
      PARSER = new com.google.protobuf.AbstractParser<MirroredSignature>() {
    public MirroredSignature parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new MirroredSignature(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<MirroredSignature> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<MirroredSignature> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.MirroredSignature getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
