// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/plan.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.CollectiveBoxingPlan}
 */
public  final class CollectiveBoxingPlan extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.CollectiveBoxingPlan)
    CollectiveBoxingPlanOrBuilder {
  // Use CollectiveBoxingPlan.newBuilder() to construct.
  private CollectiveBoxingPlan(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private CollectiveBoxingPlan() {
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private CollectiveBoxingPlan(
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
              jobId2RequestSet_ = com.google.protobuf.MapField.newMapField(
                  JobId2RequestSetDefaultEntryHolder.defaultEntry);
              mutable_bitField0_ |= 0x00000001;
            }
            com.google.protobuf.MapEntry<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet>
            jobId2RequestSet = input.readMessage(
                JobId2RequestSetDefaultEntryHolder.defaultEntry.getParserForType(), extensionRegistry);
            jobId2RequestSet_.getMutableMap().put(jobId2RequestSet.getKey(), jobId2RequestSet.getValue());
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
    return org.oneflow.core.job.PlanOuterClass.internal_static_oneflow_CollectiveBoxingPlan_descriptor;
  }

  @SuppressWarnings({"rawtypes"})
  protected com.google.protobuf.MapField internalGetMapField(
      int number) {
    switch (number) {
      case 1:
        return internalGetJobId2RequestSet();
      default:
        throw new RuntimeException(
            "Invalid map field number: " + number);
    }
  }
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.PlanOuterClass.internal_static_oneflow_CollectiveBoxingPlan_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.CollectiveBoxingPlan.class, org.oneflow.core.job.CollectiveBoxingPlan.Builder.class);
  }

  public static final int JOB_ID2REQUEST_SET_FIELD_NUMBER = 1;
  private static final class JobId2RequestSetDefaultEntryHolder {
    static final com.google.protobuf.MapEntry<
        java.lang.Long, org.oneflow.core.graph.boxing.RequestSet> defaultEntry =
            com.google.protobuf.MapEntry
            .<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet>newDefaultInstance(
                org.oneflow.core.job.PlanOuterClass.internal_static_oneflow_CollectiveBoxingPlan_JobId2requestSetEntry_descriptor, 
                com.google.protobuf.WireFormat.FieldType.INT64,
                0L,
                com.google.protobuf.WireFormat.FieldType.MESSAGE,
                org.oneflow.core.graph.boxing.RequestSet.getDefaultInstance());
  }
  private com.google.protobuf.MapField<
      java.lang.Long, org.oneflow.core.graph.boxing.RequestSet> jobId2RequestSet_;
  private com.google.protobuf.MapField<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet>
  internalGetJobId2RequestSet() {
    if (jobId2RequestSet_ == null) {
      return com.google.protobuf.MapField.emptyMapField(
          JobId2RequestSetDefaultEntryHolder.defaultEntry);
    }
    return jobId2RequestSet_;
  }

  public int getJobId2RequestSetCount() {
    return internalGetJobId2RequestSet().getMap().size();
  }
  /**
   * <code>map&lt;int64, .oneflow.boxing.collective.RequestSet&gt; job_id2request_set = 1;</code>
   */

  public boolean containsJobId2RequestSet(
      long key) {
    
    return internalGetJobId2RequestSet().getMap().containsKey(key);
  }
  /**
   * Use {@link #getJobId2RequestSetMap()} instead.
   */
  @java.lang.Deprecated
  public java.util.Map<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet> getJobId2RequestSet() {
    return getJobId2RequestSetMap();
  }
  /**
   * <code>map&lt;int64, .oneflow.boxing.collective.RequestSet&gt; job_id2request_set = 1;</code>
   */

  public java.util.Map<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet> getJobId2RequestSetMap() {
    return internalGetJobId2RequestSet().getMap();
  }
  /**
   * <code>map&lt;int64, .oneflow.boxing.collective.RequestSet&gt; job_id2request_set = 1;</code>
   */

  public org.oneflow.core.graph.boxing.RequestSet getJobId2RequestSetOrDefault(
      long key,
      org.oneflow.core.graph.boxing.RequestSet defaultValue) {
    
    java.util.Map<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet> map =
        internalGetJobId2RequestSet().getMap();
    return map.containsKey(key) ? map.get(key) : defaultValue;
  }
  /**
   * <code>map&lt;int64, .oneflow.boxing.collective.RequestSet&gt; job_id2request_set = 1;</code>
   */

  public org.oneflow.core.graph.boxing.RequestSet getJobId2RequestSetOrThrow(
      long key) {
    
    java.util.Map<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet> map =
        internalGetJobId2RequestSet().getMap();
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

    for (org.oneflow.core.graph.boxing.RequestSet item : getJobId2RequestSet().values()) {
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
    for (java.util.Map.Entry<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet> entry
         : internalGetJobId2RequestSet().getMap().entrySet()) {
      com.google.protobuf.MapEntry<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet>
      jobId2RequestSet = JobId2RequestSetDefaultEntryHolder.defaultEntry.newBuilderForType()
          .setKey(entry.getKey())
          .setValue(entry.getValue())
          .build();
      output.writeMessage(1, jobId2RequestSet);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    for (java.util.Map.Entry<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet> entry
         : internalGetJobId2RequestSet().getMap().entrySet()) {
      com.google.protobuf.MapEntry<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet>
      jobId2RequestSet = JobId2RequestSetDefaultEntryHolder.defaultEntry.newBuilderForType()
          .setKey(entry.getKey())
          .setValue(entry.getValue())
          .build();
      size += com.google.protobuf.CodedOutputStream
          .computeMessageSize(1, jobId2RequestSet);
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
    if (!(obj instanceof org.oneflow.core.job.CollectiveBoxingPlan)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.CollectiveBoxingPlan other = (org.oneflow.core.job.CollectiveBoxingPlan) obj;

    boolean result = true;
    result = result && internalGetJobId2RequestSet().equals(
        other.internalGetJobId2RequestSet());
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
    if (!internalGetJobId2RequestSet().getMap().isEmpty()) {
      hash = (37 * hash) + JOB_ID2REQUEST_SET_FIELD_NUMBER;
      hash = (53 * hash) + internalGetJobId2RequestSet().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.CollectiveBoxingPlan parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.CollectiveBoxingPlan parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.CollectiveBoxingPlan parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.CollectiveBoxingPlan parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.CollectiveBoxingPlan parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.CollectiveBoxingPlan parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.CollectiveBoxingPlan parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.CollectiveBoxingPlan parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.CollectiveBoxingPlan parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.CollectiveBoxingPlan parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.CollectiveBoxingPlan prototype) {
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
   * Protobuf type {@code oneflow.CollectiveBoxingPlan}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.CollectiveBoxingPlan)
      org.oneflow.core.job.CollectiveBoxingPlanOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.PlanOuterClass.internal_static_oneflow_CollectiveBoxingPlan_descriptor;
    }

    @SuppressWarnings({"rawtypes"})
    protected com.google.protobuf.MapField internalGetMapField(
        int number) {
      switch (number) {
        case 1:
          return internalGetJobId2RequestSet();
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
          return internalGetMutableJobId2RequestSet();
        default:
          throw new RuntimeException(
              "Invalid map field number: " + number);
      }
    }
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.PlanOuterClass.internal_static_oneflow_CollectiveBoxingPlan_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.CollectiveBoxingPlan.class, org.oneflow.core.job.CollectiveBoxingPlan.Builder.class);
    }

    // Construct using org.oneflow.core.job.CollectiveBoxingPlan.newBuilder()
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
      internalGetMutableJobId2RequestSet().clear();
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.PlanOuterClass.internal_static_oneflow_CollectiveBoxingPlan_descriptor;
    }

    public org.oneflow.core.job.CollectiveBoxingPlan getDefaultInstanceForType() {
      return org.oneflow.core.job.CollectiveBoxingPlan.getDefaultInstance();
    }

    public org.oneflow.core.job.CollectiveBoxingPlan build() {
      org.oneflow.core.job.CollectiveBoxingPlan result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.CollectiveBoxingPlan buildPartial() {
      org.oneflow.core.job.CollectiveBoxingPlan result = new org.oneflow.core.job.CollectiveBoxingPlan(this);
      int from_bitField0_ = bitField0_;
      result.jobId2RequestSet_ = internalGetJobId2RequestSet();
      result.jobId2RequestSet_.makeImmutable();
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
      if (other instanceof org.oneflow.core.job.CollectiveBoxingPlan) {
        return mergeFrom((org.oneflow.core.job.CollectiveBoxingPlan)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.CollectiveBoxingPlan other) {
      if (other == org.oneflow.core.job.CollectiveBoxingPlan.getDefaultInstance()) return this;
      internalGetMutableJobId2RequestSet().mergeFrom(
          other.internalGetJobId2RequestSet());
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      for (org.oneflow.core.graph.boxing.RequestSet item : getJobId2RequestSet().values()) {
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
      org.oneflow.core.job.CollectiveBoxingPlan parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.CollectiveBoxingPlan) e.getUnfinishedMessage();
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
        java.lang.Long, org.oneflow.core.graph.boxing.RequestSet> jobId2RequestSet_;
    private com.google.protobuf.MapField<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet>
    internalGetJobId2RequestSet() {
      if (jobId2RequestSet_ == null) {
        return com.google.protobuf.MapField.emptyMapField(
            JobId2RequestSetDefaultEntryHolder.defaultEntry);
      }
      return jobId2RequestSet_;
    }
    private com.google.protobuf.MapField<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet>
    internalGetMutableJobId2RequestSet() {
      onChanged();;
      if (jobId2RequestSet_ == null) {
        jobId2RequestSet_ = com.google.protobuf.MapField.newMapField(
            JobId2RequestSetDefaultEntryHolder.defaultEntry);
      }
      if (!jobId2RequestSet_.isMutable()) {
        jobId2RequestSet_ = jobId2RequestSet_.copy();
      }
      return jobId2RequestSet_;
    }

    public int getJobId2RequestSetCount() {
      return internalGetJobId2RequestSet().getMap().size();
    }
    /**
     * <code>map&lt;int64, .oneflow.boxing.collective.RequestSet&gt; job_id2request_set = 1;</code>
     */

    public boolean containsJobId2RequestSet(
        long key) {
      
      return internalGetJobId2RequestSet().getMap().containsKey(key);
    }
    /**
     * Use {@link #getJobId2RequestSetMap()} instead.
     */
    @java.lang.Deprecated
    public java.util.Map<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet> getJobId2RequestSet() {
      return getJobId2RequestSetMap();
    }
    /**
     * <code>map&lt;int64, .oneflow.boxing.collective.RequestSet&gt; job_id2request_set = 1;</code>
     */

    public java.util.Map<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet> getJobId2RequestSetMap() {
      return internalGetJobId2RequestSet().getMap();
    }
    /**
     * <code>map&lt;int64, .oneflow.boxing.collective.RequestSet&gt; job_id2request_set = 1;</code>
     */

    public org.oneflow.core.graph.boxing.RequestSet getJobId2RequestSetOrDefault(
        long key,
        org.oneflow.core.graph.boxing.RequestSet defaultValue) {
      
      java.util.Map<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet> map =
          internalGetJobId2RequestSet().getMap();
      return map.containsKey(key) ? map.get(key) : defaultValue;
    }
    /**
     * <code>map&lt;int64, .oneflow.boxing.collective.RequestSet&gt; job_id2request_set = 1;</code>
     */

    public org.oneflow.core.graph.boxing.RequestSet getJobId2RequestSetOrThrow(
        long key) {
      
      java.util.Map<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet> map =
          internalGetJobId2RequestSet().getMap();
      if (!map.containsKey(key)) {
        throw new java.lang.IllegalArgumentException();
      }
      return map.get(key);
    }

    public Builder clearJobId2RequestSet() {
      getMutableJobId2RequestSet().clear();
      return this;
    }
    /**
     * <code>map&lt;int64, .oneflow.boxing.collective.RequestSet&gt; job_id2request_set = 1;</code>
     */

    public Builder removeJobId2RequestSet(
        long key) {
      
      getMutableJobId2RequestSet().remove(key);
      return this;
    }
    /**
     * Use alternate mutation accessors instead.
     */
    @java.lang.Deprecated
    public java.util.Map<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet>
    getMutableJobId2RequestSet() {
      return internalGetMutableJobId2RequestSet().getMutableMap();
    }
    /**
     * <code>map&lt;int64, .oneflow.boxing.collective.RequestSet&gt; job_id2request_set = 1;</code>
     */
    public Builder putJobId2RequestSet(
        long key,
        org.oneflow.core.graph.boxing.RequestSet value) {
      
      if (value == null) { throw new java.lang.NullPointerException(); }
      getMutableJobId2RequestSet().put(key, value);
      return this;
    }
    /**
     * <code>map&lt;int64, .oneflow.boxing.collective.RequestSet&gt; job_id2request_set = 1;</code>
     */

    public Builder putAllJobId2RequestSet(
        java.util.Map<java.lang.Long, org.oneflow.core.graph.boxing.RequestSet> values) {
      getMutableJobId2RequestSet().putAll(values);
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


    // @@protoc_insertion_point(builder_scope:oneflow.CollectiveBoxingPlan)
  }

  // @@protoc_insertion_point(class_scope:oneflow.CollectiveBoxingPlan)
  private static final org.oneflow.core.job.CollectiveBoxingPlan DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.CollectiveBoxingPlan();
  }

  public static org.oneflow.core.job.CollectiveBoxingPlan getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<CollectiveBoxingPlan>
      PARSER = new com.google.protobuf.AbstractParser<CollectiveBoxingPlan>() {
    public CollectiveBoxingPlan parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new CollectiveBoxingPlan(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<CollectiveBoxingPlan> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<CollectiveBoxingPlan> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.CollectiveBoxingPlan getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

