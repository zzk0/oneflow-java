// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/lbi_diff_watcher_info.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.LbiDiffWatcherInfo}
 */
public  final class LbiDiffWatcherInfo extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.LbiDiffWatcherInfo)
    LbiDiffWatcherInfoOrBuilder {
  // Use LbiDiffWatcherInfo.newBuilder() to construct.
  private LbiDiffWatcherInfo(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private LbiDiffWatcherInfo() {
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private LbiDiffWatcherInfo(
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
              jobName2LbiAndWatcherUuids_ = com.google.protobuf.MapField.newMapField(
                  JobName2LbiAndWatcherUuidsDefaultEntryHolder.defaultEntry);
              mutable_bitField0_ |= 0x00000001;
            }
            com.google.protobuf.MapEntry<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList>
            jobName2LbiAndWatcherUuids = input.readMessage(
                JobName2LbiAndWatcherUuidsDefaultEntryHolder.defaultEntry.getParserForType(), extensionRegistry);
            jobName2LbiAndWatcherUuids_.getMutableMap().put(jobName2LbiAndWatcherUuids.getKey(), jobName2LbiAndWatcherUuids.getValue());
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
    return org.oneflow.core.job.LbiDiffWatcherInfoOuterClass.internal_static_oneflow_LbiDiffWatcherInfo_descriptor;
  }

  @SuppressWarnings({"rawtypes"})
  protected com.google.protobuf.MapField internalGetMapField(
      int number) {
    switch (number) {
      case 1:
        return internalGetJobName2LbiAndWatcherUuids();
      default:
        throw new RuntimeException(
            "Invalid map field number: " + number);
    }
  }
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.LbiDiffWatcherInfoOuterClass.internal_static_oneflow_LbiDiffWatcherInfo_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.LbiDiffWatcherInfo.class, org.oneflow.core.job.LbiDiffWatcherInfo.Builder.class);
  }

  public static final int JOB_NAME2LBI_AND_WATCHER_UUIDS_FIELD_NUMBER = 1;
  private static final class JobName2LbiAndWatcherUuidsDefaultEntryHolder {
    static final com.google.protobuf.MapEntry<
        java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList> defaultEntry =
            com.google.protobuf.MapEntry
            .<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList>newDefaultInstance(
                org.oneflow.core.job.LbiDiffWatcherInfoOuterClass.internal_static_oneflow_LbiDiffWatcherInfo_JobName2lbiAndWatcherUuidsEntry_descriptor, 
                com.google.protobuf.WireFormat.FieldType.STRING,
                "",
                com.google.protobuf.WireFormat.FieldType.MESSAGE,
                org.oneflow.core.job.LbiAndDiffWatcherUuidPairList.getDefaultInstance());
  }
  private com.google.protobuf.MapField<
      java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList> jobName2LbiAndWatcherUuids_;
  private com.google.protobuf.MapField<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList>
  internalGetJobName2LbiAndWatcherUuids() {
    if (jobName2LbiAndWatcherUuids_ == null) {
      return com.google.protobuf.MapField.emptyMapField(
          JobName2LbiAndWatcherUuidsDefaultEntryHolder.defaultEntry);
    }
    return jobName2LbiAndWatcherUuids_;
  }

  public int getJobName2LbiAndWatcherUuidsCount() {
    return internalGetJobName2LbiAndWatcherUuids().getMap().size();
  }
  /**
   * <code>map&lt;string, .oneflow.LbiAndDiffWatcherUuidPairList&gt; job_name2lbi_and_watcher_uuids = 1;</code>
   */

  public boolean containsJobName2LbiAndWatcherUuids(
      java.lang.String key) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    return internalGetJobName2LbiAndWatcherUuids().getMap().containsKey(key);
  }
  /**
   * Use {@link #getJobName2LbiAndWatcherUuidsMap()} instead.
   */
  @java.lang.Deprecated
  public java.util.Map<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList> getJobName2LbiAndWatcherUuids() {
    return getJobName2LbiAndWatcherUuidsMap();
  }
  /**
   * <code>map&lt;string, .oneflow.LbiAndDiffWatcherUuidPairList&gt; job_name2lbi_and_watcher_uuids = 1;</code>
   */

  public java.util.Map<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList> getJobName2LbiAndWatcherUuidsMap() {
    return internalGetJobName2LbiAndWatcherUuids().getMap();
  }
  /**
   * <code>map&lt;string, .oneflow.LbiAndDiffWatcherUuidPairList&gt; job_name2lbi_and_watcher_uuids = 1;</code>
   */

  public org.oneflow.core.job.LbiAndDiffWatcherUuidPairList getJobName2LbiAndWatcherUuidsOrDefault(
      java.lang.String key,
      org.oneflow.core.job.LbiAndDiffWatcherUuidPairList defaultValue) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    java.util.Map<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList> map =
        internalGetJobName2LbiAndWatcherUuids().getMap();
    return map.containsKey(key) ? map.get(key) : defaultValue;
  }
  /**
   * <code>map&lt;string, .oneflow.LbiAndDiffWatcherUuidPairList&gt; job_name2lbi_and_watcher_uuids = 1;</code>
   */

  public org.oneflow.core.job.LbiAndDiffWatcherUuidPairList getJobName2LbiAndWatcherUuidsOrThrow(
      java.lang.String key) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    java.util.Map<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList> map =
        internalGetJobName2LbiAndWatcherUuids().getMap();
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

    for (org.oneflow.core.job.LbiAndDiffWatcherUuidPairList item : getJobName2LbiAndWatcherUuids().values()) {
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
    for (java.util.Map.Entry<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList> entry
         : internalGetJobName2LbiAndWatcherUuids().getMap().entrySet()) {
      com.google.protobuf.MapEntry<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList>
      jobName2LbiAndWatcherUuids = JobName2LbiAndWatcherUuidsDefaultEntryHolder.defaultEntry.newBuilderForType()
          .setKey(entry.getKey())
          .setValue(entry.getValue())
          .build();
      output.writeMessage(1, jobName2LbiAndWatcherUuids);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    for (java.util.Map.Entry<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList> entry
         : internalGetJobName2LbiAndWatcherUuids().getMap().entrySet()) {
      com.google.protobuf.MapEntry<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList>
      jobName2LbiAndWatcherUuids = JobName2LbiAndWatcherUuidsDefaultEntryHolder.defaultEntry.newBuilderForType()
          .setKey(entry.getKey())
          .setValue(entry.getValue())
          .build();
      size += com.google.protobuf.CodedOutputStream
          .computeMessageSize(1, jobName2LbiAndWatcherUuids);
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
    if (!(obj instanceof org.oneflow.core.job.LbiDiffWatcherInfo)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.LbiDiffWatcherInfo other = (org.oneflow.core.job.LbiDiffWatcherInfo) obj;

    boolean result = true;
    result = result && internalGetJobName2LbiAndWatcherUuids().equals(
        other.internalGetJobName2LbiAndWatcherUuids());
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
    if (!internalGetJobName2LbiAndWatcherUuids().getMap().isEmpty()) {
      hash = (37 * hash) + JOB_NAME2LBI_AND_WATCHER_UUIDS_FIELD_NUMBER;
      hash = (53 * hash) + internalGetJobName2LbiAndWatcherUuids().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.LbiDiffWatcherInfo parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.LbiDiffWatcherInfo parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.LbiDiffWatcherInfo parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.LbiDiffWatcherInfo parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.LbiDiffWatcherInfo parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.LbiDiffWatcherInfo parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.LbiDiffWatcherInfo parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.LbiDiffWatcherInfo parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.LbiDiffWatcherInfo parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.LbiDiffWatcherInfo parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.LbiDiffWatcherInfo prototype) {
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
   * Protobuf type {@code oneflow.LbiDiffWatcherInfo}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.LbiDiffWatcherInfo)
      org.oneflow.core.job.LbiDiffWatcherInfoOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.LbiDiffWatcherInfoOuterClass.internal_static_oneflow_LbiDiffWatcherInfo_descriptor;
    }

    @SuppressWarnings({"rawtypes"})
    protected com.google.protobuf.MapField internalGetMapField(
        int number) {
      switch (number) {
        case 1:
          return internalGetJobName2LbiAndWatcherUuids();
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
          return internalGetMutableJobName2LbiAndWatcherUuids();
        default:
          throw new RuntimeException(
              "Invalid map field number: " + number);
      }
    }
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.LbiDiffWatcherInfoOuterClass.internal_static_oneflow_LbiDiffWatcherInfo_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.LbiDiffWatcherInfo.class, org.oneflow.core.job.LbiDiffWatcherInfo.Builder.class);
    }

    // Construct using org.oneflow.core.job.LbiDiffWatcherInfo.newBuilder()
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
      internalGetMutableJobName2LbiAndWatcherUuids().clear();
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.LbiDiffWatcherInfoOuterClass.internal_static_oneflow_LbiDiffWatcherInfo_descriptor;
    }

    public org.oneflow.core.job.LbiDiffWatcherInfo getDefaultInstanceForType() {
      return org.oneflow.core.job.LbiDiffWatcherInfo.getDefaultInstance();
    }

    public org.oneflow.core.job.LbiDiffWatcherInfo build() {
      org.oneflow.core.job.LbiDiffWatcherInfo result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.LbiDiffWatcherInfo buildPartial() {
      org.oneflow.core.job.LbiDiffWatcherInfo result = new org.oneflow.core.job.LbiDiffWatcherInfo(this);
      int from_bitField0_ = bitField0_;
      result.jobName2LbiAndWatcherUuids_ = internalGetJobName2LbiAndWatcherUuids();
      result.jobName2LbiAndWatcherUuids_.makeImmutable();
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
      if (other instanceof org.oneflow.core.job.LbiDiffWatcherInfo) {
        return mergeFrom((org.oneflow.core.job.LbiDiffWatcherInfo)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.LbiDiffWatcherInfo other) {
      if (other == org.oneflow.core.job.LbiDiffWatcherInfo.getDefaultInstance()) return this;
      internalGetMutableJobName2LbiAndWatcherUuids().mergeFrom(
          other.internalGetJobName2LbiAndWatcherUuids());
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      for (org.oneflow.core.job.LbiAndDiffWatcherUuidPairList item : getJobName2LbiAndWatcherUuids().values()) {
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
      org.oneflow.core.job.LbiDiffWatcherInfo parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.LbiDiffWatcherInfo) e.getUnfinishedMessage();
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
        java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList> jobName2LbiAndWatcherUuids_;
    private com.google.protobuf.MapField<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList>
    internalGetJobName2LbiAndWatcherUuids() {
      if (jobName2LbiAndWatcherUuids_ == null) {
        return com.google.protobuf.MapField.emptyMapField(
            JobName2LbiAndWatcherUuidsDefaultEntryHolder.defaultEntry);
      }
      return jobName2LbiAndWatcherUuids_;
    }
    private com.google.protobuf.MapField<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList>
    internalGetMutableJobName2LbiAndWatcherUuids() {
      onChanged();;
      if (jobName2LbiAndWatcherUuids_ == null) {
        jobName2LbiAndWatcherUuids_ = com.google.protobuf.MapField.newMapField(
            JobName2LbiAndWatcherUuidsDefaultEntryHolder.defaultEntry);
      }
      if (!jobName2LbiAndWatcherUuids_.isMutable()) {
        jobName2LbiAndWatcherUuids_ = jobName2LbiAndWatcherUuids_.copy();
      }
      return jobName2LbiAndWatcherUuids_;
    }

    public int getJobName2LbiAndWatcherUuidsCount() {
      return internalGetJobName2LbiAndWatcherUuids().getMap().size();
    }
    /**
     * <code>map&lt;string, .oneflow.LbiAndDiffWatcherUuidPairList&gt; job_name2lbi_and_watcher_uuids = 1;</code>
     */

    public boolean containsJobName2LbiAndWatcherUuids(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      return internalGetJobName2LbiAndWatcherUuids().getMap().containsKey(key);
    }
    /**
     * Use {@link #getJobName2LbiAndWatcherUuidsMap()} instead.
     */
    @java.lang.Deprecated
    public java.util.Map<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList> getJobName2LbiAndWatcherUuids() {
      return getJobName2LbiAndWatcherUuidsMap();
    }
    /**
     * <code>map&lt;string, .oneflow.LbiAndDiffWatcherUuidPairList&gt; job_name2lbi_and_watcher_uuids = 1;</code>
     */

    public java.util.Map<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList> getJobName2LbiAndWatcherUuidsMap() {
      return internalGetJobName2LbiAndWatcherUuids().getMap();
    }
    /**
     * <code>map&lt;string, .oneflow.LbiAndDiffWatcherUuidPairList&gt; job_name2lbi_and_watcher_uuids = 1;</code>
     */

    public org.oneflow.core.job.LbiAndDiffWatcherUuidPairList getJobName2LbiAndWatcherUuidsOrDefault(
        java.lang.String key,
        org.oneflow.core.job.LbiAndDiffWatcherUuidPairList defaultValue) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      java.util.Map<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList> map =
          internalGetJobName2LbiAndWatcherUuids().getMap();
      return map.containsKey(key) ? map.get(key) : defaultValue;
    }
    /**
     * <code>map&lt;string, .oneflow.LbiAndDiffWatcherUuidPairList&gt; job_name2lbi_and_watcher_uuids = 1;</code>
     */

    public org.oneflow.core.job.LbiAndDiffWatcherUuidPairList getJobName2LbiAndWatcherUuidsOrThrow(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      java.util.Map<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList> map =
          internalGetJobName2LbiAndWatcherUuids().getMap();
      if (!map.containsKey(key)) {
        throw new java.lang.IllegalArgumentException();
      }
      return map.get(key);
    }

    public Builder clearJobName2LbiAndWatcherUuids() {
      getMutableJobName2LbiAndWatcherUuids().clear();
      return this;
    }
    /**
     * <code>map&lt;string, .oneflow.LbiAndDiffWatcherUuidPairList&gt; job_name2lbi_and_watcher_uuids = 1;</code>
     */

    public Builder removeJobName2LbiAndWatcherUuids(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      getMutableJobName2LbiAndWatcherUuids().remove(key);
      return this;
    }
    /**
     * Use alternate mutation accessors instead.
     */
    @java.lang.Deprecated
    public java.util.Map<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList>
    getMutableJobName2LbiAndWatcherUuids() {
      return internalGetMutableJobName2LbiAndWatcherUuids().getMutableMap();
    }
    /**
     * <code>map&lt;string, .oneflow.LbiAndDiffWatcherUuidPairList&gt; job_name2lbi_and_watcher_uuids = 1;</code>
     */
    public Builder putJobName2LbiAndWatcherUuids(
        java.lang.String key,
        org.oneflow.core.job.LbiAndDiffWatcherUuidPairList value) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      if (value == null) { throw new java.lang.NullPointerException(); }
      getMutableJobName2LbiAndWatcherUuids().put(key, value);
      return this;
    }
    /**
     * <code>map&lt;string, .oneflow.LbiAndDiffWatcherUuidPairList&gt; job_name2lbi_and_watcher_uuids = 1;</code>
     */

    public Builder putAllJobName2LbiAndWatcherUuids(
        java.util.Map<java.lang.String, org.oneflow.core.job.LbiAndDiffWatcherUuidPairList> values) {
      getMutableJobName2LbiAndWatcherUuids().putAll(values);
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


    // @@protoc_insertion_point(builder_scope:oneflow.LbiDiffWatcherInfo)
  }

  // @@protoc_insertion_point(class_scope:oneflow.LbiDiffWatcherInfo)
  private static final org.oneflow.core.job.LbiDiffWatcherInfo DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.LbiDiffWatcherInfo();
  }

  public static org.oneflow.core.job.LbiDiffWatcherInfo getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<LbiDiffWatcherInfo>
      PARSER = new com.google.protobuf.AbstractParser<LbiDiffWatcherInfo>() {
    public LbiDiffWatcherInfo parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new LbiDiffWatcherInfo(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<LbiDiffWatcherInfo> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<LbiDiffWatcherInfo> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.LbiDiffWatcherInfo getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}
