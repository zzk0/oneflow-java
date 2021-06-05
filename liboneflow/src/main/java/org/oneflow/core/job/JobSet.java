// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/job_set.proto

package org.oneflow.core.job;

/**
 * Protobuf type {@code oneflow.JobSet}
 */
public  final class JobSet extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.JobSet)
    JobSetOrBuilder {
  // Use JobSet.newBuilder() to construct.
  private JobSet(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private JobSet() {
    job_ = java.util.Collections.emptyList();
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private JobSet(
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
              job_ = new java.util.ArrayList<org.oneflow.core.job.Job>();
              mutable_bitField0_ |= 0x00000001;
            }
            job_.add(
                input.readMessage(org.oneflow.core.job.Job.PARSER, extensionRegistry));
            break;
          }
          case 42: {
            org.oneflow.core.job.InterJobReuseMemStrategy.Builder subBuilder = null;
            if (((bitField0_ & 0x00000001) == 0x00000001)) {
              subBuilder = interJobReuseMemStrategy_.toBuilder();
            }
            interJobReuseMemStrategy_ = input.readMessage(org.oneflow.core.job.InterJobReuseMemStrategy.PARSER, extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(interJobReuseMemStrategy_);
              interJobReuseMemStrategy_ = subBuilder.buildPartial();
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
      if (((mutable_bitField0_ & 0x00000001) == 0x00000001)) {
        job_ = java.util.Collections.unmodifiableList(job_);
      }
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.oneflow.core.job.JobSetOuterClass.internal_static_oneflow_JobSet_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.job.JobSetOuterClass.internal_static_oneflow_JobSet_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.job.JobSet.class, org.oneflow.core.job.JobSet.Builder.class);
  }

  private int bitField0_;
  public static final int JOB_FIELD_NUMBER = 1;
  private java.util.List<org.oneflow.core.job.Job> job_;
  /**
   * <code>repeated .oneflow.Job job = 1;</code>
   */
  public java.util.List<org.oneflow.core.job.Job> getJobList() {
    return job_;
  }
  /**
   * <code>repeated .oneflow.Job job = 1;</code>
   */
  public java.util.List<? extends org.oneflow.core.job.JobOrBuilder> 
      getJobOrBuilderList() {
    return job_;
  }
  /**
   * <code>repeated .oneflow.Job job = 1;</code>
   */
  public int getJobCount() {
    return job_.size();
  }
  /**
   * <code>repeated .oneflow.Job job = 1;</code>
   */
  public org.oneflow.core.job.Job getJob(int index) {
    return job_.get(index);
  }
  /**
   * <code>repeated .oneflow.Job job = 1;</code>
   */
  public org.oneflow.core.job.JobOrBuilder getJobOrBuilder(
      int index) {
    return job_.get(index);
  }

  public static final int INTER_JOB_REUSE_MEM_STRATEGY_FIELD_NUMBER = 5;
  private org.oneflow.core.job.InterJobReuseMemStrategy interJobReuseMemStrategy_;
  /**
   * <code>optional .oneflow.InterJobReuseMemStrategy inter_job_reuse_mem_strategy = 5;</code>
   */
  public boolean hasInterJobReuseMemStrategy() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>optional .oneflow.InterJobReuseMemStrategy inter_job_reuse_mem_strategy = 5;</code>
   */
  public org.oneflow.core.job.InterJobReuseMemStrategy getInterJobReuseMemStrategy() {
    return interJobReuseMemStrategy_ == null ? org.oneflow.core.job.InterJobReuseMemStrategy.getDefaultInstance() : interJobReuseMemStrategy_;
  }
  /**
   * <code>optional .oneflow.InterJobReuseMemStrategy inter_job_reuse_mem_strategy = 5;</code>
   */
  public org.oneflow.core.job.InterJobReuseMemStrategyOrBuilder getInterJobReuseMemStrategyOrBuilder() {
    return interJobReuseMemStrategy_ == null ? org.oneflow.core.job.InterJobReuseMemStrategy.getDefaultInstance() : interJobReuseMemStrategy_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    for (int i = 0; i < getJobCount(); i++) {
      if (!getJob(i).isInitialized()) {
        memoizedIsInitialized = 0;
        return false;
      }
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    for (int i = 0; i < job_.size(); i++) {
      output.writeMessage(1, job_.get(i));
    }
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      output.writeMessage(5, getInterJobReuseMemStrategy());
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    for (int i = 0; i < job_.size(); i++) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(1, job_.get(i));
    }
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(5, getInterJobReuseMemStrategy());
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
    if (!(obj instanceof org.oneflow.core.job.JobSet)) {
      return super.equals(obj);
    }
    org.oneflow.core.job.JobSet other = (org.oneflow.core.job.JobSet) obj;

    boolean result = true;
    result = result && getJobList()
        .equals(other.getJobList());
    result = result && (hasInterJobReuseMemStrategy() == other.hasInterJobReuseMemStrategy());
    if (hasInterJobReuseMemStrategy()) {
      result = result && getInterJobReuseMemStrategy()
          .equals(other.getInterJobReuseMemStrategy());
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
    if (getJobCount() > 0) {
      hash = (37 * hash) + JOB_FIELD_NUMBER;
      hash = (53 * hash) + getJobList().hashCode();
    }
    if (hasInterJobReuseMemStrategy()) {
      hash = (37 * hash) + INTER_JOB_REUSE_MEM_STRATEGY_FIELD_NUMBER;
      hash = (53 * hash) + getInterJobReuseMemStrategy().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.job.JobSet parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.JobSet parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.JobSet parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.job.JobSet parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.job.JobSet parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.JobSet parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.JobSet parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.JobSet parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.job.JobSet parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.job.JobSet parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.job.JobSet prototype) {
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
   * Protobuf type {@code oneflow.JobSet}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.JobSet)
      org.oneflow.core.job.JobSetOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.job.JobSetOuterClass.internal_static_oneflow_JobSet_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.job.JobSetOuterClass.internal_static_oneflow_JobSet_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.job.JobSet.class, org.oneflow.core.job.JobSet.Builder.class);
    }

    // Construct using org.oneflow.core.job.JobSet.newBuilder()
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
        getJobFieldBuilder();
        getInterJobReuseMemStrategyFieldBuilder();
      }
    }
    public Builder clear() {
      super.clear();
      if (jobBuilder_ == null) {
        job_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000001);
      } else {
        jobBuilder_.clear();
      }
      if (interJobReuseMemStrategyBuilder_ == null) {
        interJobReuseMemStrategy_ = null;
      } else {
        interJobReuseMemStrategyBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000002);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.job.JobSetOuterClass.internal_static_oneflow_JobSet_descriptor;
    }

    public org.oneflow.core.job.JobSet getDefaultInstanceForType() {
      return org.oneflow.core.job.JobSet.getDefaultInstance();
    }

    public org.oneflow.core.job.JobSet build() {
      org.oneflow.core.job.JobSet result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.job.JobSet buildPartial() {
      org.oneflow.core.job.JobSet result = new org.oneflow.core.job.JobSet(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (jobBuilder_ == null) {
        if (((bitField0_ & 0x00000001) == 0x00000001)) {
          job_ = java.util.Collections.unmodifiableList(job_);
          bitField0_ = (bitField0_ & ~0x00000001);
        }
        result.job_ = job_;
      } else {
        result.job_ = jobBuilder_.build();
      }
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000001;
      }
      if (interJobReuseMemStrategyBuilder_ == null) {
        result.interJobReuseMemStrategy_ = interJobReuseMemStrategy_;
      } else {
        result.interJobReuseMemStrategy_ = interJobReuseMemStrategyBuilder_.build();
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
      if (other instanceof org.oneflow.core.job.JobSet) {
        return mergeFrom((org.oneflow.core.job.JobSet)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.job.JobSet other) {
      if (other == org.oneflow.core.job.JobSet.getDefaultInstance()) return this;
      if (jobBuilder_ == null) {
        if (!other.job_.isEmpty()) {
          if (job_.isEmpty()) {
            job_ = other.job_;
            bitField0_ = (bitField0_ & ~0x00000001);
          } else {
            ensureJobIsMutable();
            job_.addAll(other.job_);
          }
          onChanged();
        }
      } else {
        if (!other.job_.isEmpty()) {
          if (jobBuilder_.isEmpty()) {
            jobBuilder_.dispose();
            jobBuilder_ = null;
            job_ = other.job_;
            bitField0_ = (bitField0_ & ~0x00000001);
            jobBuilder_ = 
              com.google.protobuf.GeneratedMessageV3.alwaysUseFieldBuilders ?
                 getJobFieldBuilder() : null;
          } else {
            jobBuilder_.addAllMessages(other.job_);
          }
        }
      }
      if (other.hasInterJobReuseMemStrategy()) {
        mergeInterJobReuseMemStrategy(other.getInterJobReuseMemStrategy());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      for (int i = 0; i < getJobCount(); i++) {
        if (!getJob(i).isInitialized()) {
          return false;
        }
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.job.JobSet parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.job.JobSet) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private java.util.List<org.oneflow.core.job.Job> job_ =
      java.util.Collections.emptyList();
    private void ensureJobIsMutable() {
      if (!((bitField0_ & 0x00000001) == 0x00000001)) {
        job_ = new java.util.ArrayList<org.oneflow.core.job.Job>(job_);
        bitField0_ |= 0x00000001;
       }
    }

    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.oneflow.core.job.Job, org.oneflow.core.job.Job.Builder, org.oneflow.core.job.JobOrBuilder> jobBuilder_;

    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public java.util.List<org.oneflow.core.job.Job> getJobList() {
      if (jobBuilder_ == null) {
        return java.util.Collections.unmodifiableList(job_);
      } else {
        return jobBuilder_.getMessageList();
      }
    }
    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public int getJobCount() {
      if (jobBuilder_ == null) {
        return job_.size();
      } else {
        return jobBuilder_.getCount();
      }
    }
    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public org.oneflow.core.job.Job getJob(int index) {
      if (jobBuilder_ == null) {
        return job_.get(index);
      } else {
        return jobBuilder_.getMessage(index);
      }
    }
    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public Builder setJob(
        int index, org.oneflow.core.job.Job value) {
      if (jobBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureJobIsMutable();
        job_.set(index, value);
        onChanged();
      } else {
        jobBuilder_.setMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public Builder setJob(
        int index, org.oneflow.core.job.Job.Builder builderForValue) {
      if (jobBuilder_ == null) {
        ensureJobIsMutable();
        job_.set(index, builderForValue.build());
        onChanged();
      } else {
        jobBuilder_.setMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public Builder addJob(org.oneflow.core.job.Job value) {
      if (jobBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureJobIsMutable();
        job_.add(value);
        onChanged();
      } else {
        jobBuilder_.addMessage(value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public Builder addJob(
        int index, org.oneflow.core.job.Job value) {
      if (jobBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        ensureJobIsMutable();
        job_.add(index, value);
        onChanged();
      } else {
        jobBuilder_.addMessage(index, value);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public Builder addJob(
        org.oneflow.core.job.Job.Builder builderForValue) {
      if (jobBuilder_ == null) {
        ensureJobIsMutable();
        job_.add(builderForValue.build());
        onChanged();
      } else {
        jobBuilder_.addMessage(builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public Builder addJob(
        int index, org.oneflow.core.job.Job.Builder builderForValue) {
      if (jobBuilder_ == null) {
        ensureJobIsMutable();
        job_.add(index, builderForValue.build());
        onChanged();
      } else {
        jobBuilder_.addMessage(index, builderForValue.build());
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public Builder addAllJob(
        java.lang.Iterable<? extends org.oneflow.core.job.Job> values) {
      if (jobBuilder_ == null) {
        ensureJobIsMutable();
        com.google.protobuf.AbstractMessageLite.Builder.addAll(
            values, job_);
        onChanged();
      } else {
        jobBuilder_.addAllMessages(values);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public Builder clearJob() {
      if (jobBuilder_ == null) {
        job_ = java.util.Collections.emptyList();
        bitField0_ = (bitField0_ & ~0x00000001);
        onChanged();
      } else {
        jobBuilder_.clear();
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public Builder removeJob(int index) {
      if (jobBuilder_ == null) {
        ensureJobIsMutable();
        job_.remove(index);
        onChanged();
      } else {
        jobBuilder_.remove(index);
      }
      return this;
    }
    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public org.oneflow.core.job.Job.Builder getJobBuilder(
        int index) {
      return getJobFieldBuilder().getBuilder(index);
    }
    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public org.oneflow.core.job.JobOrBuilder getJobOrBuilder(
        int index) {
      if (jobBuilder_ == null) {
        return job_.get(index);  } else {
        return jobBuilder_.getMessageOrBuilder(index);
      }
    }
    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public java.util.List<? extends org.oneflow.core.job.JobOrBuilder> 
         getJobOrBuilderList() {
      if (jobBuilder_ != null) {
        return jobBuilder_.getMessageOrBuilderList();
      } else {
        return java.util.Collections.unmodifiableList(job_);
      }
    }
    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public org.oneflow.core.job.Job.Builder addJobBuilder() {
      return getJobFieldBuilder().addBuilder(
          org.oneflow.core.job.Job.getDefaultInstance());
    }
    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public org.oneflow.core.job.Job.Builder addJobBuilder(
        int index) {
      return getJobFieldBuilder().addBuilder(
          index, org.oneflow.core.job.Job.getDefaultInstance());
    }
    /**
     * <code>repeated .oneflow.Job job = 1;</code>
     */
    public java.util.List<org.oneflow.core.job.Job.Builder> 
         getJobBuilderList() {
      return getJobFieldBuilder().getBuilderList();
    }
    private com.google.protobuf.RepeatedFieldBuilderV3<
        org.oneflow.core.job.Job, org.oneflow.core.job.Job.Builder, org.oneflow.core.job.JobOrBuilder> 
        getJobFieldBuilder() {
      if (jobBuilder_ == null) {
        jobBuilder_ = new com.google.protobuf.RepeatedFieldBuilderV3<
            org.oneflow.core.job.Job, org.oneflow.core.job.Job.Builder, org.oneflow.core.job.JobOrBuilder>(
                job_,
                ((bitField0_ & 0x00000001) == 0x00000001),
                getParentForChildren(),
                isClean());
        job_ = null;
      }
      return jobBuilder_;
    }

    private org.oneflow.core.job.InterJobReuseMemStrategy interJobReuseMemStrategy_ = null;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.job.InterJobReuseMemStrategy, org.oneflow.core.job.InterJobReuseMemStrategy.Builder, org.oneflow.core.job.InterJobReuseMemStrategyOrBuilder> interJobReuseMemStrategyBuilder_;
    /**
     * <code>optional .oneflow.InterJobReuseMemStrategy inter_job_reuse_mem_strategy = 5;</code>
     */
    public boolean hasInterJobReuseMemStrategy() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>optional .oneflow.InterJobReuseMemStrategy inter_job_reuse_mem_strategy = 5;</code>
     */
    public org.oneflow.core.job.InterJobReuseMemStrategy getInterJobReuseMemStrategy() {
      if (interJobReuseMemStrategyBuilder_ == null) {
        return interJobReuseMemStrategy_ == null ? org.oneflow.core.job.InterJobReuseMemStrategy.getDefaultInstance() : interJobReuseMemStrategy_;
      } else {
        return interJobReuseMemStrategyBuilder_.getMessage();
      }
    }
    /**
     * <code>optional .oneflow.InterJobReuseMemStrategy inter_job_reuse_mem_strategy = 5;</code>
     */
    public Builder setInterJobReuseMemStrategy(org.oneflow.core.job.InterJobReuseMemStrategy value) {
      if (interJobReuseMemStrategyBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        interJobReuseMemStrategy_ = value;
        onChanged();
      } else {
        interJobReuseMemStrategyBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000002;
      return this;
    }
    /**
     * <code>optional .oneflow.InterJobReuseMemStrategy inter_job_reuse_mem_strategy = 5;</code>
     */
    public Builder setInterJobReuseMemStrategy(
        org.oneflow.core.job.InterJobReuseMemStrategy.Builder builderForValue) {
      if (interJobReuseMemStrategyBuilder_ == null) {
        interJobReuseMemStrategy_ = builderForValue.build();
        onChanged();
      } else {
        interJobReuseMemStrategyBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000002;
      return this;
    }
    /**
     * <code>optional .oneflow.InterJobReuseMemStrategy inter_job_reuse_mem_strategy = 5;</code>
     */
    public Builder mergeInterJobReuseMemStrategy(org.oneflow.core.job.InterJobReuseMemStrategy value) {
      if (interJobReuseMemStrategyBuilder_ == null) {
        if (((bitField0_ & 0x00000002) == 0x00000002) &&
            interJobReuseMemStrategy_ != null &&
            interJobReuseMemStrategy_ != org.oneflow.core.job.InterJobReuseMemStrategy.getDefaultInstance()) {
          interJobReuseMemStrategy_ =
            org.oneflow.core.job.InterJobReuseMemStrategy.newBuilder(interJobReuseMemStrategy_).mergeFrom(value).buildPartial();
        } else {
          interJobReuseMemStrategy_ = value;
        }
        onChanged();
      } else {
        interJobReuseMemStrategyBuilder_.mergeFrom(value);
      }
      bitField0_ |= 0x00000002;
      return this;
    }
    /**
     * <code>optional .oneflow.InterJobReuseMemStrategy inter_job_reuse_mem_strategy = 5;</code>
     */
    public Builder clearInterJobReuseMemStrategy() {
      if (interJobReuseMemStrategyBuilder_ == null) {
        interJobReuseMemStrategy_ = null;
        onChanged();
      } else {
        interJobReuseMemStrategyBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000002);
      return this;
    }
    /**
     * <code>optional .oneflow.InterJobReuseMemStrategy inter_job_reuse_mem_strategy = 5;</code>
     */
    public org.oneflow.core.job.InterJobReuseMemStrategy.Builder getInterJobReuseMemStrategyBuilder() {
      bitField0_ |= 0x00000002;
      onChanged();
      return getInterJobReuseMemStrategyFieldBuilder().getBuilder();
    }
    /**
     * <code>optional .oneflow.InterJobReuseMemStrategy inter_job_reuse_mem_strategy = 5;</code>
     */
    public org.oneflow.core.job.InterJobReuseMemStrategyOrBuilder getInterJobReuseMemStrategyOrBuilder() {
      if (interJobReuseMemStrategyBuilder_ != null) {
        return interJobReuseMemStrategyBuilder_.getMessageOrBuilder();
      } else {
        return interJobReuseMemStrategy_ == null ?
            org.oneflow.core.job.InterJobReuseMemStrategy.getDefaultInstance() : interJobReuseMemStrategy_;
      }
    }
    /**
     * <code>optional .oneflow.InterJobReuseMemStrategy inter_job_reuse_mem_strategy = 5;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.job.InterJobReuseMemStrategy, org.oneflow.core.job.InterJobReuseMemStrategy.Builder, org.oneflow.core.job.InterJobReuseMemStrategyOrBuilder> 
        getInterJobReuseMemStrategyFieldBuilder() {
      if (interJobReuseMemStrategyBuilder_ == null) {
        interJobReuseMemStrategyBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.oneflow.core.job.InterJobReuseMemStrategy, org.oneflow.core.job.InterJobReuseMemStrategy.Builder, org.oneflow.core.job.InterJobReuseMemStrategyOrBuilder>(
                getInterJobReuseMemStrategy(),
                getParentForChildren(),
                isClean());
        interJobReuseMemStrategy_ = null;
      }
      return interJobReuseMemStrategyBuilder_;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:oneflow.JobSet)
  }

  // @@protoc_insertion_point(class_scope:oneflow.JobSet)
  private static final org.oneflow.core.job.JobSet DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.job.JobSet();
  }

  public static org.oneflow.core.job.JobSet getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<JobSet>
      PARSER = new com.google.protobuf.AbstractParser<JobSet>() {
    public JobSet parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new JobSet(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<JobSet> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<JobSet> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.job.JobSet getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

