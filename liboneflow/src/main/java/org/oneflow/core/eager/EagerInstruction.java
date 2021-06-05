// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/eager/eager_instruction.proto

package org.oneflow.core.eager;

/**
 * Protobuf type {@code oneflow.vm.EagerInstruction}
 */
public  final class EagerInstruction extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:oneflow.vm.EagerInstruction)
    EagerInstructionOrBuilder {
  // Use EagerInstruction.newBuilder() to construct.
  private EagerInstruction(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private EagerInstruction() {
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private EagerInstruction(
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
            oneflow.vm.Instruction.InstructionListProto.Builder subBuilder = null;
            if (((bitField0_ & 0x00000001) == 0x00000001)) {
              subBuilder = instructionList_.toBuilder();
            }
            instructionList_ = input.readMessage(oneflow.vm.Instruction.InstructionListProto.PARSER, extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(instructionList_);
              instructionList_ = subBuilder.buildPartial();
            }
            bitField0_ |= 0x00000001;
            break;
          }
          case 18: {
            org.oneflow.core.eager.EagerSymbolList.Builder subBuilder = null;
            if (((bitField0_ & 0x00000002) == 0x00000002)) {
              subBuilder = eagerSymbolList_.toBuilder();
            }
            eagerSymbolList_ = input.readMessage(org.oneflow.core.eager.EagerSymbolList.PARSER, extensionRegistry);
            if (subBuilder != null) {
              subBuilder.mergeFrom(eagerSymbolList_);
              eagerSymbolList_ = subBuilder.buildPartial();
            }
            bitField0_ |= 0x00000002;
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
    return org.oneflow.core.eager.EagerInstructionOuterClass.internal_static_oneflow_vm_EagerInstruction_descriptor;
  }

  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.oneflow.core.eager.EagerInstructionOuterClass.internal_static_oneflow_vm_EagerInstruction_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.oneflow.core.eager.EagerInstruction.class, org.oneflow.core.eager.EagerInstruction.Builder.class);
  }

  private int bitField0_;
  public static final int INSTRUCTION_LIST_FIELD_NUMBER = 1;
  private oneflow.vm.Instruction.InstructionListProto instructionList_;
  /**
   * <code>optional .oneflow.vm.InstructionListProto instruction_list = 1;</code>
   */
  public boolean hasInstructionList() {
    return ((bitField0_ & 0x00000001) == 0x00000001);
  }
  /**
   * <code>optional .oneflow.vm.InstructionListProto instruction_list = 1;</code>
   */
  public oneflow.vm.Instruction.InstructionListProto getInstructionList() {
    return instructionList_ == null ? oneflow.vm.Instruction.InstructionListProto.getDefaultInstance() : instructionList_;
  }
  /**
   * <code>optional .oneflow.vm.InstructionListProto instruction_list = 1;</code>
   */
  public oneflow.vm.Instruction.InstructionListProtoOrBuilder getInstructionListOrBuilder() {
    return instructionList_ == null ? oneflow.vm.Instruction.InstructionListProto.getDefaultInstance() : instructionList_;
  }

  public static final int EAGER_SYMBOL_LIST_FIELD_NUMBER = 2;
  private org.oneflow.core.eager.EagerSymbolList eagerSymbolList_;
  /**
   * <code>optional .oneflow.vm.EagerSymbolList eager_symbol_list = 2;</code>
   */
  public boolean hasEagerSymbolList() {
    return ((bitField0_ & 0x00000002) == 0x00000002);
  }
  /**
   * <code>optional .oneflow.vm.EagerSymbolList eager_symbol_list = 2;</code>
   */
  public org.oneflow.core.eager.EagerSymbolList getEagerSymbolList() {
    return eagerSymbolList_ == null ? org.oneflow.core.eager.EagerSymbolList.getDefaultInstance() : eagerSymbolList_;
  }
  /**
   * <code>optional .oneflow.vm.EagerSymbolList eager_symbol_list = 2;</code>
   */
  public org.oneflow.core.eager.EagerSymbolListOrBuilder getEagerSymbolListOrBuilder() {
    return eagerSymbolList_ == null ? org.oneflow.core.eager.EagerSymbolList.getDefaultInstance() : eagerSymbolList_;
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    if (hasInstructionList()) {
      if (!getInstructionList().isInitialized()) {
        memoizedIsInitialized = 0;
        return false;
      }
    }
    if (hasEagerSymbolList()) {
      if (!getEagerSymbolList().isInitialized()) {
        memoizedIsInitialized = 0;
        return false;
      }
    }
    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      output.writeMessage(1, getInstructionList());
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      output.writeMessage(2, getEagerSymbolList());
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    if (((bitField0_ & 0x00000001) == 0x00000001)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(1, getInstructionList());
    }
    if (((bitField0_ & 0x00000002) == 0x00000002)) {
      size += com.google.protobuf.CodedOutputStream
        .computeMessageSize(2, getEagerSymbolList());
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
    if (!(obj instanceof org.oneflow.core.eager.EagerInstruction)) {
      return super.equals(obj);
    }
    org.oneflow.core.eager.EagerInstruction other = (org.oneflow.core.eager.EagerInstruction) obj;

    boolean result = true;
    result = result && (hasInstructionList() == other.hasInstructionList());
    if (hasInstructionList()) {
      result = result && getInstructionList()
          .equals(other.getInstructionList());
    }
    result = result && (hasEagerSymbolList() == other.hasEagerSymbolList());
    if (hasEagerSymbolList()) {
      result = result && getEagerSymbolList()
          .equals(other.getEagerSymbolList());
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
    if (hasInstructionList()) {
      hash = (37 * hash) + INSTRUCTION_LIST_FIELD_NUMBER;
      hash = (53 * hash) + getInstructionList().hashCode();
    }
    if (hasEagerSymbolList()) {
      hash = (37 * hash) + EAGER_SYMBOL_LIST_FIELD_NUMBER;
      hash = (53 * hash) + getEagerSymbolList().hashCode();
    }
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.oneflow.core.eager.EagerInstruction parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.eager.EagerInstruction parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.eager.EagerInstruction parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.oneflow.core.eager.EagerInstruction parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.oneflow.core.eager.EagerInstruction parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.eager.EagerInstruction parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.eager.EagerInstruction parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.oneflow.core.eager.EagerInstruction parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.oneflow.core.eager.EagerInstruction parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.oneflow.core.eager.EagerInstruction parseFrom(
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
  public static Builder newBuilder(org.oneflow.core.eager.EagerInstruction prototype) {
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
   * Protobuf type {@code oneflow.vm.EagerInstruction}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:oneflow.vm.EagerInstruction)
      org.oneflow.core.eager.EagerInstructionOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.oneflow.core.eager.EagerInstructionOuterClass.internal_static_oneflow_vm_EagerInstruction_descriptor;
    }

    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.eager.EagerInstructionOuterClass.internal_static_oneflow_vm_EagerInstruction_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.eager.EagerInstruction.class, org.oneflow.core.eager.EagerInstruction.Builder.class);
    }

    // Construct using org.oneflow.core.eager.EagerInstruction.newBuilder()
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
        getInstructionListFieldBuilder();
        getEagerSymbolListFieldBuilder();
      }
    }
    public Builder clear() {
      super.clear();
      if (instructionListBuilder_ == null) {
        instructionList_ = null;
      } else {
        instructionListBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000001);
      if (eagerSymbolListBuilder_ == null) {
        eagerSymbolList_ = null;
      } else {
        eagerSymbolListBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000002);
      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.oneflow.core.eager.EagerInstructionOuterClass.internal_static_oneflow_vm_EagerInstruction_descriptor;
    }

    public org.oneflow.core.eager.EagerInstruction getDefaultInstanceForType() {
      return org.oneflow.core.eager.EagerInstruction.getDefaultInstance();
    }

    public org.oneflow.core.eager.EagerInstruction build() {
      org.oneflow.core.eager.EagerInstruction result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.oneflow.core.eager.EagerInstruction buildPartial() {
      org.oneflow.core.eager.EagerInstruction result = new org.oneflow.core.eager.EagerInstruction(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
        to_bitField0_ |= 0x00000001;
      }
      if (instructionListBuilder_ == null) {
        result.instructionList_ = instructionList_;
      } else {
        result.instructionList_ = instructionListBuilder_.build();
      }
      if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
        to_bitField0_ |= 0x00000002;
      }
      if (eagerSymbolListBuilder_ == null) {
        result.eagerSymbolList_ = eagerSymbolList_;
      } else {
        result.eagerSymbolList_ = eagerSymbolListBuilder_.build();
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
      if (other instanceof org.oneflow.core.eager.EagerInstruction) {
        return mergeFrom((org.oneflow.core.eager.EagerInstruction)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.oneflow.core.eager.EagerInstruction other) {
      if (other == org.oneflow.core.eager.EagerInstruction.getDefaultInstance()) return this;
      if (other.hasInstructionList()) {
        mergeInstructionList(other.getInstructionList());
      }
      if (other.hasEagerSymbolList()) {
        mergeEagerSymbolList(other.getEagerSymbolList());
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      if (hasInstructionList()) {
        if (!getInstructionList().isInitialized()) {
          return false;
        }
      }
      if (hasEagerSymbolList()) {
        if (!getEagerSymbolList().isInitialized()) {
          return false;
        }
      }
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.oneflow.core.eager.EagerInstruction parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.oneflow.core.eager.EagerInstruction) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private oneflow.vm.Instruction.InstructionListProto instructionList_ = null;
    private com.google.protobuf.SingleFieldBuilderV3<
        oneflow.vm.Instruction.InstructionListProto, oneflow.vm.Instruction.InstructionListProto.Builder, oneflow.vm.Instruction.InstructionListProtoOrBuilder> instructionListBuilder_;
    /**
     * <code>optional .oneflow.vm.InstructionListProto instruction_list = 1;</code>
     */
    public boolean hasInstructionList() {
      return ((bitField0_ & 0x00000001) == 0x00000001);
    }
    /**
     * <code>optional .oneflow.vm.InstructionListProto instruction_list = 1;</code>
     */
    public oneflow.vm.Instruction.InstructionListProto getInstructionList() {
      if (instructionListBuilder_ == null) {
        return instructionList_ == null ? oneflow.vm.Instruction.InstructionListProto.getDefaultInstance() : instructionList_;
      } else {
        return instructionListBuilder_.getMessage();
      }
    }
    /**
     * <code>optional .oneflow.vm.InstructionListProto instruction_list = 1;</code>
     */
    public Builder setInstructionList(oneflow.vm.Instruction.InstructionListProto value) {
      if (instructionListBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        instructionList_ = value;
        onChanged();
      } else {
        instructionListBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>optional .oneflow.vm.InstructionListProto instruction_list = 1;</code>
     */
    public Builder setInstructionList(
        oneflow.vm.Instruction.InstructionListProto.Builder builderForValue) {
      if (instructionListBuilder_ == null) {
        instructionList_ = builderForValue.build();
        onChanged();
      } else {
        instructionListBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>optional .oneflow.vm.InstructionListProto instruction_list = 1;</code>
     */
    public Builder mergeInstructionList(oneflow.vm.Instruction.InstructionListProto value) {
      if (instructionListBuilder_ == null) {
        if (((bitField0_ & 0x00000001) == 0x00000001) &&
            instructionList_ != null &&
            instructionList_ != oneflow.vm.Instruction.InstructionListProto.getDefaultInstance()) {
          instructionList_ =
            oneflow.vm.Instruction.InstructionListProto.newBuilder(instructionList_).mergeFrom(value).buildPartial();
        } else {
          instructionList_ = value;
        }
        onChanged();
      } else {
        instructionListBuilder_.mergeFrom(value);
      }
      bitField0_ |= 0x00000001;
      return this;
    }
    /**
     * <code>optional .oneflow.vm.InstructionListProto instruction_list = 1;</code>
     */
    public Builder clearInstructionList() {
      if (instructionListBuilder_ == null) {
        instructionList_ = null;
        onChanged();
      } else {
        instructionListBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000001);
      return this;
    }
    /**
     * <code>optional .oneflow.vm.InstructionListProto instruction_list = 1;</code>
     */
    public oneflow.vm.Instruction.InstructionListProto.Builder getInstructionListBuilder() {
      bitField0_ |= 0x00000001;
      onChanged();
      return getInstructionListFieldBuilder().getBuilder();
    }
    /**
     * <code>optional .oneflow.vm.InstructionListProto instruction_list = 1;</code>
     */
    public oneflow.vm.Instruction.InstructionListProtoOrBuilder getInstructionListOrBuilder() {
      if (instructionListBuilder_ != null) {
        return instructionListBuilder_.getMessageOrBuilder();
      } else {
        return instructionList_ == null ?
            oneflow.vm.Instruction.InstructionListProto.getDefaultInstance() : instructionList_;
      }
    }
    /**
     * <code>optional .oneflow.vm.InstructionListProto instruction_list = 1;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        oneflow.vm.Instruction.InstructionListProto, oneflow.vm.Instruction.InstructionListProto.Builder, oneflow.vm.Instruction.InstructionListProtoOrBuilder> 
        getInstructionListFieldBuilder() {
      if (instructionListBuilder_ == null) {
        instructionListBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            oneflow.vm.Instruction.InstructionListProto, oneflow.vm.Instruction.InstructionListProto.Builder, oneflow.vm.Instruction.InstructionListProtoOrBuilder>(
                getInstructionList(),
                getParentForChildren(),
                isClean());
        instructionList_ = null;
      }
      return instructionListBuilder_;
    }

    private org.oneflow.core.eager.EagerSymbolList eagerSymbolList_ = null;
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.eager.EagerSymbolList, org.oneflow.core.eager.EagerSymbolList.Builder, org.oneflow.core.eager.EagerSymbolListOrBuilder> eagerSymbolListBuilder_;
    /**
     * <code>optional .oneflow.vm.EagerSymbolList eager_symbol_list = 2;</code>
     */
    public boolean hasEagerSymbolList() {
      return ((bitField0_ & 0x00000002) == 0x00000002);
    }
    /**
     * <code>optional .oneflow.vm.EagerSymbolList eager_symbol_list = 2;</code>
     */
    public org.oneflow.core.eager.EagerSymbolList getEagerSymbolList() {
      if (eagerSymbolListBuilder_ == null) {
        return eagerSymbolList_ == null ? org.oneflow.core.eager.EagerSymbolList.getDefaultInstance() : eagerSymbolList_;
      } else {
        return eagerSymbolListBuilder_.getMessage();
      }
    }
    /**
     * <code>optional .oneflow.vm.EagerSymbolList eager_symbol_list = 2;</code>
     */
    public Builder setEagerSymbolList(org.oneflow.core.eager.EagerSymbolList value) {
      if (eagerSymbolListBuilder_ == null) {
        if (value == null) {
          throw new NullPointerException();
        }
        eagerSymbolList_ = value;
        onChanged();
      } else {
        eagerSymbolListBuilder_.setMessage(value);
      }
      bitField0_ |= 0x00000002;
      return this;
    }
    /**
     * <code>optional .oneflow.vm.EagerSymbolList eager_symbol_list = 2;</code>
     */
    public Builder setEagerSymbolList(
        org.oneflow.core.eager.EagerSymbolList.Builder builderForValue) {
      if (eagerSymbolListBuilder_ == null) {
        eagerSymbolList_ = builderForValue.build();
        onChanged();
      } else {
        eagerSymbolListBuilder_.setMessage(builderForValue.build());
      }
      bitField0_ |= 0x00000002;
      return this;
    }
    /**
     * <code>optional .oneflow.vm.EagerSymbolList eager_symbol_list = 2;</code>
     */
    public Builder mergeEagerSymbolList(org.oneflow.core.eager.EagerSymbolList value) {
      if (eagerSymbolListBuilder_ == null) {
        if (((bitField0_ & 0x00000002) == 0x00000002) &&
            eagerSymbolList_ != null &&
            eagerSymbolList_ != org.oneflow.core.eager.EagerSymbolList.getDefaultInstance()) {
          eagerSymbolList_ =
            org.oneflow.core.eager.EagerSymbolList.newBuilder(eagerSymbolList_).mergeFrom(value).buildPartial();
        } else {
          eagerSymbolList_ = value;
        }
        onChanged();
      } else {
        eagerSymbolListBuilder_.mergeFrom(value);
      }
      bitField0_ |= 0x00000002;
      return this;
    }
    /**
     * <code>optional .oneflow.vm.EagerSymbolList eager_symbol_list = 2;</code>
     */
    public Builder clearEagerSymbolList() {
      if (eagerSymbolListBuilder_ == null) {
        eagerSymbolList_ = null;
        onChanged();
      } else {
        eagerSymbolListBuilder_.clear();
      }
      bitField0_ = (bitField0_ & ~0x00000002);
      return this;
    }
    /**
     * <code>optional .oneflow.vm.EagerSymbolList eager_symbol_list = 2;</code>
     */
    public org.oneflow.core.eager.EagerSymbolList.Builder getEagerSymbolListBuilder() {
      bitField0_ |= 0x00000002;
      onChanged();
      return getEagerSymbolListFieldBuilder().getBuilder();
    }
    /**
     * <code>optional .oneflow.vm.EagerSymbolList eager_symbol_list = 2;</code>
     */
    public org.oneflow.core.eager.EagerSymbolListOrBuilder getEagerSymbolListOrBuilder() {
      if (eagerSymbolListBuilder_ != null) {
        return eagerSymbolListBuilder_.getMessageOrBuilder();
      } else {
        return eagerSymbolList_ == null ?
            org.oneflow.core.eager.EagerSymbolList.getDefaultInstance() : eagerSymbolList_;
      }
    }
    /**
     * <code>optional .oneflow.vm.EagerSymbolList eager_symbol_list = 2;</code>
     */
    private com.google.protobuf.SingleFieldBuilderV3<
        org.oneflow.core.eager.EagerSymbolList, org.oneflow.core.eager.EagerSymbolList.Builder, org.oneflow.core.eager.EagerSymbolListOrBuilder> 
        getEagerSymbolListFieldBuilder() {
      if (eagerSymbolListBuilder_ == null) {
        eagerSymbolListBuilder_ = new com.google.protobuf.SingleFieldBuilderV3<
            org.oneflow.core.eager.EagerSymbolList, org.oneflow.core.eager.EagerSymbolList.Builder, org.oneflow.core.eager.EagerSymbolListOrBuilder>(
                getEagerSymbolList(),
                getParentForChildren(),
                isClean());
        eagerSymbolList_ = null;
      }
      return eagerSymbolListBuilder_;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFields(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:oneflow.vm.EagerInstruction)
  }

  // @@protoc_insertion_point(class_scope:oneflow.vm.EagerInstruction)
  private static final org.oneflow.core.eager.EagerInstruction DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.oneflow.core.eager.EagerInstruction();
  }

  public static org.oneflow.core.eager.EagerInstruction getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  @java.lang.Deprecated public static final com.google.protobuf.Parser<EagerInstruction>
      PARSER = new com.google.protobuf.AbstractParser<EagerInstruction>() {
    public EagerInstruction parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
        return new EagerInstruction(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<EagerInstruction> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<EagerInstruction> getParserForType() {
    return PARSER;
  }

  public org.oneflow.core.eager.EagerInstruction getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

