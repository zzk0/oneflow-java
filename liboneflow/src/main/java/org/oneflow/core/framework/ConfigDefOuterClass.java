// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/framework/config_def.proto

package org.oneflow.core.framework;

public final class ConfigDefOuterClass {
  private ConfigDefOuterClass() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  /**
   * Protobuf enum {@code oneflow.ConfigDefType}
   */
  public enum ConfigDefType
      implements com.google.protobuf.ProtocolMessageEnum {
    /**
     * <code>kEnvConfigDefType = 1;</code>
     */
    kEnvConfigDefType(1),
    /**
     * <code>kSessionConfigDefType = 2;</code>
     */
    kSessionConfigDefType(2),
    /**
     * <code>kFunctionConfigDefType = 3;</code>
     */
    kFunctionConfigDefType(3),
    /**
     * <code>kScopeConfigDefType = 4;</code>
     */
    kScopeConfigDefType(4),
    ;

    /**
     * <code>kEnvConfigDefType = 1;</code>
     */
    public static final int kEnvConfigDefType_VALUE = 1;
    /**
     * <code>kSessionConfigDefType = 2;</code>
     */
    public static final int kSessionConfigDefType_VALUE = 2;
    /**
     * <code>kFunctionConfigDefType = 3;</code>
     */
    public static final int kFunctionConfigDefType_VALUE = 3;
    /**
     * <code>kScopeConfigDefType = 4;</code>
     */
    public static final int kScopeConfigDefType_VALUE = 4;


    public final int getNumber() {
      return value;
    }

    /**
     * @deprecated Use {@link #forNumber(int)} instead.
     */
    @java.lang.Deprecated
    public static ConfigDefType valueOf(int value) {
      return forNumber(value);
    }

    public static ConfigDefType forNumber(int value) {
      switch (value) {
        case 1: return kEnvConfigDefType;
        case 2: return kSessionConfigDefType;
        case 3: return kFunctionConfigDefType;
        case 4: return kScopeConfigDefType;
        default: return null;
      }
    }

    public static com.google.protobuf.Internal.EnumLiteMap<ConfigDefType>
        internalGetValueMap() {
      return internalValueMap;
    }
    private static final com.google.protobuf.Internal.EnumLiteMap<
        ConfigDefType> internalValueMap =
          new com.google.protobuf.Internal.EnumLiteMap<ConfigDefType>() {
            public ConfigDefType findValueByNumber(int number) {
              return ConfigDefType.forNumber(number);
            }
          };

    public final com.google.protobuf.Descriptors.EnumValueDescriptor
        getValueDescriptor() {
      return getDescriptor().getValues().get(ordinal());
    }
    public final com.google.protobuf.Descriptors.EnumDescriptor
        getDescriptorForType() {
      return getDescriptor();
    }
    public static final com.google.protobuf.Descriptors.EnumDescriptor
        getDescriptor() {
      return org.oneflow.core.framework.ConfigDefOuterClass.getDescriptor().getEnumTypes().get(0);
    }

    private static final ConfigDefType[] VALUES = values();

    public static ConfigDefType valueOf(
        com.google.protobuf.Descriptors.EnumValueDescriptor desc) {
      if (desc.getType() != getDescriptor()) {
        throw new java.lang.IllegalArgumentException(
          "EnumValueDescriptor is not for this type.");
      }
      return VALUES[desc.getIndex()];
    }

    private final int value;

    private ConfigDefType(int value) {
      this.value = value;
    }

    // @@protoc_insertion_point(enum_scope:oneflow.ConfigDefType)
  }

  public interface ConfigDefOrBuilder extends
      // @@protoc_insertion_point(interface_extends:oneflow.ConfigDef)
      com.google.protobuf.MessageOrBuilder {

    /**
     * <code>map&lt;string, .oneflow.AttrDef&gt; attr_name2attr_def = 1;</code>
     */
    int getAttrName2AttrDefCount();
    /**
     * <code>map&lt;string, .oneflow.AttrDef&gt; attr_name2attr_def = 1;</code>
     */
    boolean containsAttrName2AttrDef(
        java.lang.String key);
    /**
     * Use {@link #getAttrName2AttrDefMap()} instead.
     */
    @java.lang.Deprecated
    java.util.Map<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef>
    getAttrName2AttrDef();
    /**
     * <code>map&lt;string, .oneflow.AttrDef&gt; attr_name2attr_def = 1;</code>
     */
    java.util.Map<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef>
    getAttrName2AttrDefMap();
    /**
     * <code>map&lt;string, .oneflow.AttrDef&gt; attr_name2attr_def = 1;</code>
     */

    org.oneflow.core.framework.UserOpAttr.AttrDef getAttrName2AttrDefOrDefault(
        java.lang.String key,
        org.oneflow.core.framework.UserOpAttr.AttrDef defaultValue);
    /**
     * <code>map&lt;string, .oneflow.AttrDef&gt; attr_name2attr_def = 1;</code>
     */

    org.oneflow.core.framework.UserOpAttr.AttrDef getAttrName2AttrDefOrThrow(
        java.lang.String key);
  }
  /**
   * Protobuf type {@code oneflow.ConfigDef}
   */
  public  static final class ConfigDef extends
      com.google.protobuf.GeneratedMessageV3 implements
      // @@protoc_insertion_point(message_implements:oneflow.ConfigDef)
      ConfigDefOrBuilder {
    // Use ConfigDef.newBuilder() to construct.
    private ConfigDef(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
      super(builder);
    }
    private ConfigDef() {
    }

    @java.lang.Override
    public final com.google.protobuf.UnknownFieldSet
    getUnknownFields() {
      return this.unknownFields;
    }
    private ConfigDef(
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
                attrName2AttrDef_ = com.google.protobuf.MapField.newMapField(
                    AttrName2AttrDefDefaultEntryHolder.defaultEntry);
                mutable_bitField0_ |= 0x00000001;
              }
              com.google.protobuf.MapEntry<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef>
              attrName2AttrDef = input.readMessage(
                  AttrName2AttrDefDefaultEntryHolder.defaultEntry.getParserForType(), extensionRegistry);
              attrName2AttrDef_.getMutableMap().put(attrName2AttrDef.getKey(), attrName2AttrDef.getValue());
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
      return org.oneflow.core.framework.ConfigDefOuterClass.internal_static_oneflow_ConfigDef_descriptor;
    }

    @SuppressWarnings({"rawtypes"})
    protected com.google.protobuf.MapField internalGetMapField(
        int number) {
      switch (number) {
        case 1:
          return internalGetAttrName2AttrDef();
        default:
          throw new RuntimeException(
              "Invalid map field number: " + number);
      }
    }
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.oneflow.core.framework.ConfigDefOuterClass.internal_static_oneflow_ConfigDef_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef.class, org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef.Builder.class);
    }

    public static final int ATTR_NAME2ATTR_DEF_FIELD_NUMBER = 1;
    private static final class AttrName2AttrDefDefaultEntryHolder {
      static final com.google.protobuf.MapEntry<
          java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef> defaultEntry =
              com.google.protobuf.MapEntry
              .<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef>newDefaultInstance(
                  org.oneflow.core.framework.ConfigDefOuterClass.internal_static_oneflow_ConfigDef_AttrName2attrDefEntry_descriptor, 
                  com.google.protobuf.WireFormat.FieldType.STRING,
                  "",
                  com.google.protobuf.WireFormat.FieldType.MESSAGE,
                  org.oneflow.core.framework.UserOpAttr.AttrDef.getDefaultInstance());
    }
    private com.google.protobuf.MapField<
        java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef> attrName2AttrDef_;
    private com.google.protobuf.MapField<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef>
    internalGetAttrName2AttrDef() {
      if (attrName2AttrDef_ == null) {
        return com.google.protobuf.MapField.emptyMapField(
            AttrName2AttrDefDefaultEntryHolder.defaultEntry);
      }
      return attrName2AttrDef_;
    }

    public int getAttrName2AttrDefCount() {
      return internalGetAttrName2AttrDef().getMap().size();
    }
    /**
     * <code>map&lt;string, .oneflow.AttrDef&gt; attr_name2attr_def = 1;</code>
     */

    public boolean containsAttrName2AttrDef(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      return internalGetAttrName2AttrDef().getMap().containsKey(key);
    }
    /**
     * Use {@link #getAttrName2AttrDefMap()} instead.
     */
    @java.lang.Deprecated
    public java.util.Map<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef> getAttrName2AttrDef() {
      return getAttrName2AttrDefMap();
    }
    /**
     * <code>map&lt;string, .oneflow.AttrDef&gt; attr_name2attr_def = 1;</code>
     */

    public java.util.Map<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef> getAttrName2AttrDefMap() {
      return internalGetAttrName2AttrDef().getMap();
    }
    /**
     * <code>map&lt;string, .oneflow.AttrDef&gt; attr_name2attr_def = 1;</code>
     */

    public org.oneflow.core.framework.UserOpAttr.AttrDef getAttrName2AttrDefOrDefault(
        java.lang.String key,
        org.oneflow.core.framework.UserOpAttr.AttrDef defaultValue) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      java.util.Map<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef> map =
          internalGetAttrName2AttrDef().getMap();
      return map.containsKey(key) ? map.get(key) : defaultValue;
    }
    /**
     * <code>map&lt;string, .oneflow.AttrDef&gt; attr_name2attr_def = 1;</code>
     */

    public org.oneflow.core.framework.UserOpAttr.AttrDef getAttrName2AttrDefOrThrow(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      java.util.Map<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef> map =
          internalGetAttrName2AttrDef().getMap();
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

      for (org.oneflow.core.framework.UserOpAttr.AttrDef item : getAttrName2AttrDef().values()) {
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
      for (java.util.Map.Entry<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef> entry
           : internalGetAttrName2AttrDef().getMap().entrySet()) {
        com.google.protobuf.MapEntry<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef>
        attrName2AttrDef = AttrName2AttrDefDefaultEntryHolder.defaultEntry.newBuilderForType()
            .setKey(entry.getKey())
            .setValue(entry.getValue())
            .build();
        output.writeMessage(1, attrName2AttrDef);
      }
      unknownFields.writeTo(output);
    }

    public int getSerializedSize() {
      int size = memoizedSize;
      if (size != -1) return size;

      size = 0;
      for (java.util.Map.Entry<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef> entry
           : internalGetAttrName2AttrDef().getMap().entrySet()) {
        com.google.protobuf.MapEntry<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef>
        attrName2AttrDef = AttrName2AttrDefDefaultEntryHolder.defaultEntry.newBuilderForType()
            .setKey(entry.getKey())
            .setValue(entry.getValue())
            .build();
        size += com.google.protobuf.CodedOutputStream
            .computeMessageSize(1, attrName2AttrDef);
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
      if (!(obj instanceof org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef)) {
        return super.equals(obj);
      }
      org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef other = (org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef) obj;

      boolean result = true;
      result = result && internalGetAttrName2AttrDef().equals(
          other.internalGetAttrName2AttrDef());
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
      if (!internalGetAttrName2AttrDef().getMap().isEmpty()) {
        hash = (37 * hash) + ATTR_NAME2ATTR_DEF_FIELD_NUMBER;
        hash = (53 * hash) + internalGetAttrName2AttrDef().hashCode();
      }
      hash = (29 * hash) + unknownFields.hashCode();
      memoizedHashCode = hash;
      return hash;
    }

    public static org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef parseFrom(
        com.google.protobuf.ByteString data)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data);
    }
    public static org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef parseFrom(
        com.google.protobuf.ByteString data,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data, extensionRegistry);
    }
    public static org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef parseFrom(byte[] data)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data);
    }
    public static org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef parseFrom(
        byte[] data,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return PARSER.parseFrom(data, extensionRegistry);
    }
    public static org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef parseFrom(java.io.InputStream input)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseWithIOException(PARSER, input);
    }
    public static org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef parseFrom(
        java.io.InputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseWithIOException(PARSER, input, extensionRegistry);
    }
    public static org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef parseDelimitedFrom(java.io.InputStream input)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseDelimitedWithIOException(PARSER, input);
    }
    public static org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef parseDelimitedFrom(
        java.io.InputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
    }
    public static org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef parseFrom(
        com.google.protobuf.CodedInputStream input)
        throws java.io.IOException {
      return com.google.protobuf.GeneratedMessageV3
          .parseWithIOException(PARSER, input);
    }
    public static org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef parseFrom(
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
    public static Builder newBuilder(org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef prototype) {
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
     * Protobuf type {@code oneflow.ConfigDef}
     */
    public static final class Builder extends
        com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
        // @@protoc_insertion_point(builder_implements:oneflow.ConfigDef)
        org.oneflow.core.framework.ConfigDefOuterClass.ConfigDefOrBuilder {
      public static final com.google.protobuf.Descriptors.Descriptor
          getDescriptor() {
        return org.oneflow.core.framework.ConfigDefOuterClass.internal_static_oneflow_ConfigDef_descriptor;
      }

      @SuppressWarnings({"rawtypes"})
      protected com.google.protobuf.MapField internalGetMapField(
          int number) {
        switch (number) {
          case 1:
            return internalGetAttrName2AttrDef();
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
            return internalGetMutableAttrName2AttrDef();
          default:
            throw new RuntimeException(
                "Invalid map field number: " + number);
        }
      }
      protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
          internalGetFieldAccessorTable() {
        return org.oneflow.core.framework.ConfigDefOuterClass.internal_static_oneflow_ConfigDef_fieldAccessorTable
            .ensureFieldAccessorsInitialized(
                org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef.class, org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef.Builder.class);
      }

      // Construct using org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef.newBuilder()
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
        internalGetMutableAttrName2AttrDef().clear();
        return this;
      }

      public com.google.protobuf.Descriptors.Descriptor
          getDescriptorForType() {
        return org.oneflow.core.framework.ConfigDefOuterClass.internal_static_oneflow_ConfigDef_descriptor;
      }

      public org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef getDefaultInstanceForType() {
        return org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef.getDefaultInstance();
      }

      public org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef build() {
        org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef result = buildPartial();
        if (!result.isInitialized()) {
          throw newUninitializedMessageException(result);
        }
        return result;
      }

      public org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef buildPartial() {
        org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef result = new org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef(this);
        int from_bitField0_ = bitField0_;
        result.attrName2AttrDef_ = internalGetAttrName2AttrDef();
        result.attrName2AttrDef_.makeImmutable();
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
        if (other instanceof org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef) {
          return mergeFrom((org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef)other);
        } else {
          super.mergeFrom(other);
          return this;
        }
      }

      public Builder mergeFrom(org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef other) {
        if (other == org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef.getDefaultInstance()) return this;
        internalGetMutableAttrName2AttrDef().mergeFrom(
            other.internalGetAttrName2AttrDef());
        this.mergeUnknownFields(other.unknownFields);
        onChanged();
        return this;
      }

      public final boolean isInitialized() {
        for (org.oneflow.core.framework.UserOpAttr.AttrDef item : getAttrName2AttrDef().values()) {
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
        org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef parsedMessage = null;
        try {
          parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
        } catch (com.google.protobuf.InvalidProtocolBufferException e) {
          parsedMessage = (org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef) e.getUnfinishedMessage();
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
          java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef> attrName2AttrDef_;
      private com.google.protobuf.MapField<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef>
      internalGetAttrName2AttrDef() {
        if (attrName2AttrDef_ == null) {
          return com.google.protobuf.MapField.emptyMapField(
              AttrName2AttrDefDefaultEntryHolder.defaultEntry);
        }
        return attrName2AttrDef_;
      }
      private com.google.protobuf.MapField<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef>
      internalGetMutableAttrName2AttrDef() {
        onChanged();;
        if (attrName2AttrDef_ == null) {
          attrName2AttrDef_ = com.google.protobuf.MapField.newMapField(
              AttrName2AttrDefDefaultEntryHolder.defaultEntry);
        }
        if (!attrName2AttrDef_.isMutable()) {
          attrName2AttrDef_ = attrName2AttrDef_.copy();
        }
        return attrName2AttrDef_;
      }

      public int getAttrName2AttrDefCount() {
        return internalGetAttrName2AttrDef().getMap().size();
      }
      /**
       * <code>map&lt;string, .oneflow.AttrDef&gt; attr_name2attr_def = 1;</code>
       */

      public boolean containsAttrName2AttrDef(
          java.lang.String key) {
        if (key == null) { throw new java.lang.NullPointerException(); }
        return internalGetAttrName2AttrDef().getMap().containsKey(key);
      }
      /**
       * Use {@link #getAttrName2AttrDefMap()} instead.
       */
      @java.lang.Deprecated
      public java.util.Map<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef> getAttrName2AttrDef() {
        return getAttrName2AttrDefMap();
      }
      /**
       * <code>map&lt;string, .oneflow.AttrDef&gt; attr_name2attr_def = 1;</code>
       */

      public java.util.Map<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef> getAttrName2AttrDefMap() {
        return internalGetAttrName2AttrDef().getMap();
      }
      /**
       * <code>map&lt;string, .oneflow.AttrDef&gt; attr_name2attr_def = 1;</code>
       */

      public org.oneflow.core.framework.UserOpAttr.AttrDef getAttrName2AttrDefOrDefault(
          java.lang.String key,
          org.oneflow.core.framework.UserOpAttr.AttrDef defaultValue) {
        if (key == null) { throw new java.lang.NullPointerException(); }
        java.util.Map<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef> map =
            internalGetAttrName2AttrDef().getMap();
        return map.containsKey(key) ? map.get(key) : defaultValue;
      }
      /**
       * <code>map&lt;string, .oneflow.AttrDef&gt; attr_name2attr_def = 1;</code>
       */

      public org.oneflow.core.framework.UserOpAttr.AttrDef getAttrName2AttrDefOrThrow(
          java.lang.String key) {
        if (key == null) { throw new java.lang.NullPointerException(); }
        java.util.Map<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef> map =
            internalGetAttrName2AttrDef().getMap();
        if (!map.containsKey(key)) {
          throw new java.lang.IllegalArgumentException();
        }
        return map.get(key);
      }

      public Builder clearAttrName2AttrDef() {
        getMutableAttrName2AttrDef().clear();
        return this;
      }
      /**
       * <code>map&lt;string, .oneflow.AttrDef&gt; attr_name2attr_def = 1;</code>
       */

      public Builder removeAttrName2AttrDef(
          java.lang.String key) {
        if (key == null) { throw new java.lang.NullPointerException(); }
        getMutableAttrName2AttrDef().remove(key);
        return this;
      }
      /**
       * Use alternate mutation accessors instead.
       */
      @java.lang.Deprecated
      public java.util.Map<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef>
      getMutableAttrName2AttrDef() {
        return internalGetMutableAttrName2AttrDef().getMutableMap();
      }
      /**
       * <code>map&lt;string, .oneflow.AttrDef&gt; attr_name2attr_def = 1;</code>
       */
      public Builder putAttrName2AttrDef(
          java.lang.String key,
          org.oneflow.core.framework.UserOpAttr.AttrDef value) {
        if (key == null) { throw new java.lang.NullPointerException(); }
        if (value == null) { throw new java.lang.NullPointerException(); }
        getMutableAttrName2AttrDef().put(key, value);
        return this;
      }
      /**
       * <code>map&lt;string, .oneflow.AttrDef&gt; attr_name2attr_def = 1;</code>
       */

      public Builder putAllAttrName2AttrDef(
          java.util.Map<java.lang.String, org.oneflow.core.framework.UserOpAttr.AttrDef> values) {
        getMutableAttrName2AttrDef().putAll(values);
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


      // @@protoc_insertion_point(builder_scope:oneflow.ConfigDef)
    }

    // @@protoc_insertion_point(class_scope:oneflow.ConfigDef)
    private static final org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef DEFAULT_INSTANCE;
    static {
      DEFAULT_INSTANCE = new org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef();
    }

    public static org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef getDefaultInstance() {
      return DEFAULT_INSTANCE;
    }

    @java.lang.Deprecated public static final com.google.protobuf.Parser<ConfigDef>
        PARSER = new com.google.protobuf.AbstractParser<ConfigDef>() {
      public ConfigDef parsePartialFrom(
          com.google.protobuf.CodedInputStream input,
          com.google.protobuf.ExtensionRegistryLite extensionRegistry)
          throws com.google.protobuf.InvalidProtocolBufferException {
          return new ConfigDef(input, extensionRegistry);
      }
    };

    public static com.google.protobuf.Parser<ConfigDef> parser() {
      return PARSER;
    }

    @java.lang.Override
    public com.google.protobuf.Parser<ConfigDef> getParserForType() {
      return PARSER;
    }

    public org.oneflow.core.framework.ConfigDefOuterClass.ConfigDef getDefaultInstanceForType() {
      return DEFAULT_INSTANCE;
    }

  }

  private static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_ConfigDef_descriptor;
  private static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_ConfigDef_fieldAccessorTable;
  private static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_ConfigDef_AttrName2attrDefEntry_descriptor;
  private static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_ConfigDef_AttrName2attrDefEntry_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\'oneflow/core/framework/config_def.prot" +
      "o\022\007oneflow\032)oneflow/core/framework/user_" +
      "op_attr.proto\"\234\001\n\tConfigDef\022D\n\022attr_name" +
      "2attr_def\030\001 \003(\0132(.oneflow.ConfigDef.Attr" +
      "Name2attrDefEntry\032I\n\025AttrName2attrDefEnt" +
      "ry\022\013\n\003key\030\001 \001(\t\022\037\n\005value\030\002 \001(\0132\020.oneflow" +
      ".AttrDef:\0028\001*v\n\rConfigDefType\022\025\n\021kEnvCon" +
      "figDefType\020\001\022\031\n\025kSessionConfigDefType\020\002\022" +
      "\032\n\026kFunctionConfigDefType\020\003\022\027\n\023kScopeCon" +
      "figDefType\020\004B\034\n\032org.oneflow.core.framewo",
      "rk"
    };
    com.google.protobuf.Descriptors.FileDescriptor.InternalDescriptorAssigner assigner =
        new com.google.protobuf.Descriptors.FileDescriptor.    InternalDescriptorAssigner() {
          public com.google.protobuf.ExtensionRegistry assignDescriptors(
              com.google.protobuf.Descriptors.FileDescriptor root) {
            descriptor = root;
            return null;
          }
        };
    com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
          org.oneflow.core.framework.UserOpAttr.getDescriptor(),
        }, assigner);
    internal_static_oneflow_ConfigDef_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_oneflow_ConfigDef_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_ConfigDef_descriptor,
        new java.lang.String[] { "AttrName2AttrDef", });
    internal_static_oneflow_ConfigDef_AttrName2attrDefEntry_descriptor =
      internal_static_oneflow_ConfigDef_descriptor.getNestedTypes().get(0);
    internal_static_oneflow_ConfigDef_AttrName2attrDefEntry_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_ConfigDef_AttrName2attrDefEntry_descriptor,
        new java.lang.String[] { "Key", "Value", });
    org.oneflow.core.framework.UserOpAttr.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}
