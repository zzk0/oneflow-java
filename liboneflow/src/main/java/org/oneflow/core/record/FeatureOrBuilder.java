// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/record/record.proto

package org.oneflow.core.record;

public interface FeatureOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.Feature)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>optional .oneflow.BytesList bytes_list = 1;</code>
   */
  boolean hasBytesList();
  /**
   * <code>optional .oneflow.BytesList bytes_list = 1;</code>
   */
  org.oneflow.core.record.BytesList getBytesList();
  /**
   * <code>optional .oneflow.BytesList bytes_list = 1;</code>
   */
  org.oneflow.core.record.BytesListOrBuilder getBytesListOrBuilder();

  /**
   * <code>optional .oneflow.FloatList float_list = 2;</code>
   */
  boolean hasFloatList();
  /**
   * <code>optional .oneflow.FloatList float_list = 2;</code>
   */
  org.oneflow.core.record.FloatList getFloatList();
  /**
   * <code>optional .oneflow.FloatList float_list = 2;</code>
   */
  org.oneflow.core.record.FloatListOrBuilder getFloatListOrBuilder();

  /**
   * <code>optional .oneflow.DoubleList double_list = 3;</code>
   */
  boolean hasDoubleList();
  /**
   * <code>optional .oneflow.DoubleList double_list = 3;</code>
   */
  org.oneflow.core.record.DoubleList getDoubleList();
  /**
   * <code>optional .oneflow.DoubleList double_list = 3;</code>
   */
  org.oneflow.core.record.DoubleListOrBuilder getDoubleListOrBuilder();

  /**
   * <code>optional .oneflow.Int32List int32_list = 4;</code>
   */
  boolean hasInt32List();
  /**
   * <code>optional .oneflow.Int32List int32_list = 4;</code>
   */
  org.oneflow.core.record.Int32List getInt32List();
  /**
   * <code>optional .oneflow.Int32List int32_list = 4;</code>
   */
  org.oneflow.core.record.Int32ListOrBuilder getInt32ListOrBuilder();

  /**
   * <code>optional .oneflow.Int64List int64_list = 5;</code>
   */
  boolean hasInt64List();
  /**
   * <code>optional .oneflow.Int64List int64_list = 5;</code>
   */
  org.oneflow.core.record.Int64List getInt64List();
  /**
   * <code>optional .oneflow.Int64List int64_list = 5;</code>
   */
  org.oneflow.core.record.Int64ListOrBuilder getInt64ListOrBuilder();

  public org.oneflow.core.record.Feature.KindCase getKindCase();
}