// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/summary/tensor.proto

package org.oneflow.core.summary;

public interface TensorProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.summary.TensorProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>required .oneflow.summary.TensorDataType dtype = 1;</code>
   */
  boolean hasDtype();
  /**
   * <code>required .oneflow.summary.TensorDataType dtype = 1;</code>
   */
  org.oneflow.core.summary.TensorDataType getDtype();

  /**
   * <code>required .oneflow.summary.TensorShapeProto tensor_shape = 2;</code>
   */
  boolean hasTensorShape();
  /**
   * <code>required .oneflow.summary.TensorShapeProto tensor_shape = 2;</code>
   */
  org.oneflow.core.summary.TensorShapeProto getTensorShape();
  /**
   * <code>required .oneflow.summary.TensorShapeProto tensor_shape = 2;</code>
   */
  org.oneflow.core.summary.TensorShapeProtoOrBuilder getTensorShapeOrBuilder();

  /**
   * <code>optional int32 version_number = 3;</code>
   */
  boolean hasVersionNumber();
  /**
   * <code>optional int32 version_number = 3;</code>
   */
  int getVersionNumber();

  /**
   * <code>optional bytes tensor_content = 4;</code>
   */
  boolean hasTensorContent();
  /**
   * <code>optional bytes tensor_content = 4;</code>
   */
  com.google.protobuf.ByteString getTensorContent();

  /**
   * <code>repeated float float_val = 5 [packed = true];</code>
   */
  java.util.List<java.lang.Float> getFloatValList();
  /**
   * <code>repeated float float_val = 5 [packed = true];</code>
   */
  int getFloatValCount();
  /**
   * <code>repeated float float_val = 5 [packed = true];</code>
   */
  float getFloatVal(int index);

  /**
   * <code>repeated double double_val = 6 [packed = true];</code>
   */
  java.util.List<java.lang.Double> getDoubleValList();
  /**
   * <code>repeated double double_val = 6 [packed = true];</code>
   */
  int getDoubleValCount();
  /**
   * <code>repeated double double_val = 6 [packed = true];</code>
   */
  double getDoubleVal(int index);

  /**
   * <code>repeated int32 int_val = 7 [packed = true];</code>
   */
  java.util.List<java.lang.Integer> getIntValList();
  /**
   * <code>repeated int32 int_val = 7 [packed = true];</code>
   */
  int getIntValCount();
  /**
   * <code>repeated int32 int_val = 7 [packed = true];</code>
   */
  int getIntVal(int index);

  /**
   * <code>repeated bytes string_val = 8;</code>
   */
  java.util.List<com.google.protobuf.ByteString> getStringValList();
  /**
   * <code>repeated bytes string_val = 8;</code>
   */
  int getStringValCount();
  /**
   * <code>repeated bytes string_val = 8;</code>
   */
  com.google.protobuf.ByteString getStringVal(int index);

  /**
   * <code>repeated int64 int64_val = 9 [packed = true];</code>
   */
  java.util.List<java.lang.Long> getInt64ValList();
  /**
   * <code>repeated int64 int64_val = 9 [packed = true];</code>
   */
  int getInt64ValCount();
  /**
   * <code>repeated int64 int64_val = 9 [packed = true];</code>
   */
  long getInt64Val(int index);

  /**
   * <code>repeated bool bool_val = 10 [packed = true];</code>
   */
  java.util.List<java.lang.Boolean> getBoolValList();
  /**
   * <code>repeated bool bool_val = 10 [packed = true];</code>
   */
  int getBoolValCount();
  /**
   * <code>repeated bool bool_val = 10 [packed = true];</code>
   */
  boolean getBoolVal(int index);

  /**
   * <code>repeated uint32 uint32_val = 11 [packed = true];</code>
   */
  java.util.List<java.lang.Integer> getUint32ValList();
  /**
   * <code>repeated uint32 uint32_val = 11 [packed = true];</code>
   */
  int getUint32ValCount();
  /**
   * <code>repeated uint32 uint32_val = 11 [packed = true];</code>
   */
  int getUint32Val(int index);

  /**
   * <code>repeated uint64 uint64_val = 12 [packed = true];</code>
   */
  java.util.List<java.lang.Long> getUint64ValList();
  /**
   * <code>repeated uint64 uint64_val = 12 [packed = true];</code>
   */
  int getUint64ValCount();
  /**
   * <code>repeated uint64 uint64_val = 12 [packed = true];</code>
   */
  long getUint64Val(int index);

  /**
   * <code>repeated int32 half_val = 13 [packed = true];</code>
   */
  java.util.List<java.lang.Integer> getHalfValList();
  /**
   * <code>repeated int32 half_val = 13 [packed = true];</code>
   */
  int getHalfValCount();
  /**
   * <code>repeated int32 half_val = 13 [packed = true];</code>
   */
  int getHalfVal(int index);
}
