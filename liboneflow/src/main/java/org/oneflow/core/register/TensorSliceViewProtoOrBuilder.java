// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/register/tensor_slice_view.proto

package org.oneflow.core.register;

public interface TensorSliceViewProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.TensorSliceViewProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>repeated .oneflow.RangeProto dim = 1;</code>
   */
  java.util.List<oneflow.Range.RangeProto> 
      getDimList();
  /**
   * <code>repeated .oneflow.RangeProto dim = 1;</code>
   */
  oneflow.Range.RangeProto getDim(int index);
  /**
   * <code>repeated .oneflow.RangeProto dim = 1;</code>
   */
  int getDimCount();
  /**
   * <code>repeated .oneflow.RangeProto dim = 1;</code>
   */
  java.util.List<? extends oneflow.Range.RangeProtoOrBuilder> 
      getDimOrBuilderList();
  /**
   * <code>repeated .oneflow.RangeProto dim = 1;</code>
   */
  oneflow.Range.RangeProtoOrBuilder getDimOrBuilder(
      int index);
}