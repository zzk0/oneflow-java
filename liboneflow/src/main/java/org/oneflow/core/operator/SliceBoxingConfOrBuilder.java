// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/operator/op_conf.proto

package org.oneflow.core.operator;

public interface SliceBoxingConfOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.SliceBoxingConf)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
   */
  boolean hasLbi();
  /**
   * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
   */
  org.oneflow.core.register.LogicalBlobId getLbi();
  /**
   * <code>required .oneflow.LogicalBlobId lbi = 1;</code>
   */
  org.oneflow.core.register.LogicalBlobIdOrBuilder getLbiOrBuilder();

  /**
   * <code>repeated .oneflow.TensorSliceViewProto in_slice = 2;</code>
   */
  java.util.List<org.oneflow.core.register.TensorSliceViewProto> 
      getInSliceList();
  /**
   * <code>repeated .oneflow.TensorSliceViewProto in_slice = 2;</code>
   */
  org.oneflow.core.register.TensorSliceViewProto getInSlice(int index);
  /**
   * <code>repeated .oneflow.TensorSliceViewProto in_slice = 2;</code>
   */
  int getInSliceCount();
  /**
   * <code>repeated .oneflow.TensorSliceViewProto in_slice = 2;</code>
   */
  java.util.List<? extends org.oneflow.core.register.TensorSliceViewProtoOrBuilder> 
      getInSliceOrBuilderList();
  /**
   * <code>repeated .oneflow.TensorSliceViewProto in_slice = 2;</code>
   */
  org.oneflow.core.register.TensorSliceViewProtoOrBuilder getInSliceOrBuilder(
      int index);

  /**
   * <code>required .oneflow.TensorSliceViewProto out_slice = 3;</code>
   */
  boolean hasOutSlice();
  /**
   * <code>required .oneflow.TensorSliceViewProto out_slice = 3;</code>
   */
  org.oneflow.core.register.TensorSliceViewProto getOutSlice();
  /**
   * <code>required .oneflow.TensorSliceViewProto out_slice = 3;</code>
   */
  org.oneflow.core.register.TensorSliceViewProtoOrBuilder getOutSliceOrBuilder();

  /**
   * <code>optional .oneflow.ShapeProto out_shape = 4;</code>
   */
  boolean hasOutShape();
  /**
   * <code>optional .oneflow.ShapeProto out_shape = 4;</code>
   */
  org.oneflow.core.common.ShapeProto getOutShape();
  /**
   * <code>optional .oneflow.ShapeProto out_shape = 4;</code>
   */
  org.oneflow.core.common.ShapeProtoOrBuilder getOutShapeOrBuilder();
}