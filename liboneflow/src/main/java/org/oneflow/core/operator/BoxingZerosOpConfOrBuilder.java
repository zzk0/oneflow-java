// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/operator/op_conf.proto

package org.oneflow.core.operator;

public interface BoxingZerosOpConfOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.BoxingZerosOpConf)
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
   * <code>required .oneflow.ShapeProto shape = 2;</code>
   */
  boolean hasShape();
  /**
   * <code>required .oneflow.ShapeProto shape = 2;</code>
   */
  org.oneflow.core.common.ShapeProto getShape();
  /**
   * <code>required .oneflow.ShapeProto shape = 2;</code>
   */
  org.oneflow.core.common.ShapeProtoOrBuilder getShapeOrBuilder();

  /**
   * <code>required .oneflow.DataType data_type = 3;</code>
   */
  boolean hasDataType();
  /**
   * <code>required .oneflow.DataType data_type = 3;</code>
   */
  org.oneflow.core.common.DataType getDataType();
}
