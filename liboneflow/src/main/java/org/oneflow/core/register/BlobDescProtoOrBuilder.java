// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/register/blob_desc.proto

package org.oneflow.core.register;

public interface BlobDescProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.BlobDescProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>required .oneflow.ShapeProto shape = 1;</code>
   */
  boolean hasShape();
  /**
   * <code>required .oneflow.ShapeProto shape = 1;</code>
   */
  oneflow.Shape.ShapeProto getShape();
  /**
   * <code>required .oneflow.ShapeProto shape = 1;</code>
   */
  oneflow.Shape.ShapeProtoOrBuilder getShapeOrBuilder();

  /**
   * <code>required .oneflow.DataType data_type = 2;</code>
   */
  boolean hasDataType();
  /**
   * <code>required .oneflow.DataType data_type = 2;</code>
   */
  oneflow.DataTypeOuterClass.DataType getDataType();

  /**
   * <code>required bool is_dynamic = 3;</code>
   */
  boolean hasIsDynamic();
  /**
   * <code>required bool is_dynamic = 3;</code>
   */
  boolean getIsDynamic();
}