// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/operator/op_conf.proto

package org.oneflow.core.operator;

public interface ShapeElemCntOpConfOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.ShapeElemCntOpConf)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>required string x = 1;</code>
   */
  boolean hasX();
  /**
   * <code>required string x = 1;</code>
   */
  java.lang.String getX();
  /**
   * <code>required string x = 1;</code>
   */
  com.google.protobuf.ByteString
      getXBytes();

  /**
   * <code>required string y = 2;</code>
   */
  boolean hasY();
  /**
   * <code>required string y = 2;</code>
   */
  java.lang.String getY();
  /**
   * <code>required string y = 2;</code>
   */
  com.google.protobuf.ByteString
      getYBytes();

  /**
   * <code>optional .oneflow.DataType data_type = 3 [default = kInt32];</code>
   */
  boolean hasDataType();
  /**
   * <code>optional .oneflow.DataType data_type = 3 [default = kInt32];</code>
   */
  org.oneflow.core.common.DataType getDataType();

  /**
   * <code>optional .oneflow.ShapeElemCntAxisConf exclude_axis_conf = 4;</code>
   */
  boolean hasExcludeAxisConf();
  /**
   * <code>optional .oneflow.ShapeElemCntAxisConf exclude_axis_conf = 4;</code>
   */
  org.oneflow.core.operator.ShapeElemCntAxisConf getExcludeAxisConf();
  /**
   * <code>optional .oneflow.ShapeElemCntAxisConf exclude_axis_conf = 4;</code>
   */
  org.oneflow.core.operator.ShapeElemCntAxisConfOrBuilder getExcludeAxisConfOrBuilder();

  /**
   * <code>optional .oneflow.ShapeElemCntAxisConf include_axis_conf = 5;</code>
   */
  boolean hasIncludeAxisConf();
  /**
   * <code>optional .oneflow.ShapeElemCntAxisConf include_axis_conf = 5;</code>
   */
  org.oneflow.core.operator.ShapeElemCntAxisConf getIncludeAxisConf();
  /**
   * <code>optional .oneflow.ShapeElemCntAxisConf include_axis_conf = 5;</code>
   */
  org.oneflow.core.operator.ShapeElemCntAxisConfOrBuilder getIncludeAxisConfOrBuilder();

  /**
   * <code>optional .oneflow.ShapeElemCntRangeAxisConf range_axis_conf = 6;</code>
   */
  boolean hasRangeAxisConf();
  /**
   * <code>optional .oneflow.ShapeElemCntRangeAxisConf range_axis_conf = 6;</code>
   */
  org.oneflow.core.operator.ShapeElemCntRangeAxisConf getRangeAxisConf();
  /**
   * <code>optional .oneflow.ShapeElemCntRangeAxisConf range_axis_conf = 6;</code>
   */
  org.oneflow.core.operator.ShapeElemCntRangeAxisConfOrBuilder getRangeAxisConfOrBuilder();

  public org.oneflow.core.operator.ShapeElemCntOpConf.AxisConfCase getAxisConfCase();
}
