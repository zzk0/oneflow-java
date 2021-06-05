// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/summary/projector.proto

package org.oneflow.core.summary;

public interface ProjectorOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.summary.Projector)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>required string tag = 1;</code>
   */
  boolean hasTag();
  /**
   * <code>required string tag = 1;</code>
   */
  java.lang.String getTag();
  /**
   * <code>required string tag = 1;</code>
   */
  com.google.protobuf.ByteString
      getTagBytes();

  /**
   * <code>optional int64 step = 2;</code>
   */
  boolean hasStep();
  /**
   * <code>optional int64 step = 2;</code>
   */
  long getStep();

  /**
   * <code>required double WALL_TIME = 3;</code>
   */
  boolean hasWALLTIME();
  /**
   * <code>required double WALL_TIME = 3;</code>
   */
  double getWALLTIME();

  /**
   * <code>required .oneflow.summary.Tensor value = 4;</code>
   */
  boolean hasValue();
  /**
   * <code>required .oneflow.summary.Tensor value = 4;</code>
   */
  org.oneflow.core.summary.Tensor getValue();
  /**
   * <code>required .oneflow.summary.Tensor value = 4;</code>
   */
  org.oneflow.core.summary.TensorOrBuilder getValueOrBuilder();

  /**
   * <code>optional .oneflow.summary.Tensor label = 5;</code>
   */
  boolean hasLabel();
  /**
   * <code>optional .oneflow.summary.Tensor label = 5;</code>
   */
  org.oneflow.core.summary.Tensor getLabel();
  /**
   * <code>optional .oneflow.summary.Tensor label = 5;</code>
   */
  org.oneflow.core.summary.TensorOrBuilder getLabelOrBuilder();
}