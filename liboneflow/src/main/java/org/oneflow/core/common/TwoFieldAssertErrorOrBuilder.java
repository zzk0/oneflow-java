// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/common/error.proto

package org.oneflow.core.common;

public interface TwoFieldAssertErrorOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.TwoFieldAssertError)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>required .oneflow.OpcodeType compare_type = 1;</code>
   */
  boolean hasCompareType();
  /**
   * <code>required .oneflow.OpcodeType compare_type = 1;</code>
   */
  org.oneflow.core.common.OpcodeType getCompareType();

  /**
   * <code>required .oneflow.FieldValue left = 2;</code>
   */
  boolean hasLeft();
  /**
   * <code>required .oneflow.FieldValue left = 2;</code>
   */
  org.oneflow.core.common.FieldValue getLeft();
  /**
   * <code>required .oneflow.FieldValue left = 2;</code>
   */
  org.oneflow.core.common.FieldValueOrBuilder getLeftOrBuilder();

  /**
   * <code>required .oneflow.FieldValue right = 3;</code>
   */
  boolean hasRight();
  /**
   * <code>required .oneflow.FieldValue right = 3;</code>
   */
  org.oneflow.core.common.FieldValue getRight();
  /**
   * <code>required .oneflow.FieldValue right = 3;</code>
   */
  org.oneflow.core.common.FieldValueOrBuilder getRightOrBuilder();
}
