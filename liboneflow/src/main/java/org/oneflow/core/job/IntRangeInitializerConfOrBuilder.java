// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/initializer_conf.proto

package org.oneflow.core.job;

public interface IntRangeInitializerConfOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.IntRangeInitializerConf)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>optional int64 start = 1 [default = 0];</code>
   */
  boolean hasStart();
  /**
   * <code>optional int64 start = 1 [default = 0];</code>
   */
  long getStart();

  /**
   * <code>optional int64 stride = 2 [default = 1];</code>
   */
  boolean hasStride();
  /**
   * <code>optional int64 stride = 2 [default = 1];</code>
   */
  long getStride();

  /**
   * <code>optional int64 axis = 3 [default = -1];</code>
   */
  boolean hasAxis();
  /**
   * <code>optional int64 axis = 3 [default = -1];</code>
   */
  long getAxis();
}