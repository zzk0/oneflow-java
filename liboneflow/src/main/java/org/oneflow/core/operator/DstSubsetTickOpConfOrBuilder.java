// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/operator/op_conf.proto

package org.oneflow.core.operator;

public interface DstSubsetTickOpConfOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.DstSubsetTickOpConf)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>repeated string in = 1;</code>
   */
  java.util.List<java.lang.String>
      getInList();
  /**
   * <code>repeated string in = 1;</code>
   */
  int getInCount();
  /**
   * <code>repeated string in = 1;</code>
   */
  java.lang.String getIn(int index);
  /**
   * <code>repeated string in = 1;</code>
   */
  com.google.protobuf.ByteString
      getInBytes(int index);

  /**
   * <code>required string out = 2;</code>
   */
  boolean hasOut();
  /**
   * <code>required string out = 2;</code>
   */
  java.lang.String getOut();
  /**
   * <code>required string out = 2;</code>
   */
  com.google.protobuf.ByteString
      getOutBytes();
}
