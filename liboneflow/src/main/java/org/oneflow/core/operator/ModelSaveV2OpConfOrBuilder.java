// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/operator/op_conf.proto

package org.oneflow.core.operator;

public interface ModelSaveV2OpConfOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.ModelSaveV2OpConf)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>required string path = 1;</code>
   */
  boolean hasPath();
  /**
   * <code>required string path = 1;</code>
   */
  java.lang.String getPath();
  /**
   * <code>required string path = 1;</code>
   */
  com.google.protobuf.ByteString
      getPathBytes();

  /**
   * <code>repeated string in = 2;</code>
   */
  java.util.List<java.lang.String>
      getInList();
  /**
   * <code>repeated string in = 2;</code>
   */
  int getInCount();
  /**
   * <code>repeated string in = 2;</code>
   */
  java.lang.String getIn(int index);
  /**
   * <code>repeated string in = 2;</code>
   */
  com.google.protobuf.ByteString
      getInBytes(int index);

  /**
   * <code>repeated string variable_op_name = 3;</code>
   */
  java.util.List<java.lang.String>
      getVariableOpNameList();
  /**
   * <code>repeated string variable_op_name = 3;</code>
   */
  int getVariableOpNameCount();
  /**
   * <code>repeated string variable_op_name = 3;</code>
   */
  java.lang.String getVariableOpName(int index);
  /**
   * <code>repeated string variable_op_name = 3;</code>
   */
  com.google.protobuf.ByteString
      getVariableOpNameBytes(int index);

  /**
   * <code>repeated .oneflow.VariableOpConf original_variable_conf = 4;</code>
   */
  java.util.List<org.oneflow.core.operator.VariableOpConf> 
      getOriginalVariableConfList();
  /**
   * <code>repeated .oneflow.VariableOpConf original_variable_conf = 4;</code>
   */
  org.oneflow.core.operator.VariableOpConf getOriginalVariableConf(int index);
  /**
   * <code>repeated .oneflow.VariableOpConf original_variable_conf = 4;</code>
   */
  int getOriginalVariableConfCount();
  /**
   * <code>repeated .oneflow.VariableOpConf original_variable_conf = 4;</code>
   */
  java.util.List<? extends org.oneflow.core.operator.VariableOpConfOrBuilder> 
      getOriginalVariableConfOrBuilderList();
  /**
   * <code>repeated .oneflow.VariableOpConf original_variable_conf = 4;</code>
   */
  org.oneflow.core.operator.VariableOpConfOrBuilder getOriginalVariableConfOrBuilder(
      int index);
}
