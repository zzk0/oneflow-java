// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/operator/op_conf.proto

package org.oneflow.core.operator;

public interface ModelInitV2OpConfOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.ModelInitV2OpConf)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>repeated string ref = 1;</code>
   */
  java.util.List<java.lang.String>
      getRefList();
  /**
   * <code>repeated string ref = 1;</code>
   */
  int getRefCount();
  /**
   * <code>repeated string ref = 1;</code>
   */
  java.lang.String getRef(int index);
  /**
   * <code>repeated string ref = 1;</code>
   */
  com.google.protobuf.ByteString
      getRefBytes(int index);

  /**
   * <code>repeated string variable_op_name = 2;</code>
   */
  java.util.List<java.lang.String>
      getVariableOpNameList();
  /**
   * <code>repeated string variable_op_name = 2;</code>
   */
  int getVariableOpNameCount();
  /**
   * <code>repeated string variable_op_name = 2;</code>
   */
  java.lang.String getVariableOpName(int index);
  /**
   * <code>repeated string variable_op_name = 2;</code>
   */
  com.google.protobuf.ByteString
      getVariableOpNameBytes(int index);

  /**
   * <code>repeated .oneflow.VariableOpConf original_variable_conf = 3;</code>
   */
  java.util.List<org.oneflow.core.operator.VariableOpConf> 
      getOriginalVariableConfList();
  /**
   * <code>repeated .oneflow.VariableOpConf original_variable_conf = 3;</code>
   */
  org.oneflow.core.operator.VariableOpConf getOriginalVariableConf(int index);
  /**
   * <code>repeated .oneflow.VariableOpConf original_variable_conf = 3;</code>
   */
  int getOriginalVariableConfCount();
  /**
   * <code>repeated .oneflow.VariableOpConf original_variable_conf = 3;</code>
   */
  java.util.List<? extends org.oneflow.core.operator.VariableOpConfOrBuilder> 
      getOriginalVariableConfOrBuilderList();
  /**
   * <code>repeated .oneflow.VariableOpConf original_variable_conf = 3;</code>
   */
  org.oneflow.core.operator.VariableOpConfOrBuilder getOriginalVariableConfOrBuilder(
      int index);
}