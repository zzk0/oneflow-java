// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/plan.proto

package org.oneflow.core.job;

public interface OpAttributeRefTableOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.OpAttributeRefTable)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>map&lt;string, .oneflow.OpAttribute&gt; op_name2op_attribute = 1;</code>
   */
  int getOpName2OpAttributeCount();
  /**
   * <code>map&lt;string, .oneflow.OpAttribute&gt; op_name2op_attribute = 1;</code>
   */
  boolean containsOpName2OpAttribute(
      java.lang.String key);
  /**
   * Use {@link #getOpName2OpAttributeMap()} instead.
   */
  @java.lang.Deprecated
  java.util.Map<java.lang.String, org.oneflow.core.operator.OpAttribute>
  getOpName2OpAttribute();
  /**
   * <code>map&lt;string, .oneflow.OpAttribute&gt; op_name2op_attribute = 1;</code>
   */
  java.util.Map<java.lang.String, org.oneflow.core.operator.OpAttribute>
  getOpName2OpAttributeMap();
  /**
   * <code>map&lt;string, .oneflow.OpAttribute&gt; op_name2op_attribute = 1;</code>
   */

  org.oneflow.core.operator.OpAttribute getOpName2OpAttributeOrDefault(
      java.lang.String key,
      org.oneflow.core.operator.OpAttribute defaultValue);
  /**
   * <code>map&lt;string, .oneflow.OpAttribute&gt; op_name2op_attribute = 1;</code>
   */

  org.oneflow.core.operator.OpAttribute getOpName2OpAttributeOrThrow(
      java.lang.String key);
}
