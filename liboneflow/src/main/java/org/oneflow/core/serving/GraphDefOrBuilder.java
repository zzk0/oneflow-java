// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/serving/saved_model.proto

package org.oneflow.core.serving;

public interface GraphDefOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.GraphDef)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>repeated .oneflow.OperatorConf op_list = 1;</code>
   */
  java.util.List<org.oneflow.core.operator.OperatorConf> 
      getOpListList();
  /**
   * <code>repeated .oneflow.OperatorConf op_list = 1;</code>
   */
  org.oneflow.core.operator.OperatorConf getOpList(int index);
  /**
   * <code>repeated .oneflow.OperatorConf op_list = 1;</code>
   */
  int getOpListCount();
  /**
   * <code>repeated .oneflow.OperatorConf op_list = 1;</code>
   */
  java.util.List<? extends org.oneflow.core.operator.OperatorConfOrBuilder> 
      getOpListOrBuilderList();
  /**
   * <code>repeated .oneflow.OperatorConf op_list = 1;</code>
   */
  org.oneflow.core.operator.OperatorConfOrBuilder getOpListOrBuilder(
      int index);

  /**
   * <code>map&lt;string, .oneflow.JobSignatureDef&gt; signatures = 2;</code>
   */
  int getSignaturesCount();
  /**
   * <code>map&lt;string, .oneflow.JobSignatureDef&gt; signatures = 2;</code>
   */
  boolean containsSignatures(
      java.lang.String key);
  /**
   * Use {@link #getSignaturesMap()} instead.
   */
  @java.lang.Deprecated
  java.util.Map<java.lang.String, org.oneflow.core.job.JobSignatureDef>
  getSignatures();
  /**
   * <code>map&lt;string, .oneflow.JobSignatureDef&gt; signatures = 2;</code>
   */
  java.util.Map<java.lang.String, org.oneflow.core.job.JobSignatureDef>
  getSignaturesMap();
  /**
   * <code>map&lt;string, .oneflow.JobSignatureDef&gt; signatures = 2;</code>
   */

  org.oneflow.core.job.JobSignatureDef getSignaturesOrDefault(
      java.lang.String key,
      org.oneflow.core.job.JobSignatureDef defaultValue);
  /**
   * <code>map&lt;string, .oneflow.JobSignatureDef&gt; signatures = 2;</code>
   */

  org.oneflow.core.job.JobSignatureDef getSignaturesOrThrow(
      java.lang.String key);

  /**
   * <code>optional string default_signature_name = 3;</code>
   */
  boolean hasDefaultSignatureName();
  /**
   * <code>optional string default_signature_name = 3;</code>
   */
  java.lang.String getDefaultSignatureName();
  /**
   * <code>optional string default_signature_name = 3;</code>
   */
  com.google.protobuf.ByteString
      getDefaultSignatureNameBytes();
}
