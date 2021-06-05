// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/eager/eager_symbol.proto

package org.oneflow.core.eager;

public interface EagerSymbolOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.vm.EagerSymbol)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>required int64 symbol_id = 1;</code>
   */
  boolean hasSymbolId();
  /**
   * <code>required int64 symbol_id = 1;</code>
   */
  long getSymbolId();

  /**
   * <code>optional string string_symbol = 2;</code>
   */
  boolean hasStringSymbol();
  /**
   * <code>optional string string_symbol = 2;</code>
   */
  java.lang.String getStringSymbol();
  /**
   * <code>optional string string_symbol = 2;</code>
   */
  com.google.protobuf.ByteString
      getStringSymbolBytes();

  /**
   * <code>optional .oneflow.ScopeProto scope_symbol = 3;</code>
   */
  boolean hasScopeSymbol();
  /**
   * <code>optional .oneflow.ScopeProto scope_symbol = 3;</code>
   */
  org.oneflow.core.job.ScopeProto getScopeSymbol();
  /**
   * <code>optional .oneflow.ScopeProto scope_symbol = 3;</code>
   */
  org.oneflow.core.job.ScopeProtoOrBuilder getScopeSymbolOrBuilder();

  /**
   * <code>optional .oneflow.JobConfigProto job_conf_symbol = 4;</code>
   */
  boolean hasJobConfSymbol();
  /**
   * <code>optional .oneflow.JobConfigProto job_conf_symbol = 4;</code>
   */
  org.oneflow.core.job.JobConfigProto getJobConfSymbol();
  /**
   * <code>optional .oneflow.JobConfigProto job_conf_symbol = 4;</code>
   */
  org.oneflow.core.job.JobConfigProtoOrBuilder getJobConfSymbolOrBuilder();

  /**
   * <code>optional .oneflow.ParallelConf parallel_conf_symbol = 5;</code>
   */
  boolean hasParallelConfSymbol();
  /**
   * <code>optional .oneflow.ParallelConf parallel_conf_symbol = 5;</code>
   */
  org.oneflow.core.job.ParallelConf getParallelConfSymbol();
  /**
   * <code>optional .oneflow.ParallelConf parallel_conf_symbol = 5;</code>
   */
  org.oneflow.core.job.ParallelConfOrBuilder getParallelConfSymbolOrBuilder();

  /**
   * <code>optional .oneflow.OperatorConf op_conf_symbol = 6;</code>
   */
  boolean hasOpConfSymbol();
  /**
   * <code>optional .oneflow.OperatorConf op_conf_symbol = 6;</code>
   */
  org.oneflow.core.operator.OperatorConf getOpConfSymbol();
  /**
   * <code>optional .oneflow.OperatorConf op_conf_symbol = 6;</code>
   */
  org.oneflow.core.operator.OperatorConfOrBuilder getOpConfSymbolOrBuilder();

  /**
   * <code>optional .oneflow.OpNodeSignature op_node_signature_symbol = 7;</code>
   */
  boolean hasOpNodeSignatureSymbol();
  /**
   * <code>optional .oneflow.OpNodeSignature op_node_signature_symbol = 7;</code>
   */
  org.oneflow.core.operator.OpNodeSignature getOpNodeSignatureSymbol();
  /**
   * <code>optional .oneflow.OpNodeSignature op_node_signature_symbol = 7;</code>
   */
  org.oneflow.core.operator.OpNodeSignatureOrBuilder getOpNodeSignatureSymbolOrBuilder();

  public org.oneflow.core.eager.EagerSymbol.EagerSymbolTypeCase getEagerSymbolTypeCase();
}
