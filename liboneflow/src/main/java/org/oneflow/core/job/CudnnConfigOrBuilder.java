// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/resource.proto

package org.oneflow.core.job;

public interface CudnnConfigOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.CudnnConfig)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>optional bool enable_cudnn = 1 [default = true];</code>
   */
  boolean hasEnableCudnn();
  /**
   * <code>optional bool enable_cudnn = 1 [default = true];</code>
   */
  boolean getEnableCudnn();

  /**
   * <pre>
   * 1GByte
   * </pre>
   *
   * <code>optional int64 cudnn_buf_limit_mbyte = 2 [default = 1024];</code>
   */
  boolean hasCudnnBufLimitMbyte();
  /**
   * <pre>
   * 1GByte
   * </pre>
   *
   * <code>optional int64 cudnn_buf_limit_mbyte = 2 [default = 1024];</code>
   */
  long getCudnnBufLimitMbyte();

  /**
   * <code>optional int32 cudnn_conv_force_fwd_algo = 3;</code>
   */
  boolean hasCudnnConvForceFwdAlgo();
  /**
   * <code>optional int32 cudnn_conv_force_fwd_algo = 3;</code>
   */
  int getCudnnConvForceFwdAlgo();

  /**
   * <code>optional int32 cudnn_conv_force_bwd_data_algo = 4;</code>
   */
  boolean hasCudnnConvForceBwdDataAlgo();
  /**
   * <code>optional int32 cudnn_conv_force_bwd_data_algo = 4;</code>
   */
  int getCudnnConvForceBwdDataAlgo();

  /**
   * <code>optional int32 cudnn_conv_force_bwd_filter_algo = 5;</code>
   */
  boolean hasCudnnConvForceBwdFilterAlgo();
  /**
   * <code>optional int32 cudnn_conv_force_bwd_filter_algo = 5;</code>
   */
  int getCudnnConvForceBwdFilterAlgo();

  /**
   * <code>optional bool cudnn_conv_heuristic_search_algo = 6 [default = true];</code>
   */
  boolean hasCudnnConvHeuristicSearchAlgo();
  /**
   * <code>optional bool cudnn_conv_heuristic_search_algo = 6 [default = true];</code>
   */
  boolean getCudnnConvHeuristicSearchAlgo();

  /**
   * <code>optional bool cudnn_conv_use_deterministic_algo_only = 7 [default = false];</code>
   */
  boolean hasCudnnConvUseDeterministicAlgoOnly();
  /**
   * <code>optional bool cudnn_conv_use_deterministic_algo_only = 7 [default = false];</code>
   */
  boolean getCudnnConvUseDeterministicAlgoOnly();

  /**
   * <code>optional bool enable_cudnn_fused_normalization_add_relu = 8;</code>
   */
  boolean hasEnableCudnnFusedNormalizationAddRelu();
  /**
   * <code>optional bool enable_cudnn_fused_normalization_add_relu = 8;</code>
   */
  boolean getEnableCudnnFusedNormalizationAddRelu();

  /**
   * <code>optional bool cudnn_conv_enable_pseudo_half = 9 [default = true];</code>
   */
  boolean hasCudnnConvEnablePseudoHalf();
  /**
   * <code>optional bool cudnn_conv_enable_pseudo_half = 9 [default = true];</code>
   */
  boolean getCudnnConvEnablePseudoHalf();
}
