// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/sbp_parallel.proto

package org.oneflow.core.job;

public interface ParallelDistributionOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.ParallelDistribution)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>repeated .oneflow.SbpParallel sbp_parallel = 1;</code>
   */
  java.util.List<org.oneflow.core.job.SbpParallel> 
      getSbpParallelList();
  /**
   * <code>repeated .oneflow.SbpParallel sbp_parallel = 1;</code>
   */
  org.oneflow.core.job.SbpParallel getSbpParallel(int index);
  /**
   * <code>repeated .oneflow.SbpParallel sbp_parallel = 1;</code>
   */
  int getSbpParallelCount();
  /**
   * <code>repeated .oneflow.SbpParallel sbp_parallel = 1;</code>
   */
  java.util.List<? extends org.oneflow.core.job.SbpParallelOrBuilder> 
      getSbpParallelOrBuilderList();
  /**
   * <code>repeated .oneflow.SbpParallel sbp_parallel = 1;</code>
   */
  org.oneflow.core.job.SbpParallelOrBuilder getSbpParallelOrBuilder(
      int index);
}