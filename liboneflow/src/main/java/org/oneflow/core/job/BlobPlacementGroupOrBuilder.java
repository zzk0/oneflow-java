// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/placement.proto

package org.oneflow.core.job;

public interface BlobPlacementGroupOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.BlobPlacementGroup)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>repeated .oneflow.LogicalBlobId lbi = 1;</code>
   */
  java.util.List<org.oneflow.core.register.LogicalBlobId> 
      getLbiList();
  /**
   * <code>repeated .oneflow.LogicalBlobId lbi = 1;</code>
   */
  org.oneflow.core.register.LogicalBlobId getLbi(int index);
  /**
   * <code>repeated .oneflow.LogicalBlobId lbi = 1;</code>
   */
  int getLbiCount();
  /**
   * <code>repeated .oneflow.LogicalBlobId lbi = 1;</code>
   */
  java.util.List<? extends org.oneflow.core.register.LogicalBlobIdOrBuilder> 
      getLbiOrBuilderList();
  /**
   * <code>repeated .oneflow.LogicalBlobId lbi = 1;</code>
   */
  org.oneflow.core.register.LogicalBlobIdOrBuilder getLbiOrBuilder(
      int index);

  /**
   * <code>required .oneflow.ParallelConf parallel_conf = 2;</code>
   */
  boolean hasParallelConf();
  /**
   * <code>required .oneflow.ParallelConf parallel_conf = 2;</code>
   */
  org.oneflow.core.job.ParallelConf getParallelConf();
  /**
   * <code>required .oneflow.ParallelConf parallel_conf = 2;</code>
   */
  org.oneflow.core.job.ParallelConfOrBuilder getParallelConfOrBuilder();
}