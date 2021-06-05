// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/memory/memory_block.proto

package org.oneflow.core.memory;

public interface MemBlockProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.MemBlockProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>required int64 mem_block_id = 1;</code>
   */
  boolean hasMemBlockId();
  /**
   * <code>required int64 mem_block_id = 1;</code>
   */
  long getMemBlockId();

  /**
   * <code>repeated int64 job_id = 2;</code>
   */
  java.util.List<java.lang.Long> getJobIdList();
  /**
   * <code>repeated int64 job_id = 2;</code>
   */
  int getJobIdCount();
  /**
   * <code>repeated int64 job_id = 2;</code>
   */
  long getJobId(int index);

  /**
   * <code>required int64 machine_id = 3;</code>
   */
  boolean hasMachineId();
  /**
   * <code>required int64 machine_id = 3;</code>
   */
  long getMachineId();

  /**
   * <code>required .oneflow.MemoryCase mem_case = 4;</code>
   */
  boolean hasMemCase();
  /**
   * <code>required .oneflow.MemoryCase mem_case = 4;</code>
   */
  org.oneflow.core.memory.MemoryCase getMemCase();
  /**
   * <code>required .oneflow.MemoryCase mem_case = 4;</code>
   */
  org.oneflow.core.memory.MemoryCaseOrBuilder getMemCaseOrBuilder();

  /**
   * <code>required bool enable_reuse_mem = 5;</code>
   */
  boolean hasEnableReuseMem();
  /**
   * <code>required bool enable_reuse_mem = 5;</code>
   */
  boolean getEnableReuseMem();

  /**
   * <code>optional int64 chunk_id = 6 [default = -1];</code>
   */
  boolean hasChunkId();
  /**
   * <code>optional int64 chunk_id = 6 [default = -1];</code>
   */
  long getChunkId();

  /**
   * <code>optional int64 chunk_offset = 7 [default = -1];</code>
   */
  boolean hasChunkOffset();
  /**
   * <code>optional int64 chunk_offset = 7 [default = -1];</code>
   */
  long getChunkOffset();

  /**
   * <code>required int64 mem_size = 8;</code>
   */
  boolean hasMemSize();
  /**
   * <code>required int64 mem_size = 8;</code>
   */
  long getMemSize();

  /**
   * <pre>
   * NOTE(chengcheng): thrd id hint is used by packed separated block group order.
   * </pre>
   *
   * <code>optional int64 thrd_id_hint = 9 [default = -1];</code>
   */
  boolean hasThrdIdHint();
  /**
   * <pre>
   * NOTE(chengcheng): thrd id hint is used by packed separated block group order.
   * </pre>
   *
   * <code>optional int64 thrd_id_hint = 9 [default = -1];</code>
   */
  long getThrdIdHint();
}
