// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/operator/op_conf.proto

package org.oneflow.core.operator;

public interface ReentrantLockOpConfOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.ReentrantLockOpConf)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>required string start = 1;</code>
   */
  boolean hasStart();
  /**
   * <code>required string start = 1;</code>
   */
  java.lang.String getStart();
  /**
   * <code>required string start = 1;</code>
   */
  com.google.protobuf.ByteString
      getStartBytes();

  /**
   * <code>optional string end = 2;</code>
   */
  boolean hasEnd();
  /**
   * <code>optional string end = 2;</code>
   */
  java.lang.String getEnd();
  /**
   * <code>optional string end = 2;</code>
   */
  com.google.protobuf.ByteString
      getEndBytes();

  /**
   * <code>required string out = 3;</code>
   */
  boolean hasOut();
  /**
   * <code>required string out = 3;</code>
   */
  java.lang.String getOut();
  /**
   * <code>required string out = 3;</code>
   */
  com.google.protobuf.ByteString
      getOutBytes();

  /**
   * <code>repeated .oneflow.Int64List lock_id2intersecting_lock_ids = 4;</code>
   */
  java.util.List<org.oneflow.core.record.Int64List> 
      getLockId2IntersectingLockIdsList();
  /**
   * <code>repeated .oneflow.Int64List lock_id2intersecting_lock_ids = 4;</code>
   */
  org.oneflow.core.record.Int64List getLockId2IntersectingLockIds(int index);
  /**
   * <code>repeated .oneflow.Int64List lock_id2intersecting_lock_ids = 4;</code>
   */
  int getLockId2IntersectingLockIdsCount();
  /**
   * <code>repeated .oneflow.Int64List lock_id2intersecting_lock_ids = 4;</code>
   */
  java.util.List<? extends org.oneflow.core.record.Int64ListOrBuilder> 
      getLockId2IntersectingLockIdsOrBuilderList();
  /**
   * <code>repeated .oneflow.Int64List lock_id2intersecting_lock_ids = 4;</code>
   */
  org.oneflow.core.record.Int64ListOrBuilder getLockId2IntersectingLockIdsOrBuilder(
      int index);
}
