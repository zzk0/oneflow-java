// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/comm_network/ibverbs/ibverbs.proto

package org.oneflow.core.comm_network.ibverbs;

public interface IBVerbsConnectionInfoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.IBVerbsConnectionInfo)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>required uint32 lid = 1;</code>
   */
  boolean hasLid();
  /**
   * <code>required uint32 lid = 1;</code>
   */
  int getLid();

  /**
   * <code>required uint32 qp_num = 2;</code>
   */
  boolean hasQpNum();
  /**
   * <code>required uint32 qp_num = 2;</code>
   */
  int getQpNum();

  /**
   * <code>required uint64 subnet_prefix = 3;</code>
   */
  boolean hasSubnetPrefix();
  /**
   * <code>required uint64 subnet_prefix = 3;</code>
   */
  long getSubnetPrefix();

  /**
   * <code>required uint64 interface_id = 4;</code>
   */
  boolean hasInterfaceId();
  /**
   * <code>required uint64 interface_id = 4;</code>
   */
  long getInterfaceId();
}
