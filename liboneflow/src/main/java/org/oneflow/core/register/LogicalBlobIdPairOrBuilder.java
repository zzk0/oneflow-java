// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/register/logical_blob_id.proto

package org.oneflow.core.register;

public interface LogicalBlobIdPairOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.LogicalBlobIdPair)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>required .oneflow.LogicalBlobId first = 1;</code>
   */
  boolean hasFirst();
  /**
   * <code>required .oneflow.LogicalBlobId first = 1;</code>
   */
  org.oneflow.core.register.LogicalBlobId getFirst();
  /**
   * <code>required .oneflow.LogicalBlobId first = 1;</code>
   */
  org.oneflow.core.register.LogicalBlobIdOrBuilder getFirstOrBuilder();

  /**
   * <code>required .oneflow.LogicalBlobId second = 2;</code>
   */
  boolean hasSecond();
  /**
   * <code>required .oneflow.LogicalBlobId second = 2;</code>
   */
  org.oneflow.core.register.LogicalBlobId getSecond();
  /**
   * <code>required .oneflow.LogicalBlobId second = 2;</code>
   */
  org.oneflow.core.register.LogicalBlobIdOrBuilder getSecondOrBuilder();
}