// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/placement.proto

package org.oneflow.core.job;

public interface ParallelConfOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.ParallelConf)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>repeated string device_name = 1;</code>
   */
  java.util.List<java.lang.String>
      getDeviceNameList();
  /**
   * <code>repeated string device_name = 1;</code>
   */
  int getDeviceNameCount();
  /**
   * <code>repeated string device_name = 1;</code>
   */
  java.lang.String getDeviceName(int index);
  /**
   * <code>repeated string device_name = 1;</code>
   */
  com.google.protobuf.ByteString
      getDeviceNameBytes(int index);

  /**
   * <code>required string device_tag = 2;</code>
   */
  boolean hasDeviceTag();
  /**
   * <code>required string device_tag = 2;</code>
   */
  java.lang.String getDeviceTag();
  /**
   * <code>required string device_tag = 2;</code>
   */
  com.google.protobuf.ByteString
      getDeviceTagBytes();

  /**
   * <code>optional .oneflow.ShapeProto hierarchy = 3;</code>
   */
  boolean hasHierarchy();
  /**
   * <code>optional .oneflow.ShapeProto hierarchy = 3;</code>
   */
  oneflow.Shape.ShapeProto getHierarchy();
  /**
   * <code>optional .oneflow.ShapeProto hierarchy = 3;</code>
   */
  oneflow.Shape.ShapeProtoOrBuilder getHierarchyOrBuilder();
}