// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/initializer_conf.proto

package org.oneflow.core.job;

public interface RandomNormalInitializerConfOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.RandomNormalInitializerConf)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>optional float mean = 1 [default = 0];</code>
   */
  boolean hasMean();
  /**
   * <code>optional float mean = 1 [default = 0];</code>
   */
  float getMean();

  /**
   * <code>optional float std = 2 [default = 1];</code>
   */
  boolean hasStd();
  /**
   * <code>optional float std = 2 [default = 1];</code>
   */
  float getStd();
}