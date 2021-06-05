// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/operator/op_conf.proto

package org.oneflow.core.operator;

public interface LearningRateScheduleOpConfOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.LearningRateScheduleOpConf)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>required string train_step = 1;</code>
   */
  boolean hasTrainStep();
  /**
   * <code>required string train_step = 1;</code>
   */
  java.lang.String getTrainStep();
  /**
   * <code>required string train_step = 1;</code>
   */
  com.google.protobuf.ByteString
      getTrainStepBytes();

  /**
   * <code>required string out = 2;</code>
   */
  boolean hasOut();
  /**
   * <code>required string out = 2;</code>
   */
  java.lang.String getOut();
  /**
   * <code>required string out = 2;</code>
   */
  com.google.protobuf.ByteString
      getOutBytes();

  /**
   * <code>required float learning_rate = 3;</code>
   */
  boolean hasLearningRate();
  /**
   * <code>required float learning_rate = 3;</code>
   */
  float getLearningRate();

  /**
   * <code>optional .oneflow.LearningRateDecayConf learning_rate_decay = 4;</code>
   */
  boolean hasLearningRateDecay();
  /**
   * <code>optional .oneflow.LearningRateDecayConf learning_rate_decay = 4;</code>
   */
  org.oneflow.core.job.LearningRateDecayConf getLearningRateDecay();
  /**
   * <code>optional .oneflow.LearningRateDecayConf learning_rate_decay = 4;</code>
   */
  org.oneflow.core.job.LearningRateDecayConfOrBuilder getLearningRateDecayOrBuilder();

  /**
   * <code>optional .oneflow.WarmupConf warmup_conf = 5;</code>
   */
  boolean hasWarmupConf();
  /**
   * <code>optional .oneflow.WarmupConf warmup_conf = 5;</code>
   */
  org.oneflow.core.job.WarmupConf getWarmupConf();
  /**
   * <code>optional .oneflow.WarmupConf warmup_conf = 5;</code>
   */
  org.oneflow.core.job.WarmupConfOrBuilder getWarmupConfOrBuilder();
}
