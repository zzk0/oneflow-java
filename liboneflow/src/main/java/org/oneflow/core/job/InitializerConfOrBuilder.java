// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/initializer_conf.proto

package org.oneflow.core.job;

public interface InitializerConfOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.InitializerConf)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>optional .oneflow.ConstantInitializerConf constant_conf = 1;</code>
   */
  boolean hasConstantConf();
  /**
   * <code>optional .oneflow.ConstantInitializerConf constant_conf = 1;</code>
   */
  org.oneflow.core.job.ConstantInitializerConf getConstantConf();
  /**
   * <code>optional .oneflow.ConstantInitializerConf constant_conf = 1;</code>
   */
  org.oneflow.core.job.ConstantInitializerConfOrBuilder getConstantConfOrBuilder();

  /**
   * <code>optional .oneflow.ConstantIntInitializerConf constant_int_conf = 2;</code>
   */
  boolean hasConstantIntConf();
  /**
   * <code>optional .oneflow.ConstantIntInitializerConf constant_int_conf = 2;</code>
   */
  org.oneflow.core.job.ConstantIntInitializerConf getConstantIntConf();
  /**
   * <code>optional .oneflow.ConstantIntInitializerConf constant_int_conf = 2;</code>
   */
  org.oneflow.core.job.ConstantIntInitializerConfOrBuilder getConstantIntConfOrBuilder();

  /**
   * <code>optional .oneflow.RandomUniformInitializerConf random_uniform_conf = 3;</code>
   */
  boolean hasRandomUniformConf();
  /**
   * <code>optional .oneflow.RandomUniformInitializerConf random_uniform_conf = 3;</code>
   */
  org.oneflow.core.job.RandomUniformInitializerConf getRandomUniformConf();
  /**
   * <code>optional .oneflow.RandomUniformInitializerConf random_uniform_conf = 3;</code>
   */
  org.oneflow.core.job.RandomUniformInitializerConfOrBuilder getRandomUniformConfOrBuilder();

  /**
   * <code>optional .oneflow.RandomUniformIntInitializerConf random_uniform_int_conf = 4;</code>
   */
  boolean hasRandomUniformIntConf();
  /**
   * <code>optional .oneflow.RandomUniformIntInitializerConf random_uniform_int_conf = 4;</code>
   */
  org.oneflow.core.job.RandomUniformIntInitializerConf getRandomUniformIntConf();
  /**
   * <code>optional .oneflow.RandomUniformIntInitializerConf random_uniform_int_conf = 4;</code>
   */
  org.oneflow.core.job.RandomUniformIntInitializerConfOrBuilder getRandomUniformIntConfOrBuilder();

  /**
   * <code>optional .oneflow.RandomNormalInitializerConf random_normal_conf = 5;</code>
   */
  boolean hasRandomNormalConf();
  /**
   * <code>optional .oneflow.RandomNormalInitializerConf random_normal_conf = 5;</code>
   */
  org.oneflow.core.job.RandomNormalInitializerConf getRandomNormalConf();
  /**
   * <code>optional .oneflow.RandomNormalInitializerConf random_normal_conf = 5;</code>
   */
  org.oneflow.core.job.RandomNormalInitializerConfOrBuilder getRandomNormalConfOrBuilder();

  /**
   * <code>optional .oneflow.TruncatedNormalInitializerConf truncated_normal_conf = 6;</code>
   */
  boolean hasTruncatedNormalConf();
  /**
   * <code>optional .oneflow.TruncatedNormalInitializerConf truncated_normal_conf = 6;</code>
   */
  org.oneflow.core.job.TruncatedNormalInitializerConf getTruncatedNormalConf();
  /**
   * <code>optional .oneflow.TruncatedNormalInitializerConf truncated_normal_conf = 6;</code>
   */
  org.oneflow.core.job.TruncatedNormalInitializerConfOrBuilder getTruncatedNormalConfOrBuilder();

  /**
   * <code>optional .oneflow.XavierInitializerConf xavier_conf = 7;</code>
   */
  boolean hasXavierConf();
  /**
   * <code>optional .oneflow.XavierInitializerConf xavier_conf = 7;</code>
   */
  org.oneflow.core.job.XavierInitializerConf getXavierConf();
  /**
   * <code>optional .oneflow.XavierInitializerConf xavier_conf = 7;</code>
   */
  org.oneflow.core.job.XavierInitializerConfOrBuilder getXavierConfOrBuilder();

  /**
   * <code>optional .oneflow.MsraInitializerConf msra_conf = 8;</code>
   */
  boolean hasMsraConf();
  /**
   * <code>optional .oneflow.MsraInitializerConf msra_conf = 8;</code>
   */
  org.oneflow.core.job.MsraInitializerConf getMsraConf();
  /**
   * <code>optional .oneflow.MsraInitializerConf msra_conf = 8;</code>
   */
  org.oneflow.core.job.MsraInitializerConfOrBuilder getMsraConfOrBuilder();

  /**
   * <code>optional .oneflow.RangeInitializerConf range_conf = 9;</code>
   */
  boolean hasRangeConf();
  /**
   * <code>optional .oneflow.RangeInitializerConf range_conf = 9;</code>
   */
  org.oneflow.core.job.RangeInitializerConf getRangeConf();
  /**
   * <code>optional .oneflow.RangeInitializerConf range_conf = 9;</code>
   */
  org.oneflow.core.job.RangeInitializerConfOrBuilder getRangeConfOrBuilder();

  /**
   * <code>optional .oneflow.IntRangeInitializerConf int_range_conf = 10;</code>
   */
  boolean hasIntRangeConf();
  /**
   * <code>optional .oneflow.IntRangeInitializerConf int_range_conf = 10;</code>
   */
  org.oneflow.core.job.IntRangeInitializerConf getIntRangeConf();
  /**
   * <code>optional .oneflow.IntRangeInitializerConf int_range_conf = 10;</code>
   */
  org.oneflow.core.job.IntRangeInitializerConfOrBuilder getIntRangeConfOrBuilder();

  /**
   * <code>optional .oneflow.VarianceScalingInitializerConf variance_scaling_conf = 11;</code>
   */
  boolean hasVarianceScalingConf();
  /**
   * <code>optional .oneflow.VarianceScalingInitializerConf variance_scaling_conf = 11;</code>
   */
  org.oneflow.core.job.VarianceScalingInitializerConf getVarianceScalingConf();
  /**
   * <code>optional .oneflow.VarianceScalingInitializerConf variance_scaling_conf = 11;</code>
   */
  org.oneflow.core.job.VarianceScalingInitializerConfOrBuilder getVarianceScalingConfOrBuilder();

  /**
   * <code>optional .oneflow.EmptyInitializerConf empty_conf = 12;</code>
   */
  boolean hasEmptyConf();
  /**
   * <code>optional .oneflow.EmptyInitializerConf empty_conf = 12;</code>
   */
  org.oneflow.core.job.EmptyInitializerConf getEmptyConf();
  /**
   * <code>optional .oneflow.EmptyInitializerConf empty_conf = 12;</code>
   */
  org.oneflow.core.job.EmptyInitializerConfOrBuilder getEmptyConfOrBuilder();

  public org.oneflow.core.job.InitializerConf.TypeCase getTypeCase();
}
