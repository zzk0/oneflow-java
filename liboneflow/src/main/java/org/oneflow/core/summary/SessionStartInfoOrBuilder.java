// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/summary/plugin_data.proto

package org.oneflow.core.summary;

public interface SessionStartInfoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.summary.SessionStartInfo)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>map&lt;string, .google.protobuf.Value&gt; hparams = 1;</code>
   */
  int getHparamsCount();
  /**
   * <code>map&lt;string, .google.protobuf.Value&gt; hparams = 1;</code>
   */
  boolean containsHparams(
      java.lang.String key);
  /**
   * Use {@link #getHparamsMap()} instead.
   */
  @java.lang.Deprecated
  java.util.Map<java.lang.String, com.google.protobuf.Value>
  getHparams();
  /**
   * <code>map&lt;string, .google.protobuf.Value&gt; hparams = 1;</code>
   */
  java.util.Map<java.lang.String, com.google.protobuf.Value>
  getHparamsMap();
  /**
   * <code>map&lt;string, .google.protobuf.Value&gt; hparams = 1;</code>
   */

  com.google.protobuf.Value getHparamsOrDefault(
      java.lang.String key,
      com.google.protobuf.Value defaultValue);
  /**
   * <code>map&lt;string, .google.protobuf.Value&gt; hparams = 1;</code>
   */

  com.google.protobuf.Value getHparamsOrThrow(
      java.lang.String key);

  /**
   * <code>required string group_name = 4;</code>
   */
  boolean hasGroupName();
  /**
   * <code>required string group_name = 4;</code>
   */
  java.lang.String getGroupName();
  /**
   * <code>required string group_name = 4;</code>
   */
  com.google.protobuf.ByteString
      getGroupNameBytes();

  /**
   * <code>required double start_time_secs = 5;</code>
   */
  boolean hasStartTimeSecs();
  /**
   * <code>required double start_time_secs = 5;</code>
   */
  double getStartTimeSecs();

  /**
   * <code>map&lt;string, .google.protobuf.Value&gt; metrics = 6;</code>
   */
  int getMetricsCount();
  /**
   * <code>map&lt;string, .google.protobuf.Value&gt; metrics = 6;</code>
   */
  boolean containsMetrics(
      java.lang.String key);
  /**
   * Use {@link #getMetricsMap()} instead.
   */
  @java.lang.Deprecated
  java.util.Map<java.lang.String, com.google.protobuf.Value>
  getMetrics();
  /**
   * <code>map&lt;string, .google.protobuf.Value&gt; metrics = 6;</code>
   */
  java.util.Map<java.lang.String, com.google.protobuf.Value>
  getMetricsMap();
  /**
   * <code>map&lt;string, .google.protobuf.Value&gt; metrics = 6;</code>
   */

  com.google.protobuf.Value getMetricsOrDefault(
      java.lang.String key,
      com.google.protobuf.Value defaultValue);
  /**
   * <code>map&lt;string, .google.protobuf.Value&gt; metrics = 6;</code>
   */

  com.google.protobuf.Value getMetricsOrThrow(
      java.lang.String key);
}
