// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/common/cfg_reflection_test.proto

package org.oneflow.core.common;

public interface ReflectionTestBarOrBuilder extends
    // @@protoc_insertion_point(interface_extends:oneflow.ReflectionTestBar)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <code>required .oneflow.ReflectionTestFoo required_foo = 1;</code>
   */
  boolean hasRequiredFoo();
  /**
   * <code>required .oneflow.ReflectionTestFoo required_foo = 1;</code>
   */
  org.oneflow.core.common.ReflectionTestFoo getRequiredFoo();
  /**
   * <code>required .oneflow.ReflectionTestFoo required_foo = 1;</code>
   */
  org.oneflow.core.common.ReflectionTestFooOrBuilder getRequiredFooOrBuilder();

  /**
   * <code>optional .oneflow.ReflectionTestFoo optional_foo = 2;</code>
   */
  boolean hasOptionalFoo();
  /**
   * <code>optional .oneflow.ReflectionTestFoo optional_foo = 2;</code>
   */
  org.oneflow.core.common.ReflectionTestFoo getOptionalFoo();
  /**
   * <code>optional .oneflow.ReflectionTestFoo optional_foo = 2;</code>
   */
  org.oneflow.core.common.ReflectionTestFooOrBuilder getOptionalFooOrBuilder();

  /**
   * <code>repeated .oneflow.ReflectionTestFoo repeated_foo = 3;</code>
   */
  java.util.List<org.oneflow.core.common.ReflectionTestFoo> 
      getRepeatedFooList();
  /**
   * <code>repeated .oneflow.ReflectionTestFoo repeated_foo = 3;</code>
   */
  org.oneflow.core.common.ReflectionTestFoo getRepeatedFoo(int index);
  /**
   * <code>repeated .oneflow.ReflectionTestFoo repeated_foo = 3;</code>
   */
  int getRepeatedFooCount();
  /**
   * <code>repeated .oneflow.ReflectionTestFoo repeated_foo = 3;</code>
   */
  java.util.List<? extends org.oneflow.core.common.ReflectionTestFooOrBuilder> 
      getRepeatedFooOrBuilderList();
  /**
   * <code>repeated .oneflow.ReflectionTestFoo repeated_foo = 3;</code>
   */
  org.oneflow.core.common.ReflectionTestFooOrBuilder getRepeatedFooOrBuilder(
      int index);

  /**
   * <code>map&lt;int32, .oneflow.ReflectionTestFoo&gt; map_foo = 4;</code>
   */
  int getMapFooCount();
  /**
   * <code>map&lt;int32, .oneflow.ReflectionTestFoo&gt; map_foo = 4;</code>
   */
  boolean containsMapFoo(
      int key);
  /**
   * Use {@link #getMapFooMap()} instead.
   */
  @java.lang.Deprecated
  java.util.Map<java.lang.Integer, org.oneflow.core.common.ReflectionTestFoo>
  getMapFoo();
  /**
   * <code>map&lt;int32, .oneflow.ReflectionTestFoo&gt; map_foo = 4;</code>
   */
  java.util.Map<java.lang.Integer, org.oneflow.core.common.ReflectionTestFoo>
  getMapFooMap();
  /**
   * <code>map&lt;int32, .oneflow.ReflectionTestFoo&gt; map_foo = 4;</code>
   */

  org.oneflow.core.common.ReflectionTestFoo getMapFooOrDefault(
      int key,
      org.oneflow.core.common.ReflectionTestFoo defaultValue);
  /**
   * <code>map&lt;int32, .oneflow.ReflectionTestFoo&gt; map_foo = 4;</code>
   */

  org.oneflow.core.common.ReflectionTestFoo getMapFooOrThrow(
      int key);

  /**
   * <code>optional .oneflow.ReflectionTestFoo oneof_foo = 5;</code>
   */
  boolean hasOneofFoo();
  /**
   * <code>optional .oneflow.ReflectionTestFoo oneof_foo = 5;</code>
   */
  org.oneflow.core.common.ReflectionTestFoo getOneofFoo();
  /**
   * <code>optional .oneflow.ReflectionTestFoo oneof_foo = 5;</code>
   */
  org.oneflow.core.common.ReflectionTestFooOrBuilder getOneofFooOrBuilder();

  /**
   * <code>optional .oneflow.ReflectionTestFoo another_oneof_foo = 6;</code>
   */
  boolean hasAnotherOneofFoo();
  /**
   * <code>optional .oneflow.ReflectionTestFoo another_oneof_foo = 6;</code>
   */
  org.oneflow.core.common.ReflectionTestFoo getAnotherOneofFoo();
  /**
   * <code>optional .oneflow.ReflectionTestFoo another_oneof_foo = 6;</code>
   */
  org.oneflow.core.common.ReflectionTestFooOrBuilder getAnotherOneofFooOrBuilder();

  public org.oneflow.core.common.ReflectionTestBar.TypeCase getTypeCase();
}
