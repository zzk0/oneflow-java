// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/mirrored_parallel.proto

package org.oneflow.core.job;

public final class MirroredParallelOuterClass {
  private MirroredParallelOuterClass() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_MirroredParallel_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_MirroredParallel_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_OptMirroredParallel_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_OptMirroredParallel_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_MirroredSignature_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_MirroredSignature_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_MirroredSignature_BnInOp2optMirroredParallelEntry_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_MirroredSignature_BnInOp2optMirroredParallelEntry_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n(oneflow/core/job/mirrored_parallel.pro" +
      "to\022\007oneflow\"\022\n\020MirroredParallel\"K\n\023OptMi" +
      "rroredParallel\0224\n\021mirrored_parallel\030\001 \001(" +
      "\0132\031.oneflow.MirroredParallel\"\330\001\n\021Mirrore" +
      "dSignature\022b\n\036bn_in_op2opt_mirrored_para" +
      "llel\030\001 \003(\0132:.oneflow.MirroredSignature.B" +
      "nInOp2optMirroredParallelEntry\032_\n\037BnInOp" +
      "2optMirroredParallelEntry\022\013\n\003key\030\001 \001(\t\022+" +
      "\n\005value\030\002 \001(\0132\034.oneflow.OptMirroredParal" +
      "lel:\0028\001B\030\n\024org.oneflow.core.jobP\001"
    };
    com.google.protobuf.Descriptors.FileDescriptor.InternalDescriptorAssigner assigner =
        new com.google.protobuf.Descriptors.FileDescriptor.    InternalDescriptorAssigner() {
          public com.google.protobuf.ExtensionRegistry assignDescriptors(
              com.google.protobuf.Descriptors.FileDescriptor root) {
            descriptor = root;
            return null;
          }
        };
    com.google.protobuf.Descriptors.FileDescriptor
      .internalBuildGeneratedFileFrom(descriptorData,
        new com.google.protobuf.Descriptors.FileDescriptor[] {
        }, assigner);
    internal_static_oneflow_MirroredParallel_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_oneflow_MirroredParallel_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_MirroredParallel_descriptor,
        new java.lang.String[] { });
    internal_static_oneflow_OptMirroredParallel_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_oneflow_OptMirroredParallel_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_OptMirroredParallel_descriptor,
        new java.lang.String[] { "MirroredParallel", });
    internal_static_oneflow_MirroredSignature_descriptor =
      getDescriptor().getMessageTypes().get(2);
    internal_static_oneflow_MirroredSignature_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_MirroredSignature_descriptor,
        new java.lang.String[] { "BnInOp2OptMirroredParallel", });
    internal_static_oneflow_MirroredSignature_BnInOp2optMirroredParallelEntry_descriptor =
      internal_static_oneflow_MirroredSignature_descriptor.getNestedTypes().get(0);
    internal_static_oneflow_MirroredSignature_BnInOp2optMirroredParallelEntry_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_MirroredSignature_BnInOp2optMirroredParallelEntry_descriptor,
        new java.lang.String[] { "Key", "Value", });
  }

  // @@protoc_insertion_point(outer_class_scope)
}