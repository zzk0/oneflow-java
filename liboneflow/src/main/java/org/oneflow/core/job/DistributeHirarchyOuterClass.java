// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/distribute_hirarchy.proto

package org.oneflow.core.job;

public final class DistributeHirarchyOuterClass {
  private DistributeHirarchyOuterClass() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_DistributeDim_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_DistributeDim_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_DistributeHirarchy_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_DistributeHirarchy_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n*oneflow/core/job/distribute_hirarchy.p" +
      "roto\022\007oneflow\032#oneflow/core/job/sbp_para" +
      "llel.proto\"\205\001\n\rDistributeDim\0220\n\017distribu" +
      "te_type\030\001 \002(\0162\027.oneflow.DistributeType\022*" +
      "\n\014sbp_parallel\030\002 \002(\0132\024.oneflow.SbpParall" +
      "el\022\026\n\016distribute_num\030\003 \002(\003\"9\n\022Distribute" +
      "Hirarchy\022#\n\003dim\030\001 \003(\0132\026.oneflow.Distribu" +
      "teDim*W\n\016DistributeType\022\032\n\026kInvalidDistr" +
      "ibuteType\020\000\022\024\n\020kSpaceDistribute\020\002\022\023\n\017kTi" +
      "meDistribute\020\003B\030\n\024org.oneflow.core.jobP\001"
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
          oneflow.SbpParallelOuterClass.getDescriptor(),
        }, assigner);
    internal_static_oneflow_DistributeDim_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_oneflow_DistributeDim_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_DistributeDim_descriptor,
        new java.lang.String[] { "DistributeType", "SbpParallel", "DistributeNum", });
    internal_static_oneflow_DistributeHirarchy_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_oneflow_DistributeHirarchy_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_DistributeHirarchy_descriptor,
        new java.lang.String[] { "Dim", });
    oneflow.SbpParallelOuterClass.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}
