// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/sub_plan.proto

package org.oneflow.core.job;

public final class SubPlanOuterClass {
  private SubPlanOuterClass() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_ThrdIds_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_ThrdIds_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_ClusterThrdIds_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_ClusterThrdIds_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_ClusterThrdIds_MachineId2thrdIdsEntry_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_ClusterThrdIds_MachineId2thrdIdsEntry_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_SubPlan_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_SubPlan_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\037oneflow/core/job/sub_plan.proto\022\007onefl" +
      "ow\032\033oneflow/core/job/task.proto\"\032\n\007ThrdI" +
      "ds\022\017\n\007thrd_id\030\001 \003(\003\"\251\001\n\016ClusterThrdIds\022K" +
      "\n\023machine_id2thrd_ids\030\001 \003(\0132..oneflow.Cl" +
      "usterThrdIds.MachineId2thrdIdsEntry\032J\n\026M" +
      "achineId2thrdIdsEntry\022\013\n\003key\030\001 \001(\003\022\037\n\005va" +
      "lue\030\002 \001(\0132\020.oneflow.ThrdIds:\0028\001\"+\n\007SubPl" +
      "an\022 \n\004task\030\001 \003(\0132\022.oneflow.TaskProtoB\030\n\024" +
      "org.oneflow.core.jobP\001"
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
          org.oneflow.core.job.Task.getDescriptor(),
        }, assigner);
    internal_static_oneflow_ThrdIds_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_oneflow_ThrdIds_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_ThrdIds_descriptor,
        new java.lang.String[] { "ThrdId", });
    internal_static_oneflow_ClusterThrdIds_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_oneflow_ClusterThrdIds_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_ClusterThrdIds_descriptor,
        new java.lang.String[] { "MachineId2ThrdIds", });
    internal_static_oneflow_ClusterThrdIds_MachineId2thrdIdsEntry_descriptor =
      internal_static_oneflow_ClusterThrdIds_descriptor.getNestedTypes().get(0);
    internal_static_oneflow_ClusterThrdIds_MachineId2thrdIdsEntry_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_ClusterThrdIds_MachineId2thrdIdsEntry_descriptor,
        new java.lang.String[] { "Key", "Value", });
    internal_static_oneflow_SubPlan_descriptor =
      getDescriptor().getMessageTypes().get(2);
    internal_static_oneflow_SubPlan_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_SubPlan_descriptor,
        new java.lang.String[] { "Task", });
    org.oneflow.core.job.Task.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}