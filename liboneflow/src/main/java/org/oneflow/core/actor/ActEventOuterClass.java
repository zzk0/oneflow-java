// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/actor/act_event.proto

package org.oneflow.core.actor;

public final class ActEventOuterClass {
  private ActEventOuterClass() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_ReadableRegstInfo_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_ReadableRegstInfo_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_ActEvent_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_ActEvent_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\"oneflow/core/actor/act_event.proto\022\007on" +
      "eflow\":\n\021ReadableRegstInfo\022\025\n\rregst_desc" +
      "_id\030\001 \002(\003\022\016\n\006act_id\030\002 \002(\003\"\326\001\n\010ActEvent\022\033" +
      "\n\023is_experiment_phase\030\001 \002(\010\022\020\n\010actor_id\030" +
      "\002 \002(\003\022\026\n\016work_stream_id\030\003 \002(\003\022\016\n\006act_id\030" +
      "\004 \002(\003\022\022\n\nready_time\030\005 \002(\001\022\022\n\nstart_time\030" +
      "\006 \002(\001\022\021\n\tstop_time\030\007 \002(\001\0228\n\024readable_reg" +
      "st_infos\030\n \003(\0132\032.oneflow.ReadableRegstIn" +
      "foB\032\n\026org.oneflow.core.actorP\001"
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
    internal_static_oneflow_ReadableRegstInfo_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_oneflow_ReadableRegstInfo_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_ReadableRegstInfo_descriptor,
        new java.lang.String[] { "RegstDescId", "ActId", });
    internal_static_oneflow_ActEvent_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_oneflow_ActEvent_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_ActEvent_descriptor,
        new java.lang.String[] { "IsExperimentPhase", "ActorId", "WorkStreamId", "ActId", "ReadyTime", "StartTime", "StopTime", "ReadableRegstInfos", });
  }

  // @@protoc_insertion_point(outer_class_scope)
}