// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/summary/event.proto

package org.oneflow.core.summary;

public final class EventOuterClass {
  private EventOuterClass() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_summary_Event_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_summary_Event_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n oneflow/core/summary/event.proto\022\017onef" +
      "low.summary\032\"oneflow/core/summary/summar" +
      "y.proto\"\244\001\n\005Event\022\021\n\twall_time\030\001 \002(\001\022\014\n\004" +
      "step\030\002 \001(\003\022\026\n\014file_version\030\003 \001(\tH\000\022\023\n\tgr" +
      "aph_def\030\004 \001(\014H\000\022+\n\007summary\030\005 \001(\0132\030.onefl" +
      "ow.summary.SummaryH\000\022\030\n\016meta_graph_def\030\t" +
      " \001(\014H\000B\006\n\004whatB\034\n\030org.oneflow.core.summa" +
      "ryP\001"
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
          org.oneflow.core.summary.SummaryOuterClass.getDescriptor(),
        }, assigner);
    internal_static_oneflow_summary_Event_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_oneflow_summary_Event_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_summary_Event_descriptor,
        new java.lang.String[] { "WallTime", "Step", "FileVersion", "GraphDef", "Summary", "MetaGraphDef", "What", });
    org.oneflow.core.summary.SummaryOuterClass.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}
