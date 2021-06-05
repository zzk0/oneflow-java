// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/scope.proto

package org.oneflow.core.job;

public final class Scope {
  private Scope() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_ScopeProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_ScopeProto_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_ScopeProto_AttrName2attrValueEntry_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_ScopeProto_AttrName2attrValueEntry_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\034oneflow/core/job/scope.proto\022\007oneflow\032" +
      "(oneflow/core/job/mirrored_parallel.prot" +
      "o\032)oneflow/core/framework/user_op_attr.p" +
      "roto\"\374\003\n\nScopeProto\022\032\n\022job_desc_symbol_i" +
      "d\030\024 \002(\003\022&\n\036device_parallel_desc_symbol_i" +
      "d\030\036 \002(\003\022$\n\034host_parallel_desc_symbol_id\030" +
      "( \002(\003\022\'\n\031enable_cpu_alternative_op\030) \001(\010" +
      ":\004true\022@\n\032opt_mirrored_parallel_conf\0302 \002" +
      "(\0132\034.oneflow.OptMirroredParallel\022\036\n\026scop" +
      "e_op_name_prefixes\030< \003(\t\022\036\n\026parent_scope",
      "_symbol_id\030F \001(\003\022\022\n\nsession_id\030P \002(\003\022I\n\024" +
      "attr_name2attr_value\030Z \003(\0132+.oneflow.Sco" +
      "peProto.AttrName2attrValueEntry\022+\n\025calcu" +
      "lation_pass_name\030d \001(\t:\014forward_pass\032M\n\027" +
      "AttrName2attrValueEntry\022\013\n\003key\030\001 \001(\t\022!\n\005" +
      "value\030\002 \001(\0132\022.oneflow.AttrValue:\0028\001B\030\n\024o" +
      "rg.oneflow.core.jobP\001"
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
          oneflow.MirroredParallelOuterClass.getDescriptor(),
          org.oneflow.core.framework.UserOpAttr.getDescriptor(),
        }, assigner);
    internal_static_oneflow_ScopeProto_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_oneflow_ScopeProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_ScopeProto_descriptor,
        new java.lang.String[] { "JobDescSymbolId", "DeviceParallelDescSymbolId", "HostParallelDescSymbolId", "EnableCpuAlternativeOp", "OptMirroredParallelConf", "ScopeOpNamePrefixes", "ParentScopeSymbolId", "SessionId", "AttrName2AttrValue", "CalculationPassName", });
    internal_static_oneflow_ScopeProto_AttrName2attrValueEntry_descriptor =
      internal_static_oneflow_ScopeProto_descriptor.getNestedTypes().get(0);
    internal_static_oneflow_ScopeProto_AttrName2attrValueEntry_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_ScopeProto_AttrName2attrValueEntry_descriptor,
        new java.lang.String[] { "Key", "Value", });
    oneflow.MirroredParallelOuterClass.getDescriptor();
    org.oneflow.core.framework.UserOpAttr.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}
