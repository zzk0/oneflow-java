// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/parallel_conf_signature.proto

package org.oneflow.core.job;

public final class ParallelConfSignatureOuterClass {
  private ParallelConfSignatureOuterClass() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_ParallelConfSignature_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_ParallelConfSignature_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_ParallelConfSignature_BnInOp2parallelConfEntry_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_ParallelConfSignature_BnInOp2parallelConfEntry_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n.oneflow/core/job/parallel_conf_signatu" +
      "re.proto\022\007oneflow\032 oneflow/core/job/plac" +
      "ement.proto\"\364\001\n\025ParallelConfSignature\022/\n" +
      "\020op_parallel_conf\030\001 \001(\0132\025.oneflow.Parall" +
      "elConf\022W\n\026bn_in_op2parallel_conf\030\002 \003(\01327" +
      ".oneflow.ParallelConfSignature.BnInOp2pa" +
      "rallelConfEntry\032Q\n\030BnInOp2parallelConfEn" +
      "try\022\013\n\003key\030\001 \001(\t\022$\n\005value\030\002 \001(\0132\025.oneflo" +
      "w.ParallelConf:\0028\001B\030\n\024org.oneflow.core.j" +
      "obP\001"
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
          oneflow.PlacementOuterClass.getDescriptor(),
        }, assigner);
    internal_static_oneflow_ParallelConfSignature_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_oneflow_ParallelConfSignature_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_ParallelConfSignature_descriptor,
        new java.lang.String[] { "OpParallelConf", "BnInOp2ParallelConf", });
    internal_static_oneflow_ParallelConfSignature_BnInOp2parallelConfEntry_descriptor =
      internal_static_oneflow_ParallelConfSignature_descriptor.getNestedTypes().get(0);
    internal_static_oneflow_ParallelConfSignature_BnInOp2parallelConfEntry_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_ParallelConfSignature_BnInOp2parallelConfEntry_descriptor,
        new java.lang.String[] { "Key", "Value", });
    oneflow.PlacementOuterClass.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}
