// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/parallel_signature.proto

package org.oneflow.core.job;

public final class ParallelSignatureOuterClass {
  private ParallelSignatureOuterClass() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_ParallelSignature_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_ParallelSignature_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_ParallelSignature_BnInOp2parallelDescSymbolIdEntry_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_ParallelSignature_BnInOp2parallelDescSymbolIdEntry_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n)oneflow/core/job/parallel_signature.pr" +
      "oto\022\007oneflow\"\342\001\n\021ParallelSignature\022\"\n\032op" +
      "_parallel_desc_symbol_id\030\001 \001(\003\022e\n bn_in_" +
      "op2parallel_desc_symbol_id\030\002 \003(\0132;.onefl" +
      "ow.ParallelSignature.BnInOp2parallelDesc" +
      "SymbolIdEntry\032B\n BnInOp2parallelDescSymb" +
      "olIdEntry\022\013\n\003key\030\001 \001(\t\022\r\n\005value\030\002 \001(\003:\0028" +
      "\001B\030\n\024org.oneflow.core.jobP\001"
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
    internal_static_oneflow_ParallelSignature_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_oneflow_ParallelSignature_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_ParallelSignature_descriptor,
        new java.lang.String[] { "OpParallelDescSymbolId", "BnInOp2ParallelDescSymbolId", });
    internal_static_oneflow_ParallelSignature_BnInOp2parallelDescSymbolIdEntry_descriptor =
      internal_static_oneflow_ParallelSignature_descriptor.getNestedTypes().get(0);
    internal_static_oneflow_ParallelSignature_BnInOp2parallelDescSymbolIdEntry_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_ParallelSignature_BnInOp2parallelDescSymbolIdEntry_descriptor,
        new java.lang.String[] { "Key", "Value", });
  }

  // @@protoc_insertion_point(outer_class_scope)
}