// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/register/op_blob_arg.proto

package org.oneflow.core.register;

public final class OpBlobArgOuterClass {
  private OpBlobArgOuterClass() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_OpBlobArg_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_OpBlobArg_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_OpBlobArgPair_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_OpBlobArgPair_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_OpBlobArgPairs_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_OpBlobArgPairs_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_OpBlobArgList_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_OpBlobArgList_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\'oneflow/core/register/op_blob_arg.prot" +
      "o\022\007oneflow\".\n\tOpBlobArg\022\017\n\007op_name\030\001 \002(\t" +
      "\022\020\n\010bn_in_op\030\002 \002(\t\"V\n\rOpBlobArgPair\022!\n\005f" +
      "irst\030\001 \002(\0132\022.oneflow.OpBlobArg\022\"\n\006second" +
      "\030\002 \002(\0132\022.oneflow.OpBlobArg\"6\n\016OpBlobArgP" +
      "airs\022$\n\004pair\030\001 \003(\0132\026.oneflow.OpBlobArgPa" +
      "ir\"0\n\rOpBlobArgList\022\037\n\003oba\030\001 \003(\0132\022.onefl" +
      "ow.OpBlobArgB\035\n\031org.oneflow.core.registe" +
      "rP\001"
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
    internal_static_oneflow_OpBlobArg_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_oneflow_OpBlobArg_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_OpBlobArg_descriptor,
        new java.lang.String[] { "OpName", "BnInOp", });
    internal_static_oneflow_OpBlobArgPair_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_oneflow_OpBlobArgPair_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_OpBlobArgPair_descriptor,
        new java.lang.String[] { "First", "Second", });
    internal_static_oneflow_OpBlobArgPairs_descriptor =
      getDescriptor().getMessageTypes().get(2);
    internal_static_oneflow_OpBlobArgPairs_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_OpBlobArgPairs_descriptor,
        new java.lang.String[] { "Pair", });
    internal_static_oneflow_OpBlobArgList_descriptor =
      getDescriptor().getMessageTypes().get(3);
    internal_static_oneflow_OpBlobArgList_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_OpBlobArgList_descriptor,
        new java.lang.String[] { "Oba", });
  }

  // @@protoc_insertion_point(outer_class_scope)
}
