// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/job/regularizer_conf.proto

package org.oneflow.core.job;

public final class RegularizerConfOuterClass {
  private RegularizerConfOuterClass() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_L1L2RegularizerConf_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_L1L2RegularizerConf_fieldAccessorTable;
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_RegularizerConf_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_RegularizerConf_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n\'oneflow/core/job/regularizer_conf.prot" +
      "o\022\007oneflow\"3\n\023L1L2RegularizerConf\022\r\n\002l1\030" +
      "\001 \001(\002:\0010\022\r\n\002l2\030\002 \001(\002:\0010\"M\n\017RegularizerCo" +
      "nf\0222\n\nl1_l2_conf\030\001 \001(\0132\034.oneflow.L1L2Reg" +
      "ularizerConfH\000B\006\n\004typeB\030\n\024org.oneflow.co" +
      "re.jobP\001"
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
    internal_static_oneflow_L1L2RegularizerConf_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_oneflow_L1L2RegularizerConf_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_L1L2RegularizerConf_descriptor,
        new java.lang.String[] { "L1", "L2", });
    internal_static_oneflow_RegularizerConf_descriptor =
      getDescriptor().getMessageTypes().get(1);
    internal_static_oneflow_RegularizerConf_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_RegularizerConf_descriptor,
        new java.lang.String[] { "L1L2Conf", "Type", });
  }

  // @@protoc_insertion_point(outer_class_scope)
}