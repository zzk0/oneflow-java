// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: oneflow/core/register/tensor_slice_view.proto

package org.oneflow.core.register;

public final class TensorSliceView {
  private TensorSliceView() {}
  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistryLite registry) {
  }

  public static void registerAllExtensions(
      com.google.protobuf.ExtensionRegistry registry) {
    registerAllExtensions(
        (com.google.protobuf.ExtensionRegistryLite) registry);
  }
  static final com.google.protobuf.Descriptors.Descriptor
    internal_static_oneflow_TensorSliceViewProto_descriptor;
  static final 
    com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internal_static_oneflow_TensorSliceViewProto_fieldAccessorTable;

  public static com.google.protobuf.Descriptors.FileDescriptor
      getDescriptor() {
    return descriptor;
  }
  private static  com.google.protobuf.Descriptors.FileDescriptor
      descriptor;
  static {
    java.lang.String[] descriptorData = {
      "\n-oneflow/core/register/tensor_slice_vie" +
      "w.proto\022\007oneflow\032\037oneflow/core/common/ra" +
      "nge.proto\"8\n\024TensorSliceViewProto\022 \n\003dim" +
      "\030\001 \003(\0132\023.oneflow.RangeProtoB\035\n\031org.onefl" +
      "ow.core.registerP\001"
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
          oneflow.Range.getDescriptor(),
        }, assigner);
    internal_static_oneflow_TensorSliceViewProto_descriptor =
      getDescriptor().getMessageTypes().get(0);
    internal_static_oneflow_TensorSliceViewProto_fieldAccessorTable = new
      com.google.protobuf.GeneratedMessageV3.FieldAccessorTable(
        internal_static_oneflow_TensorSliceViewProto_descriptor,
        new java.lang.String[] { "Dim", });
    oneflow.Range.getDescriptor();
  }

  // @@protoc_insertion_point(outer_class_scope)
}
