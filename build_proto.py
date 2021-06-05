import os
from shutil import copytree, ignore_patterns

# only work on Linux currently
DST_DIR_ROOT = "liboneflow/src/main/java/org/oneflow/core"
SRC_DIR_ROOT = "oneflow-master/oneflow/core"
COPY_TO = "build/oneflow/core"
BUILD_DIR = "build"
ONEFLOW_PROTO_ROOT = "build/oneflow"
PROTO_JAVA_DIR = "build/proto-java"


# copy all .proto files to build, keep directory structure
copytree(SRC_DIR_ROOT, COPY_TO, ignore=ignore_patterns('*.h', '*.cpp', '*.hpp', '*.cuh', '*.cu'))
os.makedirs(PROTO_JAVA_DIR)

# add extra lines to .proto file and compile
def print_tree(path, package):
    for item in os.listdir(path):
        if os.path.isfile(path + os.sep + item):
            suffix = item.split('.')[-1]
            if suffix == 'proto':
                with open(path + os.sep + item, 'a+') as f:
                    f.write('option java_multiple_files = true;\n')
                    f.write('option java_package = "' + package + '";\n')
                command = 'protoc -I={} --java_out={} {}'.format(
                    BUILD_DIR,
                    PROTO_JAVA_DIR,
                    path + os.sep + item
                )
                os.system(command)
        if os.path.isdir(path + os.sep + item):
            print_tree(path + os.sep + item, package + '.' + item)


print_tree(COPY_TO, 'org.oneflow.core')