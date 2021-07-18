# oneflow-java

[English](README.md)

为 OneFlow 添加一门新的前端语言

# 使用指南

为了使用 OneFlow 的 Java 版本，你需要进行一些配置。

## 构建 liboneflow.so

下载 oneflow 源码，并且切换到 liboneflow_java 分支，接着执行命令以构建。步骤如下。

**clone 我的源码**

```
git clone https://github.com/zzk0/oneflow --depth=1
```

**拉取远程分支**

拉取 libOneFlow_java 这个分支，方法因 git 版本而异，下面给出我的步骤(git version 2.17.1)。

注意看 fetch 拉取到的内容，比如 `libOneFlow_java -> FETCH_HEAD`，那么下一个命令就需要从 `FETCH_HEAD` 切换分支。

```
git fetch origin libOneFlow_java
git checkout -b libOneFlow_java FETCH_HEAD
```

**构建源码**

确保您的机子上安装了 Java，CMake 会搜索 JNI 的头文件，如果不存在，那么 CMake 会失败。

```
mkdir build
cd build
cmake .. -DBUILD_JNI=ON
make oneflow -j$(nproc)
```

**设置环境变量**

将 `oneflow/build/oneflow/api/java` 加入到 `LD_LIBRARY_PATH` 中，`oneflow/build/oneflow/api/java` 这个目录下面有 liboneflow.so，这是 Java 想要寻找到的动态链接库。

具体来说，可以执行下面的命令：

```
export LD_LIBRARY_PATH="{path}/oneflow/build/oneflow/api/java:$LD_LIBRARY_PATH"
```

## 使用 Java 版本的 OneFlow

从 Github 上面下载 [release](https://github.com/zzk0/oneflow-java/releases/download/v1.0.0-alpha/liboneflow-1.0.0-alpha.zip)

解压，里面有一个 jar 包。这个 jar 包依赖了 `com.google.protobuf` 使用的时候，也需要引入。下面给出 Maven 中的使用办法。

第一，将 liboneflow-1.0.0-alpha.jar 安装到本地 Maven 仓库。

```
mvn install:install-file -DgroupId=org.oneflow -DartifactId=liboneflow -Dversion=1.0.0-alpha -Dpackaging=jar -Dfile={path}\liboneflow-1.0.0-alpha.jar
```

第二，在 Maven 中引入如下依赖。

```
<dependency>
    <groupId>org.oneflow</groupId>
    <artifactId>liboneflow</artifactId>
    <version>1.0.0-alpha</version>
</dependency>

<dependency>
    <groupId>com.google.protobuf</groupId>
    <artifactId>protobuf-java</artifactId>
    <version>3.15.3</version>
</dependency>
```

开始使用。

# 开发说明

## protobuf

因为 OneFlow 采用了 ProtoBuf 来做序列化，OneFlow 和 Python 之间的信息交换使用 ProtoBuf 来实现，模型也是使用 ProtoBuf 来保存。因此在 Java 版本中，同样采用 ProtoBuf 来进行信息交换。

为此，我们需要将 OneFlow 中定义的 .proto 文件构建为 .java 文件。我编写了一个脚本 `build_proto.py` 帮助我进行了构建。如果想要运行这个脚本，需要拉取 submodule。

构建之前，需要手动修改 `oneflow/core/framework/user_op_conf.proto` 的 `input` 为 `user_input`，`output` 为 `user_output`。为什么这么做？因为在一个 .proto 的 message 中定义了 `input` 和 `output` 这两个名字，由 ProtoBuf 生成的 .java 文件也会有这两个名字，而且正好和生成的局部变量名称撞上了，这导致了 Java 编译失败，所以没办法，只能改 message。这应该属于 ProtoBuf 的锅。

## 测试

测试用例呢？在 test 分支。

因为开发环境的限制，这个项目采用的是本地开发 Java，远程运行 jar 包，IDEA 通过监听远程端口来实现调试。

如果使用 Junit 来写测试用例，不太方便调试。所以，新建了一个 test 分支来写测试。后期折腾一下，看看如何将 Junit 的测试打包，并且运行和调试指定测试用例。

## OneFlow 版本

OneFlow 版本 0.3.5。

在 4.0.0 版本中，有些类和接口发生了改变，构建直接失败了。比如，`foreign_job_instance` 改了个名字。对于这个问题，目前没有什么比较好的解决办法，只能是多多跟进。

# 其他

有待完善中...
