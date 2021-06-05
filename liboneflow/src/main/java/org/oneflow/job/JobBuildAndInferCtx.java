package org.oneflow.job;

public class JobBuildAndInferCtx {
    static {
        System.loadLibrary("oneflow");
    }

    public static native void open(String jobName);
    public static native String getCurrentJobName();
}