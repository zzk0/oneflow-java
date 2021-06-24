package org.oneflow;

/**
 * All functions in one class
 */
public class Library {

    static {
        System.loadLibrary("oneflow");
    }

    // init
    public static native void initDefaultSession();
    public static native boolean isEnvInited();
    public static native void initEnv(String envProto);
    public static native void initScopeStack();
    public static native boolean isSessionInited();
    public static native void initSession();

    // compile
    public static native void openJobBuildAndInferCtx(String jobName);
    public static native void setJobConfForCurJobBuildAndInferCtx(String jobConfProto);
    public static native void setScopeForCurJob();
    public static native void curJobAddOp(String opConfProto);
    public static native void completeCurJobBuildAndInferCtx();
    public static native void rebuildCurJobBuildAndInferCtx();
    public static native void unsetScopeForCurJob();
    public static native void closeJobBuildAndInferCtx();

    // launch
    public static native void startLazyGlobalSession();
    public static native void loadCheckpoint();
}
