package org.oneflow;

import java.util.Map;

/**
 * All functions in one class
 */
public class InferenceSession {

    static {
        System.loadLibrary("oneflow");
    }

    private Option option;
    private SessionStateState state;

    public InferenceSession() {
        this.option = new Option();
        this.state = SessionStateState.CLOSE;
    }

    public InferenceSession(Option option) {
        this.option = option;
        this.state = SessionStateState.CLOSE;
    }

    public void open() {

    }

    public void loadSavedModel(String path) {

    }

    public void launch() {

    }

    public Tensor[] run(String jobName, Map<String, Tensor> tensors) {
        return null;
    }

    public void close() {

    }

    // ----------------------- The methods below will be removed -----------------------

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

    // forward
    public static native void runPushJob(float[] arr);
    public static native void runInferenceJob();
    public static native void runPullJob();

    // clean
    public static native void stopLazyGlobalSession();
    public static native void destroyLazyGlobalSession();
    public static native void destroyEnv();
    public static native void setShuttingDown();
}
