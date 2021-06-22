package org.oneflow;

/**
 * All functions in one class
 */
public class Library {

    static {
        System.loadLibrary("oneflow");
    }

    public static native void initDefaultSession();
    public static native boolean isEnvInited();
    public static native void initEnv(String envProto);
    public static native void initScopeStack();
    public static native boolean isSessionInited();
    public static native void initSession();
}
