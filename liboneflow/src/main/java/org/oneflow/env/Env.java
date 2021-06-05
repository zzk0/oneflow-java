package org.oneflow.env;

public class Env {
    static {
        System.loadLibrary("oneflow");
    }

    public static native boolean isEnvInited();
    public static native void initEnv(String envProto);
}