package org.oneflow;

import org.oneflow.core.job.Env.EnvProto;
import org.oneflow.core.job.Env.Machine;
import org.oneflow.core.job.JobConf.PredictConf;
import org.oneflow.core.job.JobConf.JobConfigProto;
import org.oneflow.core.operator.OpConf.OperatorConf;
import org.oneflow.core.serving.SavedModelOuterClass.SavedModel;
import org.oneflow.core.serving.SavedModelOuterClass.GraphDef;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Map;


public class InferenceSession {

    static {
        System.loadLibrary("oneflow");
    }
//    private static boolean defaultInit = false;
//
//    private final Option option;
//    private SessionStateState state;
//
//    public InferenceSession() {
//        this.option = new Option();
//        this.state = SessionStateState.CLOSE;
//    }
//
//    public InferenceSession(Option option) {
//        this.option = option;
//        this.state = SessionStateState.CLOSE;
//    }
//
//    public void open() {
//        if (!defaultInit) {
//            defaultInit = true;
//            initDefaultSession();
//        }
//
//        if (!isEnvInited()) {
//            doEnvInit();
//            initScopeStack();
//        }
//
//        if (!isSessionInited()) {
//            initSession();
//        }
//
//        state = SessionStateState.OPEN;
//    }
//
//    public void loadSavedModel(String path) {
//        File file = new File(path + "saved_model.pb");
//        SavedModel model = SavedModel.newBuilder()
//                .setName("")
//                .setVersion(1)
//                .setCheckpointDir("")
//                .build();
//        try (InputStream fis = new FileInputStream(file)) {
//            model = SavedModel.parseFrom(fis);
//        }
//        catch (Exception e) {
//            e.printStackTrace();
//        }
//        String graphName = model.getDefaultGraphName();
//        GraphDef graphDef = model.getGraphsOrThrow(graphName);
//
//        // 1, prepare environment
//        String jobName = "mlp_inference";
//        openJobBuildAndInferCtx(jobName);
//        JobConfigProto jobConfigProto = JobConfigProto.newBuilder()
//                .setJobName(jobName)
//                .setPredictConf(PredictConf.newBuilder().build())
//                .build();
//        setJobConfForCurJobBuildAndInferCtx(jobConfigProto.toString());
//        setScopeForCurJob();
//
//        // 2, do the compilation
//        for (OperatorConf conf : graphDef.getOpListList()) {
//            curJobAddOp(conf.toString());
//        }
//        completeCurJobBuildAndInferCtx();
//        rebuildCurJobBuildAndInferCtx();
//
//        // 3, clean the environment
//        unsetScopeForCurJob();
//        closeJobBuildAndInferCtx();
//    }
//
//    public void launch() {
//        startLazyGlobalSession();
//        loadCheckpoint();
//
//        state = SessionStateState.RUNNING;
//    }
//
//    public Tensor[] run(String jobName, Map<String, Tensor> tensors) {
//        float[] image = tensors.get("Input_14").getFloatData();
//        runPushJob(image);
//        runInferenceJob();
//        runPullJob();
//        return null;
//    }
//
//    public void close() {
//        stopLazyGlobalSession();
//        destroyLazyGlobalSession();
//        destroyEnv();
//        setShuttingDown();
//
//        state = SessionStateState.CLOSE;
//    }
//
//    private void doEnvInit() {
//        EnvProto envProto = EnvProto.newBuilder()
//                .addMachine(Machine.newBuilder()
//                        .setId(0)
//                        .setAddr("127.0.0.1"))
//                .setCtrlPort(option.getControlPort())
//                .build();
//        initEnv(envProto.toString());
//    }

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
    public static native void setScopeForCurJob(String jobConfProto);
    public static native void curJobAddOp(String opConfProto);
    public static native void completeCurJobBuildAndInferCtx();
    public static native void rebuildCurJobBuildAndInferCtx();
    public static native void unsetScopeForCurJob();
    public static native void closeJobBuildAndInferCtx();

    // launch
    public static native void startLazyGlobalSession();
    public static native void loadCheckpoint(String jobName, byte[] path);

    // forward
    public static native void runPushJob(float[] arr);
    public static native void runSinglePushJob(byte[] data,
                                               long[] shape,
                                               int dTypeCode,
                                               String job_name,
                                               String op_name);
    public static native void runInferenceJob();
    public static native void runPullJob();

    // clean
    public static native void stopLazyGlobalSession();
    public static native void destroyLazyGlobalSession();
    public static native void destroyEnv();
    public static native void setShuttingDown();

    // others
    public static native String getInterUserJobInfo();
}
