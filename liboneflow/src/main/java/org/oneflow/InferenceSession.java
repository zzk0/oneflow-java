package org.oneflow;

import com.google.protobuf.InvalidProtocolBufferException;
import org.oneflow.core.job.Env;
import org.oneflow.core.job.Env.EnvProto;
import org.oneflow.core.job.InterUserJobInfoOuterClass.InterUserJobInfo;
import org.oneflow.core.job.JobConf;
import org.oneflow.core.job.JobConf.JobConfigProto;
import org.oneflow.core.operator.OpConf.OperatorConf;
import org.oneflow.core.serving.SavedModelOuterClass.SavedModel;
import org.oneflow.core.serving.SavedModelOuterClass.GraphDef;
import org.oneflow.exception.CheckNullException;
import org.oneflow.exception.InitializationException;
import org.oneflow.util.ConfigConst;

import java.io.*;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;


public class InferenceSession {

    static {
        System.loadLibrary("oneflow");

        // Default Initialization: beyond import oneflow as flow
        InferenceSession.initDefaultSession();
        if (getEndian() == 0) {
            Tensor.endian = ByteOrder.BIG_ENDIAN;
        }
    }

    private final int port;
    private String checkpointPath;
    private InterUserJobInfo interUserJobInfo;

    public InferenceSession() {
        port = ConfigConst.PORT;
    }

    public InferenceSession(int port) {
        this.port = port;
    }

    /**
     * Init the Env and Session
     */
    public void open() {
        // 1, env init
        if (!InferenceSession.isEnvInited()) {
            doEnvInit(this.port);

            // 2, scope init
            InferenceSession.initScopeStack();
        }
        if (!InferenceSession.isEnvInited()) {
            throw new InitializationException("Env is not inited correctly");
        }

        // 3, session init
        if (!InferenceSession.isSessionInited()) {
            InferenceSession.initSession();
        }
        if (!InferenceSession.isSessionInited()) {
            throw new InitializationException("Session is not inited correctly");
        }
    }

    /**
     * try search the .pb/.prototxt file from given path and load it
     * @param path
     */
    public void loadModel(String path) {
        String savedModelPath = path + "/1/";  // Todo: support different model version
        File file = new File(savedModelPath + "saved_model.pb");  // Todo: support different format
        SavedModel model = SavedModel.newBuilder()
                .setName("")
                .setVersion(1)
                .setCheckpointDir("")
                .build();
        try (InputStream fis = new FileInputStream(file)) {
            model = SavedModel.parseFrom(fis);
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        this.checkpointPath= savedModelPath + File.separator + model.getCheckpointDir();

        // [Compile]
        String graphName = model.getDefaultGraphName();
        GraphDef graphDef = model.getGraphsOrThrow(graphName);

        // 1, prepare environment
        InferenceSession.openJobBuildAndInferCtx(graphName);
        JobConfigProto jobConfigProto = JobConfigProto.newBuilder()
                .setJobName(graphName)
                .setPredictConf(JobConf.PredictConf.newBuilder().build())
                .build();
        InferenceSession.setJobConfForCurJobBuildAndInferCtx(jobConfigProto.toString());
        InferenceSession.setScopeForCurJob(jobConfigProto.toString());

        // 2, do the compilation
        for (OperatorConf conf : graphDef.getOpListList()) {
            InferenceSession.curJobAddOp(conf.toString());
        }
        InferenceSession.completeCurJobBuildAndInferCtx();
        InferenceSession.rebuildCurJobBuildAndInferCtx();

        // 3, clean the environment
        InferenceSession.unsetScopeForCurJob();
        InferenceSession.closeJobBuildAndInferCtx();
    }

    public void launch() {
        InferenceSession.startLazyGlobalSession();

        String interUserJobInfo = InferenceSession.getInterUserJobInfo();
        InterUserJobInfo info = null;
        try {
            info = InterUserJobInfo.parseFrom(interUserJobInfo.getBytes());
        } catch (InvalidProtocolBufferException e) {
            e.printStackTrace();
        }
        if (info == null) {
            throw new CheckNullException("GetInterUserJobInfo failed");
        }

        this.interUserJobInfo = info;
        byte[] checkpointBytes = checkpointPath.getBytes();
        ByteBuffer checkpointBuffer = ByteBuffer.allocateDirect(checkpointBytes.length);
        checkpointBuffer.put(checkpointBytes);
        InferenceSession.loadCheckpoint(info.getGlobalModelLoadJobName(), checkpointBuffer);
    }

    public Map<String, Tensor> run(String jobName, Map<String, Tensor> tensorMap) {
        // Push
        Map<String, String> inputNameToJobName = interUserJobInfo.getInputOrVarOpName2PushJobNameMap();
        for (Map.Entry<String, String> entry : inputNameToJobName.entrySet()) {
            Tensor tensor = tensorMap.get(entry.getKey());
            Buffer dataBuffer = tensor.getDataBuffer();

            InferenceSession.runSinglePushJob(dataBuffer, tensor.getShapeBuffer(),
                    tensor.getDataType().code, entry.getValue(), entry.getKey());
        }

        // Inference
        InferenceSession.runInferenceJob(jobName);

        // Pull
        Map<String, Tensor> resultMap = new HashMap<>();
        for (Map.Entry<String, String> entry : interUserJobInfo.getOutputOrVarOpName2PullJobNameMap().entrySet()) {
            Tensor res = InferenceSession.runPullJob(entry.getValue(), entry.getKey());
            resultMap.put(entry.getKey(), res);
        }

        return resultMap;
    }

    public void close() {
        InferenceSession.stopLazyGlobalSession();
        InferenceSession.destroyLazyGlobalSession();
        InferenceSession.destroyEnv();
        InferenceSession.setShuttingDown();
    }

    private static void doEnvInit(int port) {
        // reference: env_util.py 365 line
        EnvProto envProto = EnvProto.newBuilder()
                .addMachine(Env.Machine.newBuilder()
                        .setId(0)
                        .setAddr("127.0.0.1"))
                .setCtrlPort(port)
                .build();
        InferenceSession.initEnv(envProto.toString());
    }

    // 0 for big endian, 1 for little endian
    private static native int getEndian();

    // init
    private static native void initDefaultSession();
    private static native boolean isEnvInited();
    private static native void initEnv(String envProto);
    private static native void initScopeStack();
    private static native boolean isSessionInited();
    private static native void initSession();

    // compile
    private static native void openJobBuildAndInferCtx(String jobName);
    private static native void setJobConfForCurJobBuildAndInferCtx(String jobConfProto);
    private static native void setScopeForCurJob(String jobConfProto);
    private static native void curJobAddOp(String opConfProto);
    private static native void completeCurJobBuildAndInferCtx();
    private static native void rebuildCurJobBuildAndInferCtx();
    private static native void unsetScopeForCurJob();
    private static native void closeJobBuildAndInferCtx();

    // launch
    private static native void startLazyGlobalSession();
    private static native void loadCheckpoint(String jobName, Buffer path);

    // forward
    private static native void runSinglePushJob(Buffer data,
                                               Buffer shape,
                                               int dTypeCode,
                                               String jobName,
                                               String opName);
    private static native void runInferenceJob(String jobName);
    private static native Tensor runPullJob(String jobName, String opName);

    // clean
    private static native void stopLazyGlobalSession();
    private static native void destroyLazyGlobalSession();
    private static native void destroyEnv();
    private static native void setShuttingDown();

    // others
    private static native String getInterUserJobInfo();
}
