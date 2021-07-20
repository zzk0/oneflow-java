package org.oneflow;

import com.google.protobuf.InvalidProtocolBufferException;
import org.oneflow.core.job.Env;
import org.oneflow.core.job.Env.EnvProto;
import org.oneflow.core.job.InterUserJobInfoOuterClass.InterUserJobInfo;
import org.oneflow.core.job.JobConf;
import org.oneflow.core.job.JobConf.JobConfigProto;
import org.oneflow.core.job.JobSetOuterClass.ConfigProto;
import org.oneflow.core.job.ResourceOuterClass;
import org.oneflow.core.operator.OpConf.OperatorConf;
import org.oneflow.core.serving.SavedModelOuterClass.SavedModel;
import org.oneflow.core.serving.SavedModelOuterClass.GraphDef;
import org.oneflow.exception.CheckNullException;
import org.oneflow.exception.InitializationException;

import java.io.*;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.Map;


public class InferenceSession {

    private final Option option;
    private String checkpointPath;
    private InterUserJobInfo interUserJobInfo;

    public InferenceSession() {
        this.option = new Option();
    }

    public InferenceSession(Option option) {
        this.option = option;
    }

    /**
     * Init the Env and Session
     */
    public void open() {
        OneFlow.setIsMultiClient(false);

        // 1, env init
        if (!OneFlow.isEnvInited()) {
            doEnvInit(this.option.getControlPort());

            // 2, scope init
            OneFlow.initScopeStack();
        }
        if (!OneFlow.isEnvInited()) {
            throw new InitializationException("Env is not inited correctly");
        }

        // 3, session init
        if (!OneFlow.isSessionInited()) {
            ConfigProto configProto;
            if ("gpu".equals(option.getDeviceTag())) {
                configProto = ConfigProto.newBuilder()
                        .setResource(ResourceOuterClass.Resource.newBuilder()
                                .setMachineNum(option.getDeviceNum())
                                .setGpuDeviceNum(option.getDeviceNum())
                                .setEnableLegacyModelIo(true)
                                .build())
                        .setSessionId(0)
                        .build();
            }
            else {
                configProto = ConfigProto.newBuilder()
                        .setResource(ResourceOuterClass.Resource.newBuilder()
                                .setMachineNum(OneFlow.getNodeSize())
                                .setCpuDeviceNum(option.getDeviceNum())
                                .setEnableLegacyModelIo(true)
                                .build())
                        .setSessionId(0)
                        .build();
            }

            OneFlow.initSession(configProto.toString());
        }
        if (!OneFlow.isSessionInited()) {
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
        OneFlow.openJobBuildAndInferCtx(graphName);
        JobConfigProto jobConfigProto = JobConfigProto.newBuilder()
                .setJobName(graphName)
                .setPredictConf(JobConf.PredictConf.newBuilder().build())
                .build();
        OneFlow.setJobConfForCurJobBuildAndInferCtx(jobConfigProto.toString());

        // Todo: device_id_tags
        OneFlow.setScopeForCurJob(jobConfigProto.toString(), "0:0", option.getDeviceTag());

        // 2, do the compilation
        for (OperatorConf conf : graphDef.getOpListList()) {
            OneFlow.curJobAddOp(conf.toString());
        }
        OneFlow.completeCurJobBuildAndInferCtx();
        OneFlow.rebuildCurJobBuildAndInferCtx();

        // 3, clean the environment
        OneFlow.unsetScopeForCurJob();
        OneFlow.closeJobBuildAndInferCtx();
    }

    public void launch() {
        OneFlow.startLazyGlobalSession();

        String interUserJobInfo = OneFlow.getInterUserJobInfo();
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
        OneFlow.loadCheckpoint(info.getGlobalModelLoadJobName(), checkpointBuffer);
    }

    public Map<String, Tensor> run(String jobName, Map<String, Tensor> tensorMap) {
        // Push
        Map<String, String> inputNameToJobName = interUserJobInfo.getInputOrVarOpName2PushJobNameMap();
        for (Map.Entry<String, String> entry : inputNameToJobName.entrySet()) {
            Tensor tensor = tensorMap.get(entry.getKey());
            Buffer dataBuffer = tensor.getDataBuffer();

            OneFlow.runSinglePushJob(dataBuffer, tensor.getShapeBuffer(),
                    tensor.getDataType().code, entry.getValue(), entry.getKey());
        }

        // Inference
        OneFlow.runInferenceJob(jobName);

        // Pull
        Map<String, Tensor> resultMap = new HashMap<>();
        for (Map.Entry<String, String> entry : interUserJobInfo.getOutputOrVarOpName2PullJobNameMap().entrySet()) {
            Tensor res = OneFlow.runPullJob(entry.getValue(), entry.getKey());
            resultMap.put(entry.getKey(), res);
        }

        return resultMap;
    }

    public void close() {
        OneFlow.stopLazyGlobalSession();
        OneFlow.destroyLazyGlobalSession();
        OneFlow.destroyEnv();
        OneFlow.setShuttingDown();
    }

    private static void doEnvInit(int port) {
        // reference: env_util.py 365 line
        EnvProto envProto = EnvProto.newBuilder()
                .addMachine(Env.Machine.newBuilder()
                        .setId(0)
                        .setAddr("127.0.0.1"))
                .setCtrlPort(port)
                .build();
        OneFlow.initEnv(envProto.toString());
    }

}
