package org.oneflow;

import com.google.protobuf.InvalidProtocolBufferException;
import org.oneflow.core.job.Env;
import org.oneflow.core.job.Env.EnvProto;
import org.oneflow.core.job.InterUserJobInfoOuterClass.InterUserJobInfo;
import org.oneflow.core.job.JobConf;
import org.oneflow.core.job.JobConf.JobConfigProto;
import org.oneflow.core.serving.SavedModelOuterClass.SavedModel;
import org.oneflow.core.serving.SavedModelOuterClass.GraphDef;
import org.oneflow.core.operator.OpConf.OperatorConf;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;


/**
 * All in one
 */
public class App {
    public static void main(String[] args) {
        // ------------------ [User Configuration Start] ------------------
        String jobName = "mlp_inference";
        String savedModelDir = "./models";
        float[] image = readImage("./7.png");
        Tensor imageTensor = Tensor.fromBlob(image, new long[]{ 1, 1, 28, 28 });
        Tensor tagTensor = Tensor.fromBlob(new int[]{ 1 }, new long[]{ 1 });
        Map<String, Tensor> tensorMap = new HashMap<>();
        tensorMap.put("Input_14", imageTensor);  // Todo: support different signature
        tensorMap.put("Input_15", tagTensor);
        // ------------------ [User Configuration End] ------------------

        // ------------------ [Default Init Stage Start] ------------------
        InferenceSession.initDefaultSession();
        // ------------------ [Default Init Stage End] ------------------

        // ------------------ [Init Stage Start] ------------------
        // 1, env init
        if (!InferenceSession.isEnvInited()) {
            doEnvInit(8888);
        }
        if (InferenceSession.isEnvInited()) {
            System.out.println("env is inited");
        }
        else {
            System.out.println("env is not inited and the program will exit");
            System.exit(-1);
        }

        // 2, scope init
        InferenceSession.initScopeStack();

        // 3, session init
        if (!InferenceSession.isSessionInited()) {
            InferenceSession.initSession();
        }
        if (InferenceSession.isSessionInited()) {
            System.out.println("session is inited");
        }
        else {
            System.out.println("session is not inited and the program will exit");
            System.exit(-1);
        }
        // ------------------ [Init Stage End] ------------------

        // ------------------ [Load Computation Graph Stage Start] ------------------
        String savedModelPath = savedModelDir + "/1/";  // Todo: version
        File file = new File(savedModelPath + "saved_model.pb");  // Todo: support different format
        SavedModel model = SavedModel.newBuilder()
                .setName("")
                .setVersion(1)
                .setCheckpointDir("")
                .build();
        try (InputStream fis = new FileInputStream(file)) {
            model = SavedModel.parseFrom(fis);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        String checkpointPath = savedModelPath + model.getCheckpointDir();
        String graphName = model.getDefaultGraphName();
        GraphDef graphDef = model.getGraphsOrThrow(graphName);
        System.out.println(checkpointPath);
        System.out.println(graphName);
        // ------------------ [Load Computation Graph Stage End] ------------------

        // ------------------ [Compile Computation Graph Stage Start] ------------------
        // 1, prepare environment
        InferenceSession.openJobBuildAndInferCtx(jobName);
        JobConfigProto jobConfigProto = JobConfigProto.newBuilder()
                .setJobName(jobName)
                .setPredictConf(JobConf.PredictConf.newBuilder().build())
                .build();
//        System.out.println(jobConfigProto.toString());
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
        // ------------------ [Compile Computation Graph Stage End] ------------------

        // ------------------ [Launch Stage Start] ------------------
        InferenceSession.startLazyGlobalSession();

        String interUserJobInfo = InferenceSession.getInterUserJobInfo();
        InterUserJobInfo info = null;
        try {
            info = InterUserJobInfo.parseFrom(interUserJobInfo.getBytes());
        } catch (InvalidProtocolBufferException e) {
            e.printStackTrace();
        }
        if (info == null) {
            System.out.println("info is null");
            return;
        }

        InferenceSession.loadCheckpoint(info.getGlobalModelLoadJobName(), checkpointPath.getBytes());
        // ------------------ [Launch Stage End] ------------------

        // ------------------ [Forward Stage 1: Push Start] ------------------
        Map<String, String> inputNameToJobName = info.getInputOrVarOpName2PushJobNameMap();
        for (Map.Entry<String, String> entry : inputNameToJobName.entrySet()) {
            System.out.println(entry.getKey() + " " + entry.getValue());
            Tensor tensor = tensorMap.get(entry.getKey());
            byte[] tensorBytes = tensor.getBytes();
            InferenceSession.runSinglePushJob(tensorBytes, tensor.getShape(),
                    tensor.getDataType().code, entry.getValue(), entry.getKey());
        }

        // ------------------ [Forward Stage 1: Push End] ------------------

        // ------------------ [Forward Stage 2: Inference Start] ------------------
        InferenceSession.runInferenceJob(jobName);
        // ------------------ [Forward Stage 2: Inference End] ------------------

        // ------------------ [Forward Stage 3: Pull Start] ------------------
        long curTime = System.currentTimeMillis();
        for (int i = 0; i < 10000; i++) {
            for (Map.Entry<String, String> entry : info.getOutputOrVarOpName2PullJobNameMap().entrySet()) {
                Tensor res = InferenceSession.runPullJob(entry.getValue(), entry.getKey());
//                float[] pred = res.getDataAsFloatArray();
//                for (float x : pred) {
//                    System.out.print(x + " ");
//                }
//                System.out.println();
            }
        }
        System.out.printf("It takes %d ms to forward %d times%n",
                System.currentTimeMillis() - curTime,
                5);
        // ------------------ [Forward Stage 3: Pull End] ------------------

        // ------------------ [Clean Stage Start] ------------------
        InferenceSession.stopLazyGlobalSession();
        InferenceSession.destroyLazyGlobalSession();
        InferenceSession.destroyEnv();
        InferenceSession.setShuttingDown();
        // ------------------ [Clean Stage End] ------------------
    }

    public static void doEnvInit(int port) {
        // reference: env_util.py 365 line
        EnvProto envProto = EnvProto.newBuilder()
                .addMachine(Env.Machine.newBuilder()
                        .setId(0)
                        .setAddr("127.0.0.1"))
                .setCtrlPort(port)
                .build();
        InferenceSession.initEnv(envProto.toString());
    }

    public static float[] readImage(String filePath) {
        File file = new File(filePath);
        BufferedImage image = null;
        try {
            image = ImageIO.read(file);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        assert image != null;

        int width = image.getWidth();
        int height = image.getHeight();
        Raster raster = image.getRaster();
        float[] pixels = new float[width * height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                // Caution: transform the image
                pixels[i * width + j] = (raster.getSample(j, i, 0) - 128.0f) / 255.0f;
            }
        }
        return pixels;
    }
}

// Todo: some deleted code, do we need to set device tag to "gpu"?
// line: 294 ~ 303, it doesn't  do that, so we don't need to
//
//            System.out.println(conf.toString());
//            OpConf.OperatorConf.Builder builder = OperatorConf.newBuilder();
//            builder.setName(conf.getName());
//            builder.setDeviceTag("gpu");
//            builder.setScopeSymbolId(scopeSymbolId);
//
//            if (!conf.getVariableConf().toString().isEmpty()) {
//                builder.setVariableConf(conf.getVariableConf());
//            }
//            else if (!conf.getUserConf().toString().isEmpty()) {
//                builder.setUserConf(conf.getUserConf());
//            }
//            else if (!conf.getInputConf().toString().isEmpty()) {
//                builder.setInputConf(conf.getInputConf());
//            }
//            else if (!conf.getOutputConf().toString().isEmpty()) {
//                builder.setOutputConf(conf.getOutputConf());
//            }
//            else if (!conf.getReturnConf().toString().isEmpty()) {
//                builder.setReturnConf(conf.getReturnConf());
//            }
//
//            Library.curJobAddOp(builder.build().toString());

// Todo: new interface
//
//        InferenceSession session = new InferenceSession();
//        session.open();
//        session.loadSavedModel("./models/1/");
//        session.launch();
//
//        // input
//        float[] image = readImage("./7.png");
//        Tensor tensor = Tensor.fromBlob(new int[]{ 0 }, new long[]{ 1 }, DType.INT);
//        Tensor imageTensor = Tensor.fromBlob(image, new long[]{ 28, 28 }, DType.FLOAT);
//        Map<String, Tensor> tensors = new HashMap<>();
//        tensors.put("Input_14", imageTensor);
//        tensors.put("Input_15", tensor);
//
//        // forward
//        Tensor[] result = session.run("mlp_inference", tensors);
//        Tensor prediction = result[0];
//        float[] vector = prediction.getFloatData();
//
//        // close
//        // session.close();
//
//        // assert
//        assert (10 == vector.length);
//        float[] expectedVector = { -129.57167f, -89.084816f, -139.21355f , -103.455025f, -9.179366f,
//                -69.568474f, -133.39594f,  -16.204329f, -114.90876f,  -47.933548f };
//        float delta = 0.0001f;
//        for (int i = 0; i < 10; i++) {
//            assert (Math.abs(expectedVector[i] - vector[i]) < delta);
//        }