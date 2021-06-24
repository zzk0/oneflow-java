package org.oneflow;

import org.oneflow.core.job.Env;
import org.oneflow.core.job.Env.EnvProto;
import org.oneflow.core.job.JobConf;
import org.oneflow.core.job.JobConf.JobConfigProto;
import org.oneflow.core.operator.OpConf;
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


/**
 * All in one
 */
public class App {
    public static void main(String[] args) {
        // ------------------ [Default Init Stage Start] ------------------
        Library.initDefaultSession();
        // ------------------ [Default Init Stage End] ------------------

        // ------------------ [Init Stage Start] ------------------
        // 1, env init
        if (!Library.isEnvInited()) {
            doEnvInit();
        }
        if (Library.isEnvInited()) {
            System.out.println("env is inited");
        }
        else {
            System.out.println("env is not inited and the program will exit");
            System.exit(-1);
        }

        // 2, scope init
        Library.initScopeStack();

        // 3, session init
        if (!Library.isSessionInited()) {
            Library.initSession();
        }
        if (Library.isSessionInited()) {
            System.out.println("session is inited");
        }
        else {
            System.out.println("session is not inited and the program will exit");
            System.exit(-1);
        }
        // ------------------ [Init Stage End] ------------------

        // ------------------ [Load Computation Graph Stage Start] ------------------
        String savedModelPath = "./models/1/";
        File file = new File(savedModelPath + "saved_model.pb");
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
        String jobName = "mlp_inference";
        Library.openJobBuildAndInferCtx(jobName);
        JobConfigProto jobConfigProto = JobConfigProto.newBuilder()
                .setJobName(jobName)
                .setPredictConf(JobConf.PredictConf.newBuilder().build())
                .build();
//        System.out.println(jobConfigProto.toString());
        Library.setJobConfForCurJobBuildAndInferCtx(jobConfigProto.toString());
        Library.setScopeForCurJob();

        // 2, do the compilation
        for (OperatorConf conf : graphDef.getOpListList()) {
            Library.curJobAddOp(conf.toString());
        }
        Library.completeCurJobBuildAndInferCtx();
        Library.rebuildCurJobBuildAndInferCtx();

        // 3, clean the environment
        Library.unsetScopeForCurJob();
        Library.closeJobBuildAndInferCtx();
        // ------------------ [Compile Computation Graph Stage End] ------------------

        // ------------------ [Launch Stage Start] ------------------
        Library.startLazyGlobalSession();
        Library.loadCheckpoint();
        // ------------------ [Launch Stage End] ------------------

        // ------------------ [Forward Stage 1: Push Start] ------------------
        float[] image = readImage("./7.png");
        Library.runPushJob(image);
        // ------------------ [Forward Stage 1: Push End] ------------------

        // ------------------ [Forward Stage 2: Inference Start] ------------------
        Library.runInferenceJob();
        // ------------------ [Forward Stage 2: Inference End] ------------------

        // ------------------ [Forward Stage 3: Pull Start] ------------------
        Library.runPullJob();
        // ------------------ [Forward Stage 3: Pull End] ------------------

        // ------------------ [Clean Stage Start] ------------------
//        Library.stopLazyGlobalSession();
//        Library.destroyLazyGlobalSession();
//        Library.destroyEnv();
//        Library.setShuttingDown();
        // ------------------ [Clean Stage End] ------------------

        System.out.println("pause");
    }

    public static void doEnvInit() {
        // reference: env_util.py 365 line
        EnvProto envProto = EnvProto.newBuilder()
                .addMachine(Env.Machine.newBuilder()
                        .setId(0)
                        .setAddr("127.0.0.1"))
                .setCtrlPort(8888)
                .build();
        Library.initEnv(envProto.toString());
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

// some deleted code
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
