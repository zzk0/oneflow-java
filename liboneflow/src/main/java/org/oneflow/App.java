package org.oneflow;

import org.oneflow.core.job.Env;
import org.oneflow.core.job.Env.EnvProto;
import org.oneflow.core.serving.SavedModelOuterClass;
import org.oneflow.core.serving.SavedModelOuterClass.SavedModel;
import org.oneflow.core.serving.SavedModelOuterClass.GraphDef;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;


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
        SavedModel model = SavedModel.newBuilder().build();
        try (InputStream fis = new FileInputStream(file)) {
            model = SavedModel.parseFrom(fis);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        String checkpointPath = savedModelPath + model.getCheckpointDir();
        String graphName = model.getDefaultGraphName();
        GraphDef graphDef = model.getGraphsOrThrow(graphName);
        // ------------------ [Load Computation Graph Stage End] ------------------

        // ------------------ [Compile Computation Graph Stage Start] ------------------
        // 1, prepare environment
        // 2, do the compilation
        // 3, clean the environment
        // ------------------ [Compile Computation Graph Stage End] ------------------
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
}
